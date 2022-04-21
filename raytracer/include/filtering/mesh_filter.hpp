#ifndef MESH_FILTER_HPP
#define MESH_FILTER_HPP
#include <glm/glm.hpp>
#include "intersect/aabb.hpp"
#include "utility/qualifiers.hpp"
#include "sampling/sampler.hpp"
#include "scene/device_scene_impl.hpp"
#include "intersect/ray.hpp"
#include "filtered_data.hpp"
#include "material/material.hpp"
#include "shapes/sphere.hpp"
#include "medium/sggx_phase_function.hpp"

// Includes for debug below
#include "scene/device_sceneobject.hpp"

namespace filter {

  class CMeshFilter {
  public:
    DH_CALLABLE CMeshFilter(const glm::ivec3& currentVoxel,
                            const glm::mat4x3& indexToModel,
                            const glm::mat4x3& modelToIndex,
                            const glm::mat4x3& modelToWorld,
                            const glm::mat4x3& worldToModel,
                            const glm::ivec3& numVoxels,
                            const rt::SAABB& worldBB,
                            rt::CSampler& sampler,
                            float sigma_t,
                            uint32_t estimationIterations,
                            float alpha,
                            bool clipRays,
                            float scaling):
      m_currentVoxel(currentVoxel),
      m_indexToModel(indexToModel),
      m_modelToIndex(modelToIndex),
      m_modelToWorld(modelToWorld),
      m_worldToModel(worldToModel),
      m_numVoxels(numVoxels),
      m_worldBB(worldBB),
      m_sampler(sampler),
      m_sigma_t(sigma_t),
      m_estimationIterations(estimationIterations),
      m_alpha(alpha),
      m_clipRays(clipRays),
      m_scaling(scaling) {
      m_voxelCenter = modelToWorld * glm::vec4(indexToModel * glm::vec4(glm::vec3(currentVoxel) + 0.5f, 1.f), 1.f); // +0.5 to get voxel center in world space
      m_voxelSize = (worldBB.m_max - worldBB.m_min) / glm::vec3(numVoxels);
    }

    D_CALLABLE SFilteredDataCompact run(rt::CDeviceScene& scene, uint32_t numSamples) {
      glm::vec3 currentVoxel(m_currentVoxel);
      glm::mat4x3 indexToWorld = glm::mat4(m_modelToWorld) * glm::mat4(m_indexToModel);
      float diagonal = glm::length(m_voxelSize);
      float tMax = m_voxelSize.x; // voxelSize in Loubet, Neyret "Hybrid mesh-volume LoDs for all-scale pre-filtering of complex 3D assets"
      float volume = (m_worldBB.m_max.x - m_worldBB.m_min.x) * (m_worldBB.m_max.y - m_worldBB.m_min.y) * (m_worldBB.m_max.z - m_worldBB.m_min.z);

      uint32_t hits = 0;
      glm::mat3 filteredS(0.f);
      glm::vec3 filteredDiffuseClr(0.f);
      glm::vec3 filteredSpecularClr(0.f);
      glm::vec3 filteredNormal(0.f);
      float specularRoughness = 0.f;
      float rayDistance = 0.f;
      rt::Sphere sphere(m_voxelCenter, 1.1f * diagonal / 2.f, glm::vec3(1.f, 0.f, 0.f)); // Slightly larger than Voxel
      float rayTs[1000] = {};
      float averageTMax = 0.f;
      float ior = 0.f;
      for (uint32_t i = 0; i < numSamples; ++i) {
        rt::CRay ray;
        float currentTMax;
        if (m_clipRays) {
          ray = generateRay(currentVoxel, indexToWorld, rt::CRay::DEFAULT_TMAX);
          rt::SHitInformation hit = sphere.intersect(ray);
          ray.m_t_max = hit.t;
          currentTMax = hit.t;
          averageTMax += ray.m_t_max;
        }
        else {
          ray = generateRay(currentVoxel, indexToWorld, tMax);
          currentTMax = tMax;
        }
        rt::SInteraction interaction;
        scene.intersect(ray, &interaction, rt::ESceneobjectMask::FILTER);
        if (interaction.hitInformation.hit) {
          //rayTs[hits] = ray.m_t_max;
          rayTs[hits] = currentTMax;
          ++hits;
          filteredS += rt::CSGGXMicroflakeDistribution::buildS(interaction.hitInformation.normal, interaction.material->specularRoughness());
          filteredDiffuseClr += interaction.material->diffuse(interaction.hitInformation.tc);
          filteredSpecularClr += interaction.material->glossy(interaction.hitInformation.tc);
          specularRoughness += interaction.material->specularRoughness();
          //rayDistance += ray.m_t_max;
          rayDistance += currentTMax;
          filteredNormal += interaction.hitInformation.normal;
          ior += interaction.hitInformation.ior;
        }
      }
      float averageDistance = rayDistance / (float)hits;
      SFilteredData filteredData;
      if (hits > 0) {
        filteredData.density = estimateDensity(averageDistance, hits, numSamples, tMax, rayTs);
        float invHits = 1.f / hits;
        const uint16_t MAX_U16 = -1;
        const float fMAX_U16 = MAX_U16;
        filteredData.diffuseColor = filteredDiffuseClr * invHits;
        filteredData.specularColor = filteredSpecularClr * invHits;
        filteredData.S = filteredS * invHits;
        filteredData.normal = glm::normalize(m_worldToModel * glm::vec4(filteredNormal * invHits, 0.f));
        filteredData.ior = ior * invHits;
      }
      else {
        filteredData.density = 0.f;
        filteredData.diffuseColor = glm::vec3(0.f);
        filteredData.specularColor = glm::vec3(0.f);
        filteredData.S = glm::mat3(0.f);
        filteredData.normal = glm::vec3(0.f);
        filteredData.ior = 1.f;
      }
      return filteredData;

    }

    D_CALLABLE void debug(rt::CDeviceScene& scene, uint32_t debugSamples) const {
      size_t hits = 0;
      size_t rays = debugSamples;

      float radius = glm::length(m_voxelSize) / 2.f;
      for (size_t i = 0; i < rays; ++i) {
        glm::vec3 originDir = m_sampler.uniformSampleSphere();
        glm::vec3 origin = m_voxelCenter + radius * originDir;
        rt::CCoordinateFrame frame = rt::CCoordinateFrame::fromNormal(-originDir);
        glm::vec dir = frame.tangentToWorld() * m_sampler.uniformSampleHemisphere();


        rt::SInteraction interaction;
        scene.intersect(rt::CRay(origin, glm::normalize(dir)), &interaction, rt::ESceneobjectMask::FILTER);
        if (interaction.hitInformation.hit && interaction.object->mesh()) {
          ++hits;
        }
        else if (interaction.hitInformation.hit && interaction.object->medium()) {
          rt::CRay mediumRay(interaction.hitInformation.pos, dir, rt::CRay::DEFAULT_TMAX);
          mediumRay.offsetRayOrigin(dir);
          //SInteraction siMediumEnd = m_scene->intersect(mediumRay);
          rt::SInteraction siMediumEnd;
          scene.intersect(mediumRay, &siMediumEnd, rt::ESceneobjectMask::FILTER);

          if (siMediumEnd.hitInformation.hit) {
            rt::SInteraction mediumInteraction;
            interaction.object->medium()->sample(mediumRay, m_sampler, &mediumInteraction);
            if (mediumInteraction.hitInformation.hit) {
              ++hits;
            }
          }

        }

      }

      printf("Probability of hit: %f\n", (float)hits / rays);
    }

  private:
    const glm::ivec3& m_currentVoxel;
    glm::vec3 m_voxelCenter;
    glm::vec3 m_voxelSize;

    const glm::mat4x3& m_indexToModel;
    const glm::mat4x3& m_modelToIndex;
    const glm::mat4x3& m_modelToWorld;
    const glm::mat4x3& m_worldToModel;
    const glm::ivec3& m_numVoxels;
    const rt::SAABB& m_worldBB;
    rt::CSampler& m_sampler;
    float m_sigma_t;
    uint32_t m_estimationIterations;
    float m_alpha;
    bool m_clipRays;
    float m_scaling;

    D_CALLABLE float estimateDensity(float averageDistance, uint32_t hits, uint32_t numSamples, float tMax, float* rayTs) const {
      //uint3 launchIdx = optixGetLaunchIndex();
      //uint3 launchDim = optixGetLaunchDimensions();

      float volumeCorrection = m_voxelSize.x; // Correction term (CurrentVolume/UnitVolume) because density is defined on unit cube
      const float P_hit_mesh = hits / (float)numSamples;
      //float density = (P_hit_mesh * volumeCorrection) / (tMax * m_sigma_t);
      float density = -glm::log(1.f - P_hit_mesh) / (averageDistance * m_sigma_t); // For clipping this is just initialisation
      float normalizedAlpha = m_alpha / m_sigma_t;
      //if (launchIdx.x == int(launchDim.x / 2) && launchIdx.y == int(launchDim.y / 2) && launchIdx.z == int(launchDim.z / 2)) {
      //  printf("P_hit_mesh: %f, initial density: %f\n", P_hit_mesh, density);
      //}

      //for (float d = -1.f; d < 1.f; d += 0.001f) {
      //  float pHit = estimatePhitVolume(d, tMax, 100);
      //  float loss = glm::abs(pHit - P_hit_mesh);
      //  printf("density: %f, P_hit_volume: %f, loss: %f\n", d, pHit, loss);
      //}
      //if (launchIdx.x == launchDim.x / 2 && launchIdx.y == launchDim.y / 2 && launchIdx.z == launchDim.z / 2) {
      //  printf("iteration, P_hit_volume_gt, density\n");
      //}
      float delta = 0.001f; // For loss derivative
      float P_hit_volume_gt;
      for (size_t iteration = 0; iteration < m_estimationIterations; ++iteration) {
        //float alpha = normalizedAlpha * glm::pow(0.5f, (float)iteration / 2.f);
        uint32_t volumeHits = 0;
        float P_hit_volume = estimatePhitVolume(density, tMax, numSamples);
        if (m_clipRays) {
          P_hit_volume_gt = estimatePhitVolumeGT2(density, rayTs, hits);
        }
        else {
          P_hit_volume_gt = estimatePhitVolumeGT(density, tMax);
        }
        //density += (P_hit_mesh - P_hit_volume) * alpha;
        float dLoss = estimateLoss(density, tMax, P_hit_mesh, numSamples);
        float dLossGT = estimateLossGT(density, tMax, P_hit_mesh);
        //density = density - alpha * dLossGT;
        //printf("iteration %i, P_hit_volume: %f, P_hit_volume_gt: %f, dLoss: %f, dLossGT: %f, density: %f, alpha: %f\n", (int)iteration, P_hit_volume, P_hit_volume_gt, dLoss, dLossGT, density, alpha);
        density = density - normalizedAlpha * (P_hit_volume_gt - P_hit_mesh);
        //if (launchIdx.x == launchDim.x / 2 && launchIdx.y == launchDim.y / 2 && launchIdx.z == launchDim.z / 2) {
        //  printf("%i, %f, %f\n", (int)iteration, P_hit_volume_gt, density);
        //}
        //density = density - normalizedAlpha * (P_hit_volume_gt - P_hit_mesh);
        //float invMaxDensity = 1.f / density;
        //for (size_t sample = 0; sample < numSamples; ++sample) {
        //  volumeHits += deltaTrack(invMaxDensity, tMax);
        //}
        //float P_hit_volume = (float)volumeHits / numSamples;
        //density += (P_hit_mesh - P_hit_volume) * delta;
        //printf("iteration %i, P_hit_volume: %f, density: %f\n", (int)iteration, P_hit_volume, density);
      }
      float diff = P_hit_volume_gt - P_hit_mesh;
      if (diff > 0.01f) {
        printf("[WARNING] Diff between P_hit_volume and P_hit_mesh: %f\n", diff);
      }
      return density;
    }

    D_CALLABLE float estimatePhitVolume(float density, float tMax, uint32_t numSamples) const {
      float invMaxDensity = 1.f / glm::max(density, FLT_EPSILON);
      uint32_t volumeHits = 0;
      for (size_t sample = 0; sample < numSamples; ++sample) {
        volumeHits += deltaTrack(invMaxDensity, tMax);
      }
      return (float)volumeHits / numSamples;

    }

    DH_CALLABLE float estimatePhitVolumeGT(float density, float tMax) const {
      return 1.f - glm::exp(-density * m_sigma_t * tMax);
    }

    DH_CALLABLE float estimatePhitVolumeGT2(float density, float* rayTs, uint32_t hits) const {
      float P = 0.f;
      for (uint32_t hit = 0; hit < hits; ++hit) {
        P += glm::exp(-density * m_sigma_t * rayTs[hit]);
      }
      return 1.f - P / (float)hits;
    }

    DH_CALLABLE float estimateLoss(float density, float tMax, float P_hit_mesh, uint32_t numSamples) const {
      float delta = 0.001f; // For loss derivative
      return (glm::abs(estimatePhitVolumeGT(density + delta, tMax) - P_hit_mesh) - glm::abs(estimatePhitVolumeGT(density - delta, tMax) - P_hit_mesh)) / (2.f * delta);
    }

    DH_CALLABLE float estimateLossGT(float density, float tMax, float P_hit_mesh) const {
      float zero = -glm::log(1.f - P_hit_mesh) / (m_sigma_t*tMax);
      if (density > zero) {
        float v = tMax * m_sigma_t;
        return v * glm::exp(-v * density);
      }
      else if (zero == 0.f) {
        return 0.f;
      }
      else {
        float v = tMax * m_sigma_t;
        return -v * glm::exp(-v * density);
      }
    }

    D_CALLABLE uint8_t deltaTrack(float invMaxDensity, float tMax) const {
      // Run single delta tracking step because voxel is homogeneous
      float t = 0.f;
      t -= glm::log(1.f - m_sampler.uniformSample01()) * invMaxDensity / m_sigma_t;
      if (t >= tMax) {
        return 0;
      }
      else {
        return 1;
      }
    }

    D_CALLABLE rt::CRay generateRay(const glm::vec3& currentVoxel, const glm::mat4x3& indexToWorld, float tMax) const {
      glm::vec3 originIndexSpace = currentVoxel + glm::vec3(m_sampler.uniformSample01(), m_sampler.uniformSample01(), m_sampler.uniformSample01());
      glm::vec3 originWorldSpace = indexToWorld * glm::vec4(originIndexSpace, 1.f);
      glm::vec3 directionWorldSpace = m_sampler.uniformSampleSphere();

      return rt::CRay(originWorldSpace, directionWorldSpace, tMax);
    }
  };
}

#endif