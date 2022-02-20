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

// Includes for debug below
#include "scene/device_sceneobject.hpp"

namespace filter {

  class CMeshFilter {
  public:
    DH_CALLABLE CMeshFilter(const glm::ivec3& currentVoxel, const glm::mat4x3& indexToModel, const glm::mat4x3& modelToIndex, const glm::mat4x3& modelToWorld, const glm::mat4x3& worldToModel, const glm::ivec3& numVoxels, const rt::SAABB& worldBB, rt::CSampler& sampler, float sigma_t, uint32_t estimationIterations):
      m_currentVoxel(currentVoxel),
      m_indexToModel(indexToModel),
      m_modelToIndex(modelToIndex),
      m_modelToWorld(modelToWorld),
      m_worldToModel(worldToModel),
      m_numVoxels(numVoxels),
      m_worldBB(worldBB),
      m_sampler(sampler),
      m_sigma_t(sigma_t),
      m_estimationIterations(estimationIterations) {
      m_voxelCenter = modelToWorld * glm::vec4(indexToModel * glm::vec4(glm::vec3(currentVoxel) + 0.5f, 1.f), 1.f); // +0.5 to get voxel center in world space
      m_voxelSize = (worldBB.m_max - worldBB.m_min) / glm::vec3(numVoxels);
    }

    D_CALLABLE SFilteredData run(rt::CDeviceScene& scene, uint32_t numSamples) {
      glm::vec3 currentVoxel(m_currentVoxel);
      glm::mat4x3 indexToWorld = glm::mat4(m_modelToWorld) * glm::mat4(m_indexToModel);
      float tMax = glm::length(m_voxelSize); // voxelSize in Loubet, Neyret "Hybrid mesh-volume LoDs for all-scale pre-filtering of complex 3D assets"
      float volume = (m_worldBB.m_max.x - m_worldBB.m_min.x) * (m_worldBB.m_max.y - m_worldBB.m_min.y) * (m_worldBB.m_max.z - m_worldBB.m_min.z);

      uint32_t hits = 0;
      glm::vec3 filteredNormal(0.f);
      glm::vec3 filteredDiffuseClr(0.f);
      glm::vec3 filteredSpecularClr(0.f);
      float specularRoughness = 0.f;
      float rayDistance = 0.f;
      for (uint32_t i = 0; i < numSamples; ++i) {
        rt::CRay ray = generateRay(currentVoxel, indexToWorld, tMax);
        rt::SInteraction interaction;
        scene.intersect(ray, &interaction, rt::ESceneobjectMask::FILTER);
        if (interaction.hitInformation.hit) {
          ++hits;
          filteredNormal += interaction.hitInformation.normal;
          filteredDiffuseClr += interaction.material->diffuse(interaction.hitInformation.tc);
          filteredSpecularClr += interaction.material->glossy(interaction.hitInformation.tc);
          specularRoughness += interaction.material->specularRoughness();
          rayDistance += ray.m_t_max;
        }
      }
      float averageDistance = rayDistance / (float)hits;
      SFilteredData filteredData;
      if (hits > 0) {
        filteredData.density = estimateDensity(averageDistance, hits, numSamples, tMax);
        float invHits = 1.f / hits;
        filteredData.normal = glm::normalize((m_worldToModel * glm::vec4(filteredNormal, 0.f)) * invHits);
        const uint16_t MAX_U16 = -1;
        const float fMAX_U16 = MAX_U16;
        filteredData.diffuseColor = glm::clamp(fMAX_U16 * filteredDiffuseClr * invHits, 0.f, fMAX_U16);
        filteredData.specularColor = glm::clamp(fMAX_U16 * filteredSpecularClr * invHits, 0.f, fMAX_U16);
        filteredData.specularRoughness = specularRoughness * invHits;
      }
      else {
        filteredData.density = 0.f;
        filteredData.normal = glm::vec3(0.f);
        filteredData.diffuseColor = glm::u16vec3(0);
        filteredData.specularColor = glm::u16vec3(0);
        filteredData.specularRoughness = 0.f;
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
        scene.intersect(rt::CRay(origin, dir), &interaction, rt::ESceneobjectMask::FILTER);
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

    D_CALLABLE float estimateDensity(float averageDistance, uint32_t hits, uint32_t numSamples, float tMax) const {
      
      float volumeCorrection = m_voxelSize.x; // Correction term (CurrentVolume/UnitVolume) because density is defined on unit cube
      const float P_hit_mesh = hits / (float)numSamples;
      float density = (P_hit_mesh * volumeCorrection) / (averageDistance * m_sigma_t);
      float delta = 0.1f;

      for (size_t iteration = 0; iteration < m_estimationIterations; ++iteration) {
        uint32_t volumeHits = 0;
        float invMaxDensity = 1.f / density;
        for (size_t sample = 0; sample < numSamples; ++sample) {
          volumeHits += deltaTrack(invMaxDensity, tMax);
        }
        float P_hit_volume = (float)volumeHits / numSamples;
        density += (P_hit_mesh - P_hit_volume) * delta;
      }
      return density;
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