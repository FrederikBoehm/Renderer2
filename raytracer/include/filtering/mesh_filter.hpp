#ifndef MESH_FILTER_HPP
#define MESH_FILTER_HPP
#include <glm/glm.hpp>
#include "intersect/aabb.hpp"
#include "utility/qualifiers.hpp"
#include "sampling/sampler.hpp"
#include "scene/device_scene_impl.hpp"
#include "intersect/ray.hpp"

namespace filter {
  struct SFilteredData {
    float density;
    glm::vec3 normal;
    glm::vec3 diffuseColor;
    glm::vec3 specularColor;
  };

  class CMeshFilter {
  public:
    DH_CALLABLE CMeshFilter(const glm::ivec3& currentVoxel, const glm::mat4x3& indexToModel, const glm::mat4x3& modelToIndex, const glm::mat4x3& modelToWorld, const glm::mat4x3& worldToModel, const glm::ivec3& numVoxels, const rt::SAABB& worldBB, rt::CSampler& sampler):
      m_currentVoxel(currentVoxel),
      m_indexToModel(indexToModel),
      m_modelToIndex(modelToIndex),
      m_modelToWorld(modelToWorld),
      m_worldToModel(worldToModel),
      m_numVoxels(numVoxels),
      m_worldBB(worldBB),
      m_sampler(sampler){
      m_voxelCenter = modelToWorld * glm::vec4(indexToModel * glm::vec4(glm::vec3(currentVoxel) + 0.5f, 1.f), 1.f); // +0.5 to get voxel center in world space
      m_voxelSize = (worldBB.m_max - worldBB.m_min) / glm::vec3(numVoxels);
    }

    D_CALLABLE float run(rt::CDeviceScene& scene, uint32_t numSamples) {
      glm::vec3 currentVoxel(m_currentVoxel);
      glm::mat4x3 indexToWorld = glm::mat4(m_modelToWorld) * glm::mat4(m_indexToModel);
      float tMax = glm::length(m_voxelSize); // voxelSize in Loubet, Neyret "Hybrid mesh-volume LoDs for all-scale pre-filtering of complex 3D assets"

      uint32_t hits = 0;
      for (uint32_t i = 0; i < numSamples; ++i) {
        glm::vec3 originIndexSpace = currentVoxel + glm::vec3(m_sampler.uniformSample01(), m_sampler.uniformSample01(), m_sampler.uniformSample01());
        glm::vec3 originWorldSpace = indexToWorld * glm::vec4(originIndexSpace, 1.f);
        glm::vec3 directionWorldSpace = m_sampler.uniformSampleSphere();

        rt::CRay ray(originWorldSpace, directionWorldSpace, tMax);
        rt::SInteraction interaction;
        scene.intersect(ray, &interaction, rt::ESceneobjectMask::FILTER);
        if (interaction.hitInformation.hit) {
          ++hits;
        }
      }
      float density = ((float)hits / numSamples); // Actually probability of intersection --> change to true density
      return density;
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
  };
}

#endif