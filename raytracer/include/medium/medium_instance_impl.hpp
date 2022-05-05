#ifndef MEDIUM_INSTANCE_IMPL_HPP
#define MEDIUM_INSTANCE_IMPL_HPP
#include "medium_instance.hpp"
#include "nvdb_medium_impl.hpp"

namespace rt {
  inline glm::vec3 CMediumInstance::sample(const CRay& rayWorld, CSampler& sampler, SInteraction* mi, bool useBrickGrid, size_t* numLookups) const {
    const CRay rayMedium = rayWorld.transform2(*m_worldToModel);
    glm::vec3 albedo = m_medium->sample(rayMedium, sampler, m_filterRenderRatio, mi, useBrickGrid, numLookups);
    const CRay rayWorldNew = rayMedium.transform2(*m_modelToWorld);
    rayWorld.m_t_max = rayWorldNew.m_t_max;
    mi->hitInformation.t = rayWorldNew.m_t_max;
    mi->hitInformation.pos = *m_modelToWorld * glm::vec4(mi->hitInformation.pos, 1.f);
    mi->hitInformation.normal = glm::normalize(*m_modelToWorld * glm::vec4(mi->hitInformation.normal, 0.f));
    mi->hitInformation.normalG = glm::normalize(*m_modelToWorld * glm::vec4(mi->hitInformation.normalG, 0.f));
    mi->medium = mi->hitInformation.hit ? this : nullptr;
    return albedo;
  }

  inline glm::vec3 CMediumInstance::tr(const CRay& ray, CSampler& sampler, bool useBrickGrid, size_t* numLookups) const {
    const CRay rayMedium = ray.transform2(*m_worldToModel);
    return m_medium->tr(rayMedium, sampler, m_filterRenderRatio, useBrickGrid, numLookups);
  }

  inline const CPhaseFunction& CMediumInstance::phase() const {
    return m_medium->phase();
  }

  inline SAABB CMediumInstance::worldBB() const {
    const SAABB& aabb = m_medium->worldBB();
    return aabb.transform(*m_modelToWorld);
  }

  inline SAABB CMediumInstance::modelBB() const {
    return m_medium->worldBB();
  }

  inline CRay CMediumInstance::moveToVoxelBorder(const CRay& ray) const {
    CRay modelSpaceRay = ray.transform(*m_worldToModel);
    return m_medium->moveToVoxelBorder(modelSpaceRay).robustTransform(*m_modelToWorld, -ray.m_direction);
  }

  inline float CMediumInstance::voxelSizeFiltering() const {
    return m_medium->voxelSizeFiltering();
  }
}

#endif // !MEDIUM_INSTANCE_IMPL_HPP

