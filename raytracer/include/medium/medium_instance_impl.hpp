#ifndef MEDIUM_INSTANCE_IMPL_HPP
#define MEDIUM_INSTANCE_IMPL_HPP
#include "medium_instance.hpp"
#include "nvdb_medium_impl.hpp"

namespace rt {
  inline glm::vec3 CMediumInstance::sample(const CRay& rayWorld, CSampler& sampler, SInteraction* mi) const {
    const CRay rayMedium = rayWorld.transform(*m_worldToModel);
    glm::vec3 albedo = m_medium->sample(rayMedium, sampler, mi);
    const CRay rayWorldNew = rayMedium.transform(*m_modelToWorld);
    rayWorld.m_t_max = rayWorldNew.m_t_max;
    mi->hitInformation.t = rayWorldNew.m_t_max;
    mi->hitInformation.pos = *m_modelToWorld * glm::vec4(mi->hitInformation.pos, 1.f);
    mi->hitInformation.normal = *m_modelToWorld * glm::vec4(mi->hitInformation.normal, 0.f);
    mi->hitInformation.normalG = *m_modelToWorld * glm::vec4(mi->hitInformation.normalG, 0.f);
    mi->medium = mi->hitInformation.hit ? this : nullptr;
    return albedo;
  }

  inline glm::vec3 CMediumInstance::tr(const CRay& ray, CSampler& sampler) const {
    const CRay rayMedium = ray.transform(*m_worldToModel);
    return m_medium->tr(rayMedium, sampler);
  }

  inline glm::vec3 CMediumInstance::normal(const glm::vec3& p, CSampler& sampler) const {
    const glm::vec3 pMedium = *m_worldToModel * glm::vec4(p, 1.f);
    return *m_modelToWorld * glm::vec4(m_medium->normal(pMedium, sampler), 1.f);
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
}

#endif // !MEDIUM_INSTANCE_IMPL_HPP

