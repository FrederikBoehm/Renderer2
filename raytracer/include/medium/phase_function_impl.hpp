#ifndef PHASE_FUNCTION_IMPL_HPP
#define PHASE_FUNCTION_IMPL_HPP
#include "phase_function.hpp"
#include "henyey_greenstein_phase_function.hpp"
#include "sggx_phase_function.hpp"
#include <stdio.h>

namespace rt {

  inline float CPhaseFunction::p(const glm::vec3& wo, const glm::vec3& wi, const glm::mat3& S, const glm::vec3& normal, float ior, CSampler& sampler) const {
    switch (m_type) {
    case EPhaseFunction::HENYEY_GREENSTEIN:
      return ((CHenyeyGreensteinPhaseFunction*)this)->p(wo, wi);
    case EPhaseFunction::SGGX: {
      CSGGXPhaseFunction phase(S, normal, ior);
      return phase.p(wo, wi, sampler);
    }
    default:
      fprintf(stderr, "CPhaseFunction::p: No valid phase function for type %i\n", m_type);
      return 0.f;
    }
    return 0.f;
  }

  inline float CPhaseFunction::sampleP(const glm::vec3& wo, glm::vec3* wi, const glm::mat3& S, const glm::vec3& normal, float ior, CSampler& sampler) const {
    switch (m_type) {
    case EPhaseFunction::HENYEY_GREENSTEIN:
      return ((CHenyeyGreensteinPhaseFunction*)this)->sampleP(wo, wi, glm::vec2(sampler.uniformSample01(), sampler.uniformSample01()));
    case EPhaseFunction::SGGX: {
      CSGGXPhaseFunction phase(S, normal, ior);
      return phase.sampleP(wo, wi, sampler);
    }
    default:
      fprintf(stderr, "CPhaseFunction::sampleP: No valid phase function for type %i\n", m_type);
      return 0.f;
    }
    return 0.f;
  }

  inline EPhaseFunction CPhaseFunction::type() const {
    return m_type;
  }
}

#endif