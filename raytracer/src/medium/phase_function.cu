#include "medium/phase_function.hpp"
#include "medium/henyey_greenstein_phase_function.hpp"
#include "medium/sggx_phase_function.hpp"
#include <stdio.h>
#include "sampling/sampler.hpp"

namespace rt {
  CPhaseFunction::CPhaseFunction(const EPhaseFunction type) :
    m_type(type) {

  }

  float CPhaseFunction::p(const glm::vec3& wo, const glm::vec3& wi, const glm::vec3& n, CSampler& sampler) const {
    switch (m_type) {
    case EPhaseFunction::HENYEY_GREENSTEIN:
      return ((CHenyeyGreensteinPhaseFunction*)this)->p(wo, wi);
    case EPhaseFunction::SGGX:
      return ((CSGGXPhaseFunction*)this)->p(wo, wi, n, sampler);
    default:
      fprintf(stderr, "CPhaseFunction::p: No valid phase function for type %i\n", m_type);
      return 0.f;
    }
    return 0.f;
  }

  float CPhaseFunction::sampleP(const glm::vec3& wo, glm::vec3* wi, const glm::vec3& n, CSampler& sampler) const {
    switch (m_type) {
    case EPhaseFunction::HENYEY_GREENSTEIN:
      return ((CHenyeyGreensteinPhaseFunction*)this)->sampleP(wo, wi, glm::vec2(sampler.uniformSample01(), sampler.uniformSample01()));
    case EPhaseFunction::SGGX:
      return ((CSGGXPhaseFunction*)this)->sampleP(wo, wi, n, sampler);
    default:
      fprintf(stderr, "CPhaseFunction::sampleP: No valid phase function for type %i\n", m_type);
      return 0.f;
    }
    return 0.f;
  }
}