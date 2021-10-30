#ifndef HOMOGENEOUS_MEDIUM_HPP
#define HOMOGENEOUS_MEDIUM_HPP

#include <glm/glm.hpp>
#include "henyey_greenstein_phase_function.hpp"
#include "utility/qualifiers.hpp"
#include "medium.hpp"
namespace rt {
  class CRay;
  class CSampler;
  class SInteraction;

  // Homogeneous medium
  class CHomogeneousMedium : public CMedium {
  public:
    DH_CALLABLE CHomogeneousMedium(const glm::vec3& sigma_a, const glm::vec3& sigma_s, float g);
    DH_CALLABLE glm::vec3 tr(const CRay& ray, const CSampler& sampler) const;
    DH_CALLABLE glm::vec3 sample(const CRay& ray, CSampler& sampler, SInteraction* mi) const;

    DH_CALLABLE const CHenyeyGreensteinPhaseFunction& phase() const;

  private:
    const glm::vec3 m_sigma_a;
    const glm::vec3 m_sigma_s;
    const glm::vec3 m_sigma_t;
    const CHenyeyGreensteinPhaseFunction m_phase;
  };

  DH_CALLABLE inline const CHenyeyGreensteinPhaseFunction& CHomogeneousMedium::phase() const {
    return m_phase;
  }
}

#endif