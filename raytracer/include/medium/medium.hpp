#ifndef MEDIUM_HPP
#define MEDIUM_HPP

#include <glm/glm.hpp>
#include "henyey_greenstein_phase_function.hpp"
#include "utility/qualifiers.hpp"
namespace rt {
  class CRay;
  class CSampler;
  class SInteraction;

  // Homogeneous medium
  class CMedium {
  public:
    DH_CALLABLE CMedium(const glm::vec3& sigma_a, const glm::vec3& sigma_s, float g);
    DH_CALLABLE glm::vec3 tr(const CRay& ray, const CSampler& sampler) const;
    DH_CALLABLE glm::vec3 sample(const CRay& ray, CSampler& sampler, SInteraction* mi) const;

    DH_CALLABLE const CHenyeyGreensteinPhaseFunction& phase() const;

  private:
    const glm::vec3 m_sigma_a;
    const glm::vec3 m_sigma_s;
    const glm::vec3 m_sigma_t;
    const CHenyeyGreensteinPhaseFunction m_phase;
  };

  DH_CALLABLE inline const CHenyeyGreensteinPhaseFunction& CMedium::phase() const {
    return m_phase;
  }
}

#endif