#ifndef MEDIUM_HPP
#define MEDIUM_HPP

#include "utility/qualifiers.hpp"
#include "henyey_greenstein_phase_function.hpp"
namespace rt {
  class CRay;
  class CSampler;
  class SInteraction;
  class CPhaseFunction;

  enum EMediumType {
    HOMOGENEOUS_MEDIUM,
    HETEROGENOUS_MEDIUM,
    NVDB_MEDIUM
  };

  // Medium base class
  class CMedium {
  public:
    DH_CALLABLE virtual ~CMedium();

    DH_CALLABLE EMediumType type() const;

    D_CALLABLE glm::vec3 tr(const CRay& ray, CSampler& sampler) const;
    D_CALLABLE glm::vec3 sample(const CRay& ray, CSampler& sampler, SInteraction* mi) const;
    DH_CALLABLE const CPhaseFunction& phase() const;
    D_CALLABLE glm::vec3 normal(const glm::vec3& p, CSampler& sampler) const;

  protected:
    DH_CALLABLE CMedium(const EMediumType mediumType);


  private:
    const EMediumType m_type;
  };

  DH_CALLABLE inline EMediumType CMedium::type() const {
    return m_type;
  }

  

}

#endif