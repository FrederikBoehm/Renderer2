#ifndef FRESNEL_HPP
#define FRESNEL_HPP

#include <glm/glm.hpp>

#include "utility/qualifiers.hpp"

namespace rt {
  // For now only support for dielectrics
  class CFresnel {
  public:
    DH_CALLABLE CFresnel(float etaI, float etaT);

    DH_CALLABLE glm::vec3 evaluate(float cosThetaI) const;

  private:
    float m_etaI;
    float m_etaT;
  };
}

#endif