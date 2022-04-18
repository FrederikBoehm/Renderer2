#ifndef FRESNEL_HPP
#define FRESNEL_HPP

#include <glm/glm.hpp>

#include "utility/qualifiers.hpp"

namespace rt {
  // For now only support for dielectrics
  class CFresnel {
    friend class CMaterial;
  public:
    DH_CALLABLE CFresnel(float etaI, float etaT);

    DH_CALLABLE float evaluate(float cosThetaI) const;

  private:
    float m_etaI;
    float m_etaT;
  };

  inline CFresnel::CFresnel(float etaI, float etaT) :
    m_etaI(etaI),
    m_etaT(etaT) {

  }

  inline float CFresnel::evaluate(float cosThetaI) const {
    cosThetaI = glm::clamp(cosThetaI, -1.0f, 1.0f);

    bool entering = cosThetaI > 0.f;
    float etaI = m_etaI;
    float etaT = m_etaT;
    if (!entering) {
      etaI = m_etaT;
      etaT = m_etaI;
      cosThetaI = glm::abs(cosThetaI);
    }

    float sinThetaI = glm::sqrt(glm::max(0.0f, 1.0f - cosThetaI * cosThetaI));
    float sinThetaT = etaI / etaT * sinThetaI;

    if (sinThetaT >= 1.0f) {
      return 1.0f;
    }

    float cosThetaT = glm::sqrt(glm::max(0.0f, 1.0f - sinThetaT * sinThetaT));

    float rParallel = ((etaT * cosThetaI) - (etaI * cosThetaT)) / ((etaT * cosThetaI) + (etaI * cosThetaT));
    float rPerpendicular = ((etaI * cosThetaI) - (etaT * cosThetaT)) / ((etaI * cosThetaI) + (etaT * cosThetaT));

    return (rParallel * rParallel + rPerpendicular * rPerpendicular) / 2.0f;

  }
}

#endif