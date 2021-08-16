#include "material/fresnel.hpp"

namespace rt {
  CFresnel::CFresnel(float etaI, float etaT) :
    m_etaI(etaI),
    m_etaT(etaT) {

  }

  glm::vec3 CFresnel::evaluate(float cosThetaI) const {
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
      return glm::vec3(1.0f);
    }

    float cosThetaT = std::sqrt(glm::max(0.0f, 1.0f - sinThetaT * sinThetaT));

    float rParallel = ((etaT * cosThetaI) - (etaI * cosThetaT)) / ((etaT * cosThetaI) + (etaI * cosThetaT));
    float rPerpendicular = ((etaI * cosThetaI) - (etaT * cosThetaT)) / ((etaI * cosThetaI) + (etaT * cosThetaT));

    return glm::vec3((rParallel * rParallel + rPerpendicular * rPerpendicular) / 2.0f);

  }
}