#define _USE_MATH_DEFINES
#include <math.h>

#include "material/microfacet_distribution.hpp"

#include "material/brdf_functions.hpp"

namespace rt {
  CMicrofacetDistribution::CMicrofacetDistribution(float alphaX, float alphaY) :
    m_alphaX(alphaX),
    m_alphaY(alphaY) {

  }

  float CMicrofacetDistribution::D(const glm::vec3& h) const {
    // Trowbridge-Reitz
    float vTan2Theta = tan2Theta(h);
    if (isinf(vTan2Theta)) {
      return 0.0f;
    }
    const float vCos2Theta = cos2Theta(h);
    const float vCos4Theta = vCos2Theta * vCos2Theta;
    float e = (cos2Phi(h) / (m_alphaX * m_alphaX) +
      sin2Phi(h) / (m_alphaY * m_alphaY)) * vTan2Theta;
    return 1.0f / (M_PI * m_alphaX * m_alphaY * vCos4Theta * (1 + e) * (1 + e));
  }

  float CMicrofacetDistribution::G(const glm::vec3& wo, const glm::vec3& wi) const {
    return  1.0f / (1.0f + lambda(wo) + lambda(wi));
  }

  float CMicrofacetDistribution::lambda(const glm::vec3& w) const {
    // Trowbridge-Reitz
    float absTanTheta = glm::abs(tanTheta(w));
    if (isinf(absTanTheta)) {
      return 0.0f;
    }
    float alpha = glm::sqrt(cos2Phi(w) * m_alphaX * m_alphaX + sin2Phi(w) * m_alphaY * m_alphaY);
    float alpha2Tan2Theta = (alpha * absTanTheta) * (alpha * absTanTheta);
    return (-1.0f + glm::sqrt(1.0f + alpha2Tan2Theta)) / 2.0f;
  }

}