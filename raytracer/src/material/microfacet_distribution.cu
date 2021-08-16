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
    // Beckmann
    //float vTan2Theta = tan2Theta(h);
    //if (isinf(vTan2Theta)) {
    //  return 0.0f;
    //}
    //const float vCos2Theta = cos2Theta(h);
    //const float vCos4Theta = vCos2Theta * vCos2Theta;
    //return glm::exp(-vTan2Theta * (cos2Phi(h) / (m_alphaX * m_alphaX) + sin2Phi(h) / (m_alphaY * m_alphaY))) / (M_PI * m_alphaX * m_alphaY * vCos4Theta);
    // GGX
    //const float vCos2Theta = cos2Theta(h);
    //const float vCos4Theta = vCos2Theta * vCos2Theta;
    //float squared = m_alphaX * m_alphaY * tan2Theta(h);
    //squared *= squared;
    //return m_alphaX * m_alphaY * glm::max(cosTheta(h), 0.0f) / (M_PI * vCos4Theta * squared);
  }

  float CMicrofacetDistribution::G(const glm::vec3& wo, const glm::vec3& wi) const {
    return  1.0f / (1.0f + lambda(wo) + lambda(wi));
    //float g1o = 1.0f / (1.f + lambda(wo));
    //float g1i = 1.0f / (1.f + lambda(wi));
    //return g1o * g1i;
    //const glm::vec3 h = glm::normalize(wo + wi);
    //return G1(wo, h) * G1(wi, h);
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
    //float vTan2Theta = tan2Theta(w);
    //if (isinf(vTan2Theta)) {
    //  return 0.0f;
    //}
    //float xTerm = cosPhi(w) * m_alphaX;
    //float yTerm = sinPhi(w) * m_alphaY;
    //float alpha2 = xTerm * xTerm + yTerm * yTerm;
    //return 0.5f * (glm::sqrt(1.0f + alpha2 * vTan2Theta) - 1);


    // Beckmann
    //float absTanTheta = glm::abs(tanTheta(w));
    //if (isinf(absTanTheta)) {
    //  return 0.0f;
    //}
    //float alpha = glm::sqrt(cos2Phi(w) * m_alphaX * m_alphaX + sin2Phi(w) * m_alphaY * m_alphaY);
    //float a = 1 / (alpha * absTanTheta);
    //if (a >= 1.6f) {
    //  return 0.0f;
    //}
    //return (1 - 1.259f * a + 0.396f * a * a) / (3.535f * a + 2.181f * a * a);

  }

  float CMicrofacetDistribution::G1(const glm::vec3& w, const glm::vec3& h) const {
    const float wDotH = glm::dot(w, h);
    const float wDotN = cos2Theta(w);

    return glm::max(wDotH / wDotN, 0.0f) * 2.0f / (1.0f + glm::sqrt(1.0f + m_alphaX * m_alphaY * tan2Theta(w)));
  }
}