#define _USE_MATH_DEFINES
#include <math.h>
#include <algorithm>

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

  float CMicrofacetDistribution::G1(const glm::vec3& w) const {
    return 1 / (1 + lambda(w));
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

  void CMicrofacetDistribution::sample11(float cosTheta, float u1, float u2, float* slopeX, float* slopeY) const {
    if (cosTheta > .9999) {
      float r = std::sqrt(u1 / (1 - u1));
      float phi = 6.28318530718 * u2;
      *slopeX = r * std::cos(phi);
      *slopeY = r * std::sin(phi);
      return;
    }

    float sinTheta = std::sqrt(std::max(0.f, 1.f - cosTheta * cosTheta));
    float tanTheta = sinTheta / cosTheta;
    float a = 1 / tanTheta;
    float G1 = 2 / (1 + std::sqrt(1.f + 1.f / (a * a)));

    float A = 2 * u1 / G1 - 1;
    float tmp = 1.f / (A * A - 1.f);
    if (tmp > 1e10) {
      tmp = 1e10;
    }

    float B = tanTheta;
    float D = std::sqrt(std::max(B * B * tmp * tmp - (A * A - B * B) * tmp, 0.f));
    float slope_x_1 = B * tmp - D;
    float slope_x_2 = B * tmp + D;
    *slopeX = (A < 0 || slope_x_2 > 1.f / tanTheta) ? slope_x_1 : slope_x_2;

    float S;
    if (u2 > 0.5f) {
      S = 1.f;
      u2 = 2.f * (u2 - .5f);
    }
    else {
      S = -1.f;
      u2 = 2.f * (.5f - u2);
    }

    float z =
      (u2 * (u2 * (u2 * 0.27385f - 0.73369f) + 0.46341f)) /
      (u2 * (u2 * (u2 * 0.093073f + 0.309420f) - 1.000000f) + 0.597999f);
    *slopeY = S * z * std::sqrt(1.f + *slopeX * *slopeX);
  }

  glm::vec3 CMicrofacetDistribution::sample(const glm::vec3& wi, float alphaX, float alphaY, float u1, float u2) const {
    glm::vec3 wiStretched = glm::normalize(glm::vec3(alphaX * wi.x, alphaY * wi.y, wi.z));

    float slopeX;
    float slopeY;
    sample11(cosTheta(wiStretched), u1, u2, &slopeX, &slopeY);

    float tmp = cosPhi(wiStretched) * slopeX - sinPhi(wiStretched) * slopeY;
    slopeY = sinPhi(wiStretched) * slopeX + cosPhi(wiStretched) * slopeY;
    slopeX = tmp;

    slopeX = alphaX * slopeX;
    slopeY = alphaY * slopeY;

    return glm::normalize(glm::vec3(-slopeX, -slopeY, 1.f));
  }

  glm::vec3 CMicrofacetDistribution::sampleH(const glm::vec3& wo, const glm::vec2& u) const {
    bool flip = wo.z < 0;
    glm::vec3 h = sample(flip ? -wo : wo, m_alphaX, m_alphaY, u.x, u.y);
    if (flip)
      h = -h;
    return h;
  }

  float CMicrofacetDistribution::pdf(const glm::vec3& wo, const glm::vec3 wh) const {
    return D(wh) * G1(wo) * std::abs(glm::dot(wo, wh)) / absCosTheta(wo);
  }

}