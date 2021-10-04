#ifndef MICROFACET_DISTRIBUTION_HPP
#define MICROFACET_DISTRIBUTION_HPP

#include <glm/glm.hpp>

#include "utility/qualifiers.hpp"

namespace rt {
  // Defines the distribution and shadowing functions for the Trowbridge-Reitz/GGX model
  class CMicrofacetDistribution {
  public:
    DH_CALLABLE CMicrofacetDistribution(float alphaX, float alphaY);
    DH_CALLABLE float D(const glm::vec3& h) const;
    DH_CALLABLE float G(const glm::vec3& wo, const glm::vec3& wi) const;
    DH_CALLABLE glm::vec3 sampleH(const glm::vec3& wo, const glm::vec2& u) const;
    DH_CALLABLE float pdf(const glm::vec3& wo, const glm::vec3 wh) const;
  private:
    float m_alphaX;
    float m_alphaY;

    DH_CALLABLE float lambda(const glm::vec3& w) const;
    DH_CALLABLE float G1(const glm::vec3& w) const;
    DH_CALLABLE void sample11(float cosTheta, float u1, float u2, float* slopeX, float* slopeY) const;
    DH_CALLABLE glm::vec3 sample(const glm::vec3& wi, float alphaX, float alphaY, float u1, float u2) const;
  };
}
#endif // !MICROFACET_DISTRIBUTION
