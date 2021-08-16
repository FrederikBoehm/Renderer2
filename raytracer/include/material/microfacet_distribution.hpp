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
  private:
    float m_alphaX;
    float m_alphaY;

    DH_CALLABLE float lambda(const glm::vec3& w) const;
  };
}
#endif // !MICROFACET_DISTRIBUTION
