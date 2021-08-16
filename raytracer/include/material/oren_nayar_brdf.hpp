#ifndef OREN_NAYAR_BRDF_HPP
#define OREN_NAYAR_BRDF_HPP

#include <glm/glm.hpp>

#include "utility/qualifiers.hpp"
#include "intersect/hit_information.hpp"

namespace rt {
  // Better approximation of rough surfaces than lambertian reflection
  class COrenNayarBRDF {
  public:
    DH_CALLABLE COrenNayarBRDF();
    DH_CALLABLE COrenNayarBRDF(const glm::vec3& albedo, float roughness);

    // Expects incident and outgoing directions in tangent space
    DH_CALLABLE glm::vec3 f(const SHitInformation& hitInformation, const glm::vec3& wo, const glm::vec3& wi) const;

  private:
    glm::vec3 m_albedo;
    float m_A;
    float m_B;
  };
}
#endif

