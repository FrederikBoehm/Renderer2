#ifndef MICROFACET_BRDF_HPP
#define MICROFACET_BRDF_HPP

#include <glm/glm.hpp>

#include "utility/qualifiers.hpp"

namespace rt {
  struct SHitInformation;

  class CSpecularBRDF {
  public:
    DH_CALLABLE CSpecularBRDF();
    DH_CALLABLE CSpecularBRDF(const glm::vec3& specular, float shininess);
    D_CALLABLE glm::vec3 f(const SHitInformation& hitInformation, const glm::vec3& wo, const glm::vec3& wi) const;
  private:
    glm::vec3 m_specular;
    float m_shininess;
  };
}
#endif 
