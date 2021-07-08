#ifndef MICROFACET_BRDF_HPP
#define MICROFACET_BRDF_HPP

#include <glm/glm.hpp>

#include "utility/qualifiers.hpp"
#include "intersect/hit_information.hpp"

#define PI 3.14159265359f

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

  //inline glm::vec3 CSpecularBRDF::f(const SHitInformation& hitInformation, const glm::vec3& wo, const glm::vec3& wi) const {
  //  glm::vec3 h = normalize(wo + wi);
  //  return (m_shininess + 8.0f) / (8.0f*PI) * m_specular * glm::pow(glm::max(glm::dot(hitInformation.normal, h), 0.0f), m_shininess);
  //}
}
#endif 
