#include "material/specular_brdf.hpp"

#include "intersect/hit_information.hpp"

#define PI 3.14159265359f

namespace rt {
  CSpecularBRDF::CSpecularBRDF() :
    m_specular(glm::vec3(0.0f)),
    m_shininess(0.0f) {

  }

  CSpecularBRDF::CSpecularBRDF(const glm::vec3& specular, float shininess):
    m_specular(specular),
    m_shininess(shininess) {

  }

  glm::vec3 CSpecularBRDF::f(const SHitInformation& hitInformation, const glm::vec3& wo, const glm::vec3& wi) const {
    glm::vec3 h = normalize(wo + wi);
    return (m_shininess + 8.0f) / (8.0f*PI) * m_specular * glm::pow(glm::max(glm::dot(hitInformation.normal, h), 0.0f), m_shininess);
  }


}