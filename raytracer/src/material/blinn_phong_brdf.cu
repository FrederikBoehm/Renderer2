#define _USE_MATH_DEFINES
#include <math.h>
#include "material/blinn_phong_brdf.hpp"

#include "intersect/hit_information.hpp"

namespace rt {
  CBlinnPhongBRDF::CBlinnPhongBRDF() :
    m_specular(glm::vec3(0.0f)),
    m_shininess(0.0f) {

  }

  CBlinnPhongBRDF::CBlinnPhongBRDF(const glm::vec3& specular, float shininess):
    m_specular(specular),
    m_shininess(shininess) {

  }

  // BRDF for specular reflections
  glm::vec3 CBlinnPhongBRDF::f(const SHitInformation& hitInformation, const glm::vec3& wo, const glm::vec3& wi) const {
    glm::vec3 h = normalize(wo + wi);
    return (m_shininess + 8.0f) / (float)(8.0f*M_PI) * m_specular * glm::pow(glm::max(glm::dot(glm::vec3(0.0f, 0.0f, 1.0f), h), 0.0f), m_shininess);
  }


}