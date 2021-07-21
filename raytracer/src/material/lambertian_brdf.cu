#include "material/lambertian_brdf.hpp"

#include "intersect/hit_information.hpp"

#define PI 3.14159265359f
namespace rt {
  CLambertianBRDF::CLambertianBRDF():
  m_albedo(glm::vec3(0.0f)){

  }

  CLambertianBRDF::CLambertianBRDF(const glm::vec3& albedo):
    m_albedo(albedo) {

  }

  // BRDF for diffuse lighting
  glm::vec3 CLambertianBRDF::f(const SHitInformation& hitInformation, const glm::vec3& wo, const glm::vec3& wi) const {
    return m_albedo / PI;
  }
}