#ifndef LAMBERTIAN_BRDF_HPP
#define LAMBERTIAN_BRDF_HPP

#include <glm/glm.hpp>

#include "utility/qualifiers.hpp"
#define PI 3.14159265359f
namespace rt {

  struct SHitInformation;

  class CLambertianBRDF {
  public:
    DH_CALLABLE CLambertianBRDF();
    DH_CALLABLE CLambertianBRDF(const glm::vec3& albedo);
    D_CALLABLE glm::vec3 f(const SHitInformation& hitInformation, const glm::vec3& wo, const glm::vec3& wi) const;
  private:
    glm::vec3 m_albedo;
  };

  //inline glm::vec3 CLambertianBRDF::f(const SHitInformation& hitInformation, const glm::vec3& wo, const glm::vec3& wi) const {
  //  return m_albedo / PI;
  //}
}
#endif