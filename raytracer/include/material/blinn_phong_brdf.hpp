#ifndef BLINN_PHONG_BRDF_HPP
#define BLINN_PHONG_BRDF_HPP

#include <glm/glm.hpp>

#include "utility/qualifiers.hpp"
#include "intersect/hit_information.hpp"

namespace rt {
  struct SHitInformation;

  class CBlinnPhongBRDF {
  public:
    DH_CALLABLE CBlinnPhongBRDF();
    DH_CALLABLE CBlinnPhongBRDF(const glm::vec3& specular, float shininess);
    D_CALLABLE glm::vec3 f(const SHitInformation& hitInformation, const glm::vec3& wo, const glm::vec3& wi) const;
  private:
    glm::vec3 m_specular;
    float m_shininess;
  };

}
#endif 
