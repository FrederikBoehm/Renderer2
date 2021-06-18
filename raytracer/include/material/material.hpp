#ifndef MATERIAL_HXX
#define MATERIAL_HXX
#include <glm/glm.hpp>

#include "utility/qualifiers.hpp"
#include "lambertian_brdf.hpp"
#include "specular_brdf.hpp"
namespace rt {
  class SHitInformation;

  class CMaterial {
  public:
    DH_CALLABLE CMaterial();
    DH_CALLABLE CMaterial(const glm::vec3& le);
    DH_CALLABLE CMaterial(CLambertianBRDF lambertian, CSpecularBRDF specular);

    DH_CALLABLE const glm::vec3& Le() const;

    D_CALLABLE glm::vec3 f(const SHitInformation& hitInformation, const glm::vec3& wo, const glm::vec3& wi) const;

    DH_CALLABLE CMaterial& operator=(const CMaterial& material);

  private:
    glm::vec3 m_Le; // Emissive light if light source
    CLambertianBRDF m_lambertianBRDF;
    CSpecularBRDF m_specularBRDF;
  };


  inline const glm::vec3& CMaterial::Le() const {
    return m_Le;
  }

}
#endif // !MATERIAL_HXX
