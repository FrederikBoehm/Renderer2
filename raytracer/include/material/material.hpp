#ifndef MATERIAL_HXX
#define MATERIAL_HXX
#include <glm/glm.hpp>

#include "utility/qualifiers.hpp"
#include "lambertian_brdf.hpp"
#include "oren_nayar_brdf.hpp"
#include "blinn_phong_brdf.hpp"
#include "microfacet_brdf.hpp"
namespace rt {
  class SHitInformation;

  class CMaterial {
  public:
    DH_CALLABLE CMaterial();
    DH_CALLABLE CMaterial(const glm::vec3& le);
    DH_CALLABLE CMaterial(CLambertianBRDF lambertian, CBlinnPhongBRDF specular);
    DH_CALLABLE CMaterial(COrenNayarBRDF diffuse, CBlinnPhongBRDF specular);
    DH_CALLABLE CMaterial(COrenNayarBRDF diffuse, CMicrofacetBRDF glossy);

    DH_CALLABLE const glm::vec3& Le() const;

    D_CALLABLE glm::vec3 f(const SHitInformation& hitInformation, const glm::vec3& wo, const glm::vec3& wi) const;

    DH_CALLABLE CMaterial& operator=(const CMaterial& material);

  private:
    glm::vec3 m_Le; // Emissive light if light source
    CLambertianBRDF m_lambertianBRDF;
    COrenNayarBRDF m_orenNayarBRDF;
    CBlinnPhongBRDF m_specularBRDF;
    CMicrofacetBRDF m_microfacetBRDF;
  };


  inline const glm::vec3& CMaterial::Le() const {
    return m_Le;
  }

}
#endif // !MATERIAL_HXX
