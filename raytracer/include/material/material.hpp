#ifndef MATERIAL_HXX
#define MATERIAL_HXX
#include <glm/glm.hpp>

#include "utility/qualifiers.hpp"
#include "oren_nayar_brdf.hpp"
#include "microfacet_brdf.hpp"
namespace rt {
  class SHitInformation;

  class CMaterial {
  public:
    DH_CALLABLE CMaterial();
    DH_CALLABLE CMaterial(const glm::vec3& le);
    DH_CALLABLE CMaterial(const COrenNayarBRDF& diffuse, const CMicrofacetBRDF& glossy);

    DH_CALLABLE const glm::vec3& Le() const;

    D_CALLABLE glm::vec3 f(const SHitInformation& hitInformation, const glm::vec3& wo, const glm::vec3& wi) const;

    DH_CALLABLE CMaterial& operator=(const CMaterial& material);

  private:
    glm::vec3 m_Le; // Emissive light if light source
    COrenNayarBRDF m_orenNayarBRDF;
    CMicrofacetBRDF m_microfacetBRDF;
  };


  inline const glm::vec3& CMaterial::Le() const {
    return m_Le;
  }

}
#endif // !MATERIAL_HXX
