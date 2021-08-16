#ifndef MICROFACET_BRDF
#define MICROFACET_BRDF

#include <glm/glm.hpp>

#include "utility/qualifiers.hpp"
#include "intersect/hit_information.hpp"
#include "fresnel.hpp"
#include "microfacet_distribution.hpp"

namespace rt {
  class CMicrofacetBRDF {
  public:
    DH_CALLABLE CMicrofacetBRDF();
    /*
      glossy: Color of glossy highlight
      alphaX: Roughness in x direction
      alphaY: Roughness in y direction
      etaI: Index of refraction for incident medium
      etaT: Index of refraction for transmission medium
    */
    DH_CALLABLE CMicrofacetBRDF(const glm::vec3& glossy, float alphaX, float alphaY, float etaI, float etaT);
    D_CALLABLE glm::vec3 f(const SHitInformation& hitInformation, const glm::vec3& wo, const glm::vec3& wi) const;

  private:
    glm::vec3 m_glossy;
    CMicrofacetDistribution m_distribution;
    CFresnel m_fresnel;
  };
}
#endif // !MICROFACET_BRDF
