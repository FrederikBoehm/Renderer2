#ifndef MICROFACET_BRDF
#define MICROFACET_BRDF

#include <glm/glm.hpp>

#include "utility/qualifiers.hpp"
#include "intersect/hit_information.hpp"
#include "fresnel.hpp"
#include "microfacet_distribution.hpp"
#include "sampling/sampler.hpp"
#include "utility/functions.hpp"

namespace rt {
  class CSampler;
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
    D_CALLABLE glm::vec3 f(const glm::vec3& wo, const glm::vec3& wi) const;
    D_CALLABLE glm::vec3 sampleF(const glm::vec3& wo, glm::vec3* wi, CSampler& sampler, float* pdf) const;
    D_CALLABLE float pdf(const glm::vec3& wo, const glm::vec3& wi) const;

  private:
    glm::vec3 m_glossy;
    CMicrofacetDistribution m_distribution;
    CFresnel m_fresnel;
  };

  inline CMicrofacetBRDF::CMicrofacetBRDF() :
    m_glossy(1.0f),
    m_distribution(1.0f, 1.0f),
    m_fresnel(1.0f, 1.0f) {

  }

  inline CMicrofacetBRDF::CMicrofacetBRDF(const glm::vec3& glossy, float alphaX, float alphaY, float etaI, float etaT) :
    m_glossy(glossy),
    m_distribution(alphaX, alphaY),
    m_fresnel(etaI, etaT) {

  }

  DH_CALLABLE inline glm::vec3 faceforward(const glm::vec3& v, const glm::vec3& v2) {
    return (glm::dot(v, v2) < 0.f) ? -v : v;
  }

  inline glm::vec3 CMicrofacetBRDF::f(const glm::vec3& wo, const glm::vec3& wi) const {
    float cosThetaO = absCosTheta(wo);
    float cosThetaI = absCosTheta(wi);

    glm::vec3 h = wi + wo;

    if (cosThetaI == 0.0f || cosThetaO == 0) {
      return glm::vec3(0.0f);
    }

    if (h.x == 0.0f && h.y == 0.0f && h.z == 0.0f) {
      return glm::vec3(0.0f);
    }

    h = glm::normalize(h);
    glm::vec3 F = m_fresnel.evaluate(glm::dot(wi, faceforward(h, glm::vec3(0.f, 0.f, 1.f))));
    return m_glossy * m_distribution.D(h) * m_distribution.G(wo, wi) * F / (4.0f * cosThetaI * cosThetaO);
  }

  inline glm::vec3 CMicrofacetBRDF::sampleF(const glm::vec3& wo, glm::vec3* wi, CSampler& sampler, float* pdf) const {
    if (wo.z == 0) {
      return glm::vec3(0.f);
    }
    glm::vec3 h = m_distribution.sampleH(wo, glm::vec2(sampler.uniformSample01(), sampler.uniformSample01()));

    if (glm::dot(wo, h) < 0.f) {
      return glm::vec3(0.f);
    }

    *wi = glm::reflect(-wo, h);
    if (!sameHemisphere(wo, *wi)) {
      return glm::vec3(0.f);
    }
    *pdf = m_distribution.pdf(wo, h) / (4 * glm::dot(wo, h));
    return f(wo, *wi);
  }

  inline float CMicrofacetBRDF::pdf(const glm::vec3& wo, const glm::vec3& wi) const {
    if (!sameHemisphere(wo, wi)) {
      return 0.f;
    }
    glm::vec3 h = glm::normalize(wo + wi);
    return m_distribution.pdf(wo, h) / (4 * glm::dot(wo, h));
  }
}
#endif // !MICROFACET_BRDF
