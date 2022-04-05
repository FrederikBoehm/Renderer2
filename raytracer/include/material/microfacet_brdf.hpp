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
    DH_CALLABLE CMicrofacetBRDF(float alphaX, float alphaY, float etaI, float etaT);
    D_CALLABLE float f(const glm::vec3& wo, const glm::vec3& wi) const;
    D_CALLABLE float sampleF(const glm::vec3& wo, glm::vec3* wi, CSampler& sampler, float* pdf) const;
    D_CALLABLE float pdf(const glm::vec3& wo, const glm::vec3& wi) const;
    DH_CALLABLE float roughness() const;
    DH_CALLABLE const CFresnel& fresnel() const;

  private:
    CMicrofacetDistribution m_distribution;
    CFresnel m_fresnel;
  };

  inline CMicrofacetBRDF::CMicrofacetBRDF() :
    m_distribution(1.0f, 1.0f),
    m_fresnel(1.0f, 1.0f) {

  }

  inline CMicrofacetBRDF::CMicrofacetBRDF(float alphaX, float alphaY, float etaI, float etaT) :
    m_distribution(alphaX, alphaY),
    m_fresnel(etaI, etaT) {

  }

  inline float CMicrofacetBRDF::roughness() const {
    return m_distribution.alpha();
  }

  DH_CALLABLE inline glm::vec3 faceforward(const glm::vec3& v, const glm::vec3& v2) {
    return (glm::dot(v, v2) < 0.f) ? -v : v;
  }

  inline float CMicrofacetBRDF::f(const glm::vec3& wo, const glm::vec3& wi) const {
    float cosThetaO = absCosTheta(wo);
    float cosThetaI = absCosTheta(wi);

    glm::vec3 h = wi + wo;

    if (cosThetaI == 0.0f || cosThetaO == 0) {
      return 0.f;
    }

    if (h.x == 0.0f && h.y == 0.0f && h.z == 0.0f) {
      return 0.f;
    }

    h = glm::normalize(h);
    float F = m_fresnel.evaluate(glm::dot(wi, faceforward(h, glm::vec3(0.f, 0.f, 1.f))));
    return m_distribution.D(h) * m_distribution.G(wo, wi) * F / (4.0f * cosThetaI * cosThetaO);
  }

  inline float CMicrofacetBRDF::sampleF(const glm::vec3& wo, glm::vec3* wi, CSampler& sampler, float* pdf) const {
    //if (wo.z == 0) {
    //  printf("CMicrofacetBRDF::sampleF wo.z 0\n");
    //  return 0.f;
    //}
    //glm::vec3 h = m_distribution.sampleH(wo, glm::vec2(sampler.uniformSample01(), sampler.uniformSample01()));

    ////if (glm::dot(wo, h) < 0.f) {
    ////  printf("CMicrofacetBRDF::sampleF wo dot h < 0\n");
    ////  return 0.f;
    ////}

    //*wi = glm::reflect(-wo, h);
    //if (!sameHemisphere(wo, *wi)) {
    //  printf("CMicrofacetBRDF::sampleF wo, wi not on same hemisphere\n");
    //  return 0.f;
    //}
    //*pdf = m_distribution.pdf(wo, h) / (4 * glm::dot(wo, h));
    //float result = f(wo, *wi);
    //if (result <= 0.f) {
    //  printf("CMicrofacetBRDF::sampleF result %f\n", result);
    //}
    //return result;
    if (wo.z == 0) {
      return 0.f;
    }
    glm::vec3 h = m_distribution.sampleH(wo, glm::vec2(sampler.uniformSample01(), sampler.uniformSample01()));

    if (glm::dot(wo, h) < 0.f) {
      return 0.f;
    }

    *wi = glm::reflect(-wo, h);
    if (!sameHemisphere(wo, *wi)) {
      return 0.f;
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

  inline const CFresnel& CMicrofacetBRDF::fresnel() const {
    return m_fresnel;
  }
}
#endif // !MICROFACET_BRDF
