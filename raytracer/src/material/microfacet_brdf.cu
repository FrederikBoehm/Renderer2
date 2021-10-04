#include "material/microfacet_brdf.hpp"
#include "material/brdf_functions.hpp"
#include "sampling/sampler.hpp"

namespace rt {
  CMicrofacetBRDF::CMicrofacetBRDF():
    m_glossy(1.0f),
    m_distribution(1.0f, 1.0f),
    m_fresnel(1.0f, 1.0f) {

  }

  CMicrofacetBRDF::CMicrofacetBRDF(const glm::vec3& glossy, float alphaX, float alphaY, float etaI, float etaT):
    m_glossy(glossy),
    m_distribution(alphaX, alphaY),
    m_fresnel(etaI, etaT) {

  }

  glm::vec3 CMicrofacetBRDF::f(const glm::vec3& wo, const glm::vec3& wi) const {
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
    glm::vec3 F = m_fresnel.evaluate(glm::dot(wi, h));
    return m_glossy * m_distribution.D(h) * m_distribution.G(wo, wi) * F / (4.0f * cosThetaI * cosThetaO);
  }

  glm::vec3 CMicrofacetBRDF::sampleF(const glm::vec3& wo, glm::vec3* wi, CSampler& sampler, float* pdf) const {
    glm::vec3 h = m_distribution.sampleH(wo, glm::vec2(sampler.uniformSample01(), sampler.uniformSample01()));
    *wi = glm::reflect(-wo, h);
    if (!(wo.z * wi->z) > 0) {
      return glm::vec3(0.f);
    }
    *pdf = m_distribution.pdf(wo, h) / 4 * glm::dot(wo, h);
    return f(wo, *wi);
  }

  float CMicrofacetBRDF::pdf(const glm::vec3& wo, const glm::vec3& wi) const {
    glm::vec3 h = glm::normalize(wo + wi);
    return m_distribution.pdf(wo, h) / 4 * glm::dot(wo, h);
  }


}