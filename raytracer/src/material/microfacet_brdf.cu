#include "material/microfacet_brdf.hpp"
#include "material/brdf_functions.hpp"

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

  glm::vec3 CMicrofacetBRDF::f(const SHitInformation& hitInformation, const glm::vec3& wo, const glm::vec3& wi) const {
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
    //return F * m_distribution.D(h) * m_distribution.G(wo, wi);
    //return m_glossy * m_distribution.D(h);
    //return m_glossy * m_distribution.D(h) * m_distribution.G(wo, wi) / (4.0f * cosThetaI * cosThetaO);
    //return m_glossy / (4.0f * cosThetaI * cosThetaO);
  }
}