#include "material/material.hpp"

#include "intersect/hit_information.hpp"

namespace rt {
  CMaterial::CMaterial():
    m_Le(glm::vec3(0.0f)),
    m_orenNayarBRDF(),
    m_microfacetBRDF() {
  }
  CMaterial::CMaterial(const glm::vec3& le) :
    m_Le(le),
    m_orenNayarBRDF(),
    m_microfacetBRDF() {

  }

  CMaterial::CMaterial(const COrenNayarBRDF& diffuse, const CMicrofacetBRDF& glossy) :
    m_Le(glm::vec3(0.0f)),
    m_orenNayarBRDF(diffuse),
    m_microfacetBRDF(glossy) {

  }

  // Evaluates material at a hitPoint. Gives the color of that point
  glm::vec3 CMaterial::f(const glm::vec3& wo, const glm::vec3& wi) const {
    glm::vec3 diffuse = m_orenNayarBRDF.f(wo, wi);
    glm::vec3 microfacet = m_microfacetBRDF.f(wo, wi);
    return diffuse + microfacet;
  }

  CMaterial& CMaterial::operator=(const CMaterial& material) {
    this->m_Le = material.m_Le;
    this->m_orenNayarBRDF = material.m_orenNayarBRDF;
    this->m_microfacetBRDF = material.m_microfacetBRDF;
    return *this;
  }

  glm::vec3 CMaterial::sampleF(const glm::vec3& wo, glm::vec3* wi, CSampler& sampler, float* pdf) const {
    glm::vec3 microfacet = m_microfacetBRDF.sampleF(wo, wi, sampler, pdf);
    glm::vec3 diffuse = m_orenNayarBRDF.f(wo, *wi);
    return diffuse + microfacet;
  }

  float CMaterial::pdf(const glm::vec3& wo, const glm::vec3& wi) const {
    return m_microfacetBRDF.pdf(wo, wi);
  }
}