#include "material/material.hpp"

#include "intersect/hit_information.hpp"

namespace rt {
  CMaterial::CMaterial():
    m_Le(glm::vec3(0.0f)),
    m_lambertianBRDF(glm::vec3(0.0f)),
    m_orenNayarBRDF(),
    m_specularBRDF() {
  }
  CMaterial::CMaterial(const glm::vec3& le) :
    m_Le(le),
    m_lambertianBRDF(),
    m_orenNayarBRDF(),
    m_specularBRDF() {

  }
  CMaterial::CMaterial(CLambertianBRDF lambertian, CBlinnPhongBRDF specular) :
    m_Le(glm::vec3(0.0f)),
    m_lambertianBRDF(lambertian),
    m_specularBRDF(specular) {

  }

  CMaterial::CMaterial(COrenNayarBRDF diffuse, CBlinnPhongBRDF specular) :
    m_Le(glm::vec3(0.0f)),
    m_orenNayarBRDF(diffuse),
    m_specularBRDF(specular) {

  }

  CMaterial::CMaterial(COrenNayarBRDF diffuse, CMicrofacetBRDF glossy) :
    m_Le(glm::vec3(0.0f)),
    m_orenNayarBRDF(diffuse),
    m_microfacetBRDF(glossy) {

  }

  // Evaluates material at a hitPoint. Gives the color of that point
  glm::vec3 CMaterial::f(const SHitInformation& hitInformation, const glm::vec3& wo, const glm::vec3& wi) const {
    //glm::vec3 lambertian = m_lambertianBRDF.f(hitInformation, wo, wi);
    glm::vec3 diffuse = m_orenNayarBRDF.f(hitInformation, wo, wi);
    //glm::vec3 diffuse = glm::vec3(0.0f);
    //glm::vec3 diffuse = glm::vec3(0.0f);
    glm::vec3 microfacet = m_microfacetBRDF.f(hitInformation, wo, wi);
    return diffuse + microfacet;
  }

  CMaterial& CMaterial::operator=(const CMaterial& material) {
    this->m_Le = material.m_Le;
    this->m_lambertianBRDF = material.m_lambertianBRDF;
    this->m_orenNayarBRDF = material.m_orenNayarBRDF;
    this->m_specularBRDF = material.m_specularBRDF;
    this->m_microfacetBRDF = material.m_microfacetBRDF;
    return *this;
  }
}