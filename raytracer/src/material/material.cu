#include "material/material.hpp"

#include "intersect/hit_information.hpp"

namespace rt {
  CMaterial::CMaterial():
    m_Le(glm::vec3(0.0f)),
    m_lambertianBRDF(glm::vec3(0.0f)),
    m_specularBRDF() {
  }
  CMaterial::CMaterial(const glm::vec3& le) :
    m_Le(le),
    m_lambertianBRDF(),
    m_specularBRDF() {

  }
  CMaterial::CMaterial(CLambertianBRDF lambertian, CSpecularBRDF specular) :
    m_Le(glm::vec3(0.0f)),
    m_lambertianBRDF(lambertian),
    m_specularBRDF(specular) {

  }

  glm::vec3 CMaterial::f(const SHitInformation& hitInformation, const glm::vec3& wo, const glm::vec3& wi) const {
    glm::vec3 lambertian = m_lambertianBRDF.f(hitInformation, wo, wi);
    glm::vec3 microfacet = m_specularBRDF.f(hitInformation, wo, wi);
    return lambertian + microfacet;
  }

  CMaterial& CMaterial::operator=(const CMaterial& material) {
    this->m_Le = material.m_Le;
    this->m_lambertianBRDF = material.m_lambertianBRDF;
    this->m_specularBRDF = material.m_specularBRDF;
    return *this;
  }
}