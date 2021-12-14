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

    DH_CALLABLE glm::vec3 Le() const;

    D_CALLABLE glm::vec3 f(const glm::vec3& wo, const glm::vec3& wi) const;
    D_CALLABLE glm::vec3 sampleF(const glm::vec3& wo, glm::vec3* wi, CSampler& sampler, float* pdf) const;
    D_CALLABLE float pdf(const glm::vec3& wo, const glm::vec3& wi) const;

    DH_CALLABLE CMaterial& operator=(const CMaterial& material);

    D_CALLABLE glm::vec3 color();

  private:
    glm::vec3 m_Le; // Emissive light if light source
    COrenNayarBRDF m_orenNayarBRDF;
    CMicrofacetBRDF m_microfacetBRDF;
  };

  inline CMaterial::CMaterial() :
    m_Le(glm::vec3(0.0f)),
    m_orenNayarBRDF(),
    m_microfacetBRDF() {
  }

  inline CMaterial::CMaterial(const glm::vec3& le) :
    m_Le(le),
    m_orenNayarBRDF(),
    m_microfacetBRDF() {

  }

  inline CMaterial::CMaterial(const COrenNayarBRDF& diffuse, const CMicrofacetBRDF& glossy) :
    m_Le(glm::vec3(0.0f)),
    m_orenNayarBRDF(diffuse),
    m_microfacetBRDF(glossy) {

  }

  // Evaluates material at a hitPoint. Gives the color of that point
  inline glm::vec3 CMaterial::f(const glm::vec3& wo, const glm::vec3& wi) const {
    glm::vec3 diffuse = m_orenNayarBRDF.f(wo, wi);
    glm::vec3 microfacet = m_microfacetBRDF.f(wo, wi);
    return 0.5f * (diffuse + microfacet);
  }

  inline CMaterial& CMaterial::operator=(const CMaterial& material) {
    this->m_Le = material.m_Le;
    this->m_orenNayarBRDF = material.m_orenNayarBRDF;
    this->m_microfacetBRDF = material.m_microfacetBRDF;
    return *this;
  }

  inline glm::vec3 CMaterial::sampleF(const glm::vec3& wo, glm::vec3* wi, CSampler& sampler, float* pdf) const {
    if (sampler.uniformSample01() < 0.5f) {
      // Sample diffuse
      return m_orenNayarBRDF.sampleF(wo, wi, sampler, pdf);
    }
    else {
      // Sample specular
      return m_microfacetBRDF.sampleF(wo, wi, sampler, pdf);
    }
  }

  inline float CMaterial::pdf(const glm::vec3& wo, const glm::vec3& wi) const {
    return 0.5f * (m_orenNayarBRDF.pdf(wo, wi) + m_microfacetBRDF.pdf(wo, wi));
  }


  DH_CALLABLE inline glm::vec3 CMaterial::Le() const {
    return m_Le;
  }

  inline glm::vec3 CMaterial::color() {
    return m_orenNayarBRDF.m_albedo;
  }

}
#endif // !MATERIAL_HXX
