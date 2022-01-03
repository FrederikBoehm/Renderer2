#ifndef MATERIAL_HXX
#define MATERIAL_HXX
#include <glm/glm.hpp>

#include "utility/qualifiers.hpp"
#include "oren_nayar_brdf.hpp"
#include "microfacet_brdf.hpp"
#include <assimp/material.h>
#include "texture/texture.hpp"
#include "integrators/objects.hpp"
namespace rt {
  class SHitInformation;
  class CTexture;

  struct SMaterialDeviceResource {
    CTexture* d_albedoTexture = nullptr;
    CTexture* d_normalTexture = nullptr;
    CTexture* d_alphaTexture = nullptr;
  };

  class CMaterial {
  public:
    H_CALLABLE CMaterial();
    H_CALLABLE CMaterial(const glm::vec3& le);
    H_CALLABLE CMaterial(const glm::vec3& diffuseColor, const glm::vec3& glossyColor, const COrenNayarBRDF& diffuseBRDF, const CMicrofacetBRDF& glossy);
    H_CALLABLE CMaterial(const aiMaterial* material, const std::string& assetsBasepath);
    H_CALLABLE CMaterial(CMaterial&& material);
    DH_CALLABLE ~CMaterial();

    DH_CALLABLE glm::vec3 Le() const;

    D_CALLABLE glm::vec3 f(const glm::vec2& tc, const glm::vec3& wo, const glm::vec3& wi) const;
    D_CALLABLE glm::vec3 sampleF(const glm::vec2& tc, const glm::vec3& wo, glm::vec3* wi, CSampler& sampler, float* pdf) const;
    D_CALLABLE float pdf(const glm::vec3& wo, const glm::vec3& wi) const;
    D_CALLABLE glm::vec3 normalmap(const glm::vec3& n, const glm::vec2& tc) const;
    D_CALLABLE glm::vec3 normalmap(const CCoordinateFrame& frame, const glm::vec2& tc) const;

    DH_CALLABLE CMaterial& operator=(const CMaterial& material);

    D_CALLABLE glm::vec3 color();
    D_CALLABLE bool opaque(const glm::vec2& tc) const;

    H_CALLABLE void allocateDeviceMemory();
    H_CALLABLE CMaterial copyToDevice();
    H_CALLABLE void freeDeviceMemory();

  private:
    glm::vec3 m_Le; // Emissive light if light source
    glm::vec3 m_diffuseColor;
    glm::vec3 m_glossyColor;
    COrenNayarBRDF m_orenNayarBRDF;
    CMicrofacetBRDF m_microfacetBRDF;
    CTexture* m_albedoTexture;
    CTexture* m_normalTexture;
    CTexture* m_alphaTexture;

    SMaterialDeviceResource* m_deviceResource;

    H_CALLABLE float roughnessFromExponent(float exponent) const;
    D_CALLABLE glm::vec3 diffuse(const glm::vec2& tc) const;
    D_CALLABLE glm::vec3 glossy(const glm::vec2& tc) const;
  };

  // Evaluates material at a hitPoint. Gives the color of that point
  inline glm::vec3 CMaterial::f(const glm::vec2& tc, const glm::vec3& wo, const glm::vec3& wi) const {
    glm::vec3 fDiffuse = m_orenNayarBRDF.f(wo, wi);
    glm::vec3 fGlossy = m_microfacetBRDF.f(wo, wi);
    return 0.5f * (diffuse(tc) * fDiffuse + glossy(tc) * fGlossy);
  }

  inline CMaterial& CMaterial::operator=(const CMaterial& material) {
    this->m_Le = material.m_Le;
    this->m_diffuseColor = material.m_diffuseColor;
    this->m_glossyColor = material.m_glossyColor;
    this->m_orenNayarBRDF = material.m_orenNayarBRDF;
    this->m_microfacetBRDF = material.m_microfacetBRDF;
    this->m_albedoTexture = material.m_albedoTexture;
    return *this;
  }

  inline glm::vec3 CMaterial::sampleF(const glm::vec2& tc, const glm::vec3& wo, glm::vec3* wi, CSampler& sampler, float* pdf) const {
    if (sampler.uniformSample01() < 0.5f) {
      // Sample diffuse
      return diffuse(tc) * m_orenNayarBRDF.sampleF(wo, wi, sampler, pdf);
    }
    else {
      // Sample specular
      return glossy(tc) * m_microfacetBRDF.sampleF(wo, wi, sampler, pdf);
    }
  }

  inline float CMaterial::pdf(const glm::vec3& wo, const glm::vec3& wi) const {
    return 0.5f * (m_orenNayarBRDF.pdf(wo, wi) + m_microfacetBRDF.pdf(wo, wi));
  }


  DH_CALLABLE inline glm::vec3 CMaterial::Le() const {
    return m_Le;
  }

  inline glm::vec3 CMaterial::color() {
    return m_diffuseColor;
  }

  inline glm::vec3 CMaterial::diffuse(const glm::vec2& tc) const {
    if (m_albedoTexture) {
      return m_albedoTexture->operator()(tc.x, tc.y);
      //return m_diffuseColor;
    }
    else {
      return m_diffuseColor;
    }
  }

  inline glm::vec3 CMaterial::glossy(const glm::vec2& tc) const {
    return m_glossyColor;
  }

  inline bool CMaterial::opaque(const glm::vec2& tc) const {
    if (m_alphaTexture) {
      return m_alphaTexture->operator()(tc.x, tc.y).r > 0.f;
    }
    else {
      return true;
    }
  }

  inline glm::vec3 CMaterial::normalmap(const glm::vec3& n, const glm::vec2& tc) const {
    return m_normalTexture ? CCoordinateFrame::align(n, glm::normalize(m_normalTexture->operator()(tc.x, tc.y) * 2.f - 1.f)) : n;
  }

  inline glm::vec3 CMaterial::normalmap(const CCoordinateFrame& frame, const glm::vec2& tc) const {
    //return m_normalTexture ? CCoordinateFrame::align(n, glm::normalize(m_normalTexture->operator()(tc.x, tc.y) * 2.f - 1.f)) : n;
    return m_normalTexture ? frame.tangentToWorld() * (glm::vec4(0.f, 0.f, 1.f, 0.f) + glm::vec4(glm::normalize(m_normalTexture->operator()(tc.x, tc.y) * 2.f - 1.f), 0.f)) : frame.N();
  }

}
#endif // !MATERIAL_HXX
