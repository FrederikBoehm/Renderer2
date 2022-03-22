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

  class CMaterial {
  public:
    H_CALLABLE CMaterial();
    H_CALLABLE CMaterial(const glm::vec3& le);
    H_CALLABLE CMaterial(const glm::vec3& diffuseColor, const glm::vec3& glossyColor, const COrenNayarBRDF& diffuseBRDF, const CMicrofacetBRDF& glossy);
    H_CALLABLE CMaterial(const aiMaterial* material, const std::string& assetsBasepath, const std::string& fullPath, size_t submeshId);
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
    H_CALLABLE std::string path() const;
    H_CALLABLE size_t submeshId() const;

    D_CALLABLE glm::vec3 diffuse(const glm::vec2& tc) const;
    D_CALLABLE glm::vec3 glossy(const glm::vec2& tc) const;
    DH_CALLABLE float specularRoughness() const;
  private:
    bool m_deviceObject;
    glm::vec3 m_Le; // Emissive light if light source
    glm::vec3 m_diffuseColor;
    glm::vec3 m_glossyColor;
    COrenNayarBRDF m_orenNayarBRDF;
    CMicrofacetBRDF m_microfacetBRDF;
    CTexture* m_albedoTexture;
    CTexture* m_glossyTexture;
    CTexture* m_normalTexture;
    CTexture* m_alphaTexture;

    uint16_t m_pathLength;
    char* m_meshPath;
    size_t m_submeshId; // Id of corresponding mesh

    H_CALLABLE float roughnessFromExponent(float exponent) const;
  };

  // Evaluates material at a hitPoint. Gives the color of that point
  inline glm::vec3 CMaterial::f(const glm::vec2& tc, const glm::vec3& wo, const glm::vec3& wi) const {
    float fDiffuse = m_orenNayarBRDF.f(wo, wi);
    float fGlossy = m_microfacetBRDF.f(wo, wi);
    //float weight = 0.5f;
    float weight = m_microfacetBRDF.fresnel().evaluate(absCosTheta(wo));
    return diffuse(tc) * fDiffuse * (1.f - weight) + glossy(tc) * fGlossy * weight;
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
    float p = m_microfacetBRDF.fresnel().evaluate(absCosTheta(wo));
    //float p = 0.5f;
    if (sampler.uniformSample01() < p) {
      // Sample specular
      return glossy(tc) * m_microfacetBRDF.sampleF(wo, wi, sampler, pdf);
    }
    else {
      // Sample diffuse
      return diffuse(tc) * m_orenNayarBRDF.sampleF(wo, wi, sampler, pdf);
    }
  }

  inline float CMaterial::pdf(const glm::vec3& wo, const glm::vec3& wi) const {
    float weight = m_microfacetBRDF.fresnel().evaluate(absCosTheta(wo));
    //float weight = 0.5f;
    return m_orenNayarBRDF.pdf(wo, wi) * (1.f - weight) + m_microfacetBRDF.pdf(wo, wi) * weight;
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
    if (m_glossyTexture) {
      return m_glossyTexture->operator()(tc.x, tc.y);
    }
    else {
      return m_glossyColor;
    }
  }

  inline float CMaterial::specularRoughness() const {
    return m_microfacetBRDF.roughness();
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
    if (m_normalTexture) {
      glm::vec3 normal = m_normalTexture->operator()(tc.x, tc.y) * 2.f - 1.f;
      if (normal == glm::vec3(0.f)) {
        return frame.N();
      }
      else {
        return frame.tangentToWorld() * glm::normalize(glm::vec4(normal, 0.f));
      }
    }
    else {
      return frame.N();
    }
  }

  inline std::string CMaterial::path() const {
    return std::string(m_meshPath, m_pathLength);
  }

  inline size_t CMaterial::submeshId() const {
    return m_submeshId;
  }

}
#endif // !MATERIAL_HXX
