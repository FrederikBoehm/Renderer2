#include "material/material.hpp"
#include <assimp/material.h>
#include "texture/texture.hpp"
#include "utility/debugging.hpp"

namespace rt {
  CMaterial::CMaterial() :
    m_Le(glm::vec3(0.0f)),
    m_diffuseColor(0.f),
    m_glossyColor(0.f),
    m_orenNayarBRDF(),
    m_microfacetBRDF(),
    m_albedoTexture(nullptr),
    m_normalTexture(nullptr),
    m_alphaTexture(nullptr),
    m_deviceResource(nullptr) {
  }

  CMaterial::CMaterial(const glm::vec3& le) :
    m_Le(le),
    m_diffuseColor(0.f),
    m_glossyColor(0.f),
    m_orenNayarBRDF(),
    m_microfacetBRDF(),
    m_albedoTexture(nullptr),
    m_normalTexture(nullptr),
    m_alphaTexture(nullptr),
    m_deviceResource(nullptr) {

  }

  CMaterial::CMaterial(const glm::vec3& diffuseColor, const glm::vec3& glossyColor, const COrenNayarBRDF& diffuse, const CMicrofacetBRDF& glossy) :
    m_Le(glm::vec3(0.0f)),
    m_diffuseColor(diffuseColor),
    m_glossyColor(glossyColor),
    m_orenNayarBRDF(diffuse),
    m_microfacetBRDF(glossy),
    m_albedoTexture(nullptr),
    m_normalTexture(nullptr),
    m_alphaTexture(nullptr),
    m_deviceResource(nullptr) {

  }

  CMaterial::CMaterial(CMaterial&& material) :
    m_Le(std::move(material.m_Le)),
    m_diffuseColor(std::move(material.m_diffuseColor)),
    m_glossyColor(std::move(material.m_glossyColor)),
    m_orenNayarBRDF(std::move(material.m_orenNayarBRDF)),
    m_microfacetBRDF(std::move(material.m_microfacetBRDF)),
    m_albedoTexture(std::exchange(material.m_albedoTexture, nullptr)),
    m_normalTexture(std::exchange(material.m_normalTexture, nullptr)),
    m_alphaTexture(std::exchange(material.m_alphaTexture, nullptr)),
    m_deviceResource(std::exchange(material.m_deviceResource, nullptr)) {
  }

  CMaterial::~CMaterial() {
    if (m_deviceResource) {
      freeDeviceMemory();
      delete m_deviceResource;
    }
  }

  CMaterial::CMaterial(const aiMaterial* material, const std::string& assetsBasepath):
    m_Le(0.f),
    m_albedoTexture(nullptr),
    m_normalTexture(nullptr),
    m_alphaTexture(nullptr),
    m_deviceResource(nullptr) {
    aiColor3D diffuse;
    material->Get(AI_MATKEY_COLOR_DIFFUSE, diffuse);
    m_diffuseColor = glm::vec3(diffuse.r, diffuse.g, diffuse.b);
    m_orenNayarBRDF = COrenNayarBRDF(0.f);
    aiColor3D specular;
    material->Get(AI_MATKEY_COLOR_SPECULAR, specular);

    float exponent;
    material->Get(AI_MATKEY_SHININESS, exponent);
    float roughness = roughnessFromExponent(exponent);

    float ior;
    material->Get(AI_MATKEY_REFRACTI, ior);
    m_glossyColor = glm::vec3(specular.r, specular.g, specular.b);
    m_microfacetBRDF = CMicrofacetBRDF(roughness, roughness, 1.00029f, ior);

    std::string diffuseTexPath = "";
    if (material->GetTextureCount(aiTextureType_DIFFUSE) > 0) {
      aiString pathAi;
      material->GetTexture(aiTextureType_DIFFUSE, 0, &pathAi);
      diffuseTexPath = assetsBasepath + "/" + pathAi.C_Str();
      m_albedoTexture = new CTexture(diffuseTexPath);
    }

    if (material->GetTextureCount(aiTextureType_HEIGHT) > 0) {
      aiString pathAi;
      material->GetTexture(aiTextureType_HEIGHT, 0, &pathAi);
      m_normalTexture = new CTexture(assetsBasepath + "/" + pathAi.C_Str());
    }

    if (material->GetTextureCount(aiTextureType_OPACITY) > 0) {
      aiString pathAi;
      material->GetTexture(aiTextureType_OPACITY, 0, &pathAi);
      m_alphaTexture = new CTexture(assetsBasepath + "/" + pathAi.C_Str());
    }
    else if (m_albedoTexture && m_albedoTexture->hasAlpha()) {
      m_alphaTexture = new CTexture();
      m_alphaTexture->loadAlpha(diffuseTexPath);
    }
  }

  float CMaterial::roughnessFromExponent(float exponent) const {
    return powf(2.f / (exponent + 2.f), 0.5f);
  }

  void CMaterial::allocateDeviceMemory() {
    if (!m_deviceResource) {
      m_deviceResource = new SMaterialDeviceResource();
      
      if (m_albedoTexture) {
        CUDA_ASSERT(cudaMalloc(&m_deviceResource->d_albedoTexture, sizeof(CTexture)));
        m_albedoTexture->allocateDeviceMemory();
      }

      if (m_normalTexture) {
        CUDA_ASSERT(cudaMalloc(&m_deviceResource->d_normalTexture, sizeof(CTexture)));
        m_normalTexture->allocateDeviceMemory();
      }

      if (m_alphaTexture) {
        CUDA_ASSERT(cudaMalloc(&m_deviceResource->d_alphaTexture, sizeof(CTexture)));
        m_alphaTexture->allocateDeviceMemory();
      }
    }
  }

  CMaterial CMaterial::copyToDevice() {
    CMaterial material;
    material.m_Le = m_Le;
    material.m_diffuseColor = m_diffuseColor;
    material.m_glossyColor = m_glossyColor;
    material.m_orenNayarBRDF = m_orenNayarBRDF;
    material.m_microfacetBRDF = m_microfacetBRDF;
    material.m_albedoTexture = nullptr;
    material.m_normalTexture = nullptr;
    material.m_alphaTexture = nullptr;
    material.m_deviceResource = nullptr;
    if (m_deviceResource) {
      if (m_albedoTexture) {
        CUDA_ASSERT(cudaMemcpy(m_deviceResource->d_albedoTexture, &m_albedoTexture->copyToDevice(), sizeof(CTexture), cudaMemcpyHostToDevice));
      }
      material.m_albedoTexture = m_deviceResource->d_albedoTexture;

      if (m_normalTexture) {
        CUDA_ASSERT(cudaMemcpy(m_deviceResource->d_normalTexture, &m_normalTexture->copyToDevice(), sizeof(CTexture), cudaMemcpyHostToDevice));
      }
      material.m_normalTexture = m_deviceResource->d_normalTexture;

      if (m_alphaTexture) {
        CUDA_ASSERT(cudaMemcpy(m_deviceResource->d_alphaTexture, &m_alphaTexture->copyToDevice(), sizeof(CTexture), cudaMemcpyHostToDevice));
      }
      material.m_alphaTexture = m_deviceResource->d_alphaTexture;
    }
    return material;
  }

  void CMaterial::freeDeviceMemory() {
    if (m_deviceResource) {
      CUDA_ASSERT(cudaFree(m_deviceResource->d_albedoTexture));
      CUDA_ASSERT(cudaFree(m_deviceResource->d_normalTexture));
      CUDA_ASSERT(cudaFree(m_deviceResource->d_alphaTexture));
    }
  }
}