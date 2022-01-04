#include "texture/texture_manager.hpp"
#include "texture/texture.hpp"
#include "utility/debugging.hpp"

namespace rt {
  std::unordered_map<STextureKey, CTexture*, STextureKey> CTextureManager::m_hostTextures;
  std::unordered_map<STextureKey, CTexture*, STextureKey> CTextureManager::m_deviceTextures;

  CTexture* CTextureManager::loadTexture(const std::string& path, ETextureType type) {
    auto texIter = m_hostTextures.find({ path, type });
    if (texIter != m_hostTextures.end()) {
      return texIter->second;
    }
    else {
      CTexture* tex = new CTexture(path, type);
      m_hostTextures[{path, type}] = tex;
      return tex;
    }
  }

  CTexture* CTextureManager::loadAlpha(const std::string& path) {
    auto texIter = m_hostTextures.find({ path, ALPHA });
    if (texIter != m_hostTextures.end()) {
      return texIter->second;
    }
    else {
      CTexture* tex = new CTexture();
      tex->loadAlpha(path);
      m_hostTextures[{path, ALPHA}] = tex;
      return tex;
    }
  }

  void CTextureManager::allocateDeviceMemory() {
    for (auto tex : m_hostTextures) {
      auto test = m_deviceTextures[tex.first];
      CUDA_ASSERT(cudaMalloc(&m_deviceTextures[tex.first], sizeof(CTexture)));
      tex.second->allocateDeviceMemory();
    }
  }

  void CTextureManager::copyToDevice() {
    for (auto tex : m_hostTextures) {
      CUDA_ASSERT(cudaMemcpy(m_deviceTextures[tex.first], &tex.second->copyToDevice(), sizeof(CTexture), cudaMemcpyHostToDevice));
    }
  }

  void CTextureManager::freeDeviceMemory() {
    for (auto tex : m_deviceTextures) {
      CUDA_ASSERT(cudaFree(tex.second));
    }
  }

  CTexture* CTextureManager::deviceTexture(const std::string& path, ETextureType type) {
    auto texIter = m_deviceTextures.find({ path, type });
    if (texIter != m_deviceTextures.end()) {
      return texIter->second;
    }
    else {
      return nullptr;
    }
  }
}