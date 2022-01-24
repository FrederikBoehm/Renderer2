#include "texture/texture_manager.hpp"
#include "texture/texture.hpp"
#include "utility/debugging.hpp"

namespace rt {
  std::unordered_map<STextureKey, CTexture*, STextureKey> CTextureManager::s_hostTextures;
  std::unordered_map<STextureKey, CTexture*, STextureKey> CTextureManager::s_deviceTextures;

  CTexture* CTextureManager::loadTexture(const std::string& path, ETextureType type) {
    auto texIter = s_hostTextures.find({ path, type });
    if (texIter != s_hostTextures.end()) {
      return texIter->second;
    }
    else {
      CTexture* tex = new CTexture(path, type);
      s_hostTextures[{path, type}] = tex;
      return tex;
    }
  }

  CTexture* CTextureManager::loadAlpha(const std::string& path) {
    auto texIter = s_hostTextures.find({ path, ALPHA });
    if (texIter != s_hostTextures.end()) {
      return texIter->second;
    }
    else {
      CTexture* tex = new CTexture();
      tex->loadAlpha(path);
      s_hostTextures[{path, ALPHA}] = tex;
      return tex;
    }
  }

  void CTextureManager::allocateDeviceMemory() {
    for (auto tex : s_hostTextures) {
      CUDA_ASSERT(cudaMalloc(&s_deviceTextures[tex.first], sizeof(CTexture)));
      tex.second->allocateDeviceMemory();
    }
  }

  void CTextureManager::copyToDevice() {
    for (auto tex : s_hostTextures) {
      CUDA_ASSERT(cudaMemcpy(s_deviceTextures[tex.first], &tex.second->copyToDevice(), sizeof(CTexture), cudaMemcpyHostToDevice));
    }
  }

  void CTextureManager::freeDeviceMemory() {
    for (auto tex : s_deviceTextures) {
      s_hostTextures[tex.first]->freeDeviceMemory();
      CUDA_ASSERT(cudaFree(tex.second));
    }
  }

  CTexture* CTextureManager::deviceTexture(const std::string& path, ETextureType type) {
    auto texIter = s_deviceTextures.find({ path, type });
    if (texIter != s_deviceTextures.end()) {
      return texIter->second;
    }
    else {
      return nullptr;
    }
  }
}