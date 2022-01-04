#ifndef TEXTURE_MANAGER_HPP
#define TEXTURE_MANAGER_HPP
#include <unordered_map>
#include <string>
#include "utility/qualifiers.hpp"
#include <functional>
namespace rt {
  class CTexture;

  enum ETextureType {
    DIFFUSE,
    NORMAL,
    ALPHA
  };

  struct STextureKey {
    std::string m_path;
    ETextureType m_type;

    std::size_t operator()(const STextureKey& textureKey) const noexcept {
      std::size_t h1 = std::hash<std::string>{}(textureKey.m_path);
      std::size_t h2 = std::hash<size_t>{}(m_type);
      return h1 ^ (h2 << 1);
    }
  };

  inline bool operator==(const STextureKey& l, const STextureKey& r) {
    return l.m_path == r.m_path && l.m_type == r.m_type;
  }

  inline bool operator!=(const STextureKey& l, const STextureKey& r) {
    return !(l == r);
  }

  class CTextureManager {
  public:
    H_CALLABLE static CTexture* loadTexture(const std::string& path, ETextureType type);
    H_CALLABLE static CTexture* loadAlpha(const std::string& path);
    H_CALLABLE static void allocateDeviceMemory();
    H_CALLABLE static void copyToDevice();
    H_CALLABLE static void freeDeviceMemory();
    H_CALLABLE static CTexture* deviceTexture(const std::string& path, const ETextureType type);
  private:
    static std::unordered_map<STextureKey, CTexture*, STextureKey> m_hostTextures;
    static std::unordered_map<STextureKey, CTexture*, STextureKey> m_deviceTextures;

    H_CALLABLE CTextureManager() = delete;
  };
}
#endif // !TEXTURE_MANAGER_HPP
