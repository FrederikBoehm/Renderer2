#ifndef TEXTURE_HPP
#define TEXTURE_HPP
#include <string>
#include "utility/qualifiers.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <cuda_runtime.h>
#include "backend/asset_manager.hpp"
#include <optix/optix_types.h>
namespace rt {
  struct STexture_DeviceResource {
    cudaArray_t d_data;
    cudaTextureObject_t d_texObj;
  };

  class CTexture {
    friend class CAssetManager;
  public:
    DH_CALLABLE CTexture();
    H_CALLABLE CTexture(const std::string& path, ETextureType type = DIFFUSE);
    H_CALLABLE CTexture(CTexture&& texture);
    H_CALLABLE CTexture(const CTexture& texture);
    H_CALLABLE ~CTexture();
    H_CALLABLE CTexture& operator=(CTexture&& texture);
    D_CALLABLE glm::vec3 operator()(float x, float y) const;

    DH_CALLABLE int width() const;
    DH_CALLABLE int height() const;
    DH_CALLABLE int channels() const;
    DH_CALLABLE const float* data() const;
    DH_CALLABLE bool hasAlpha() const;
    H_CALLABLE std::string path() const;
    DH_CALLABLE ETextureType type() const;

    H_CALLABLE void allocateDeviceMemory();
    H_CALLABLE CTexture copyToDevice() const;
    H_CALLABLE void freeDeviceMemory() const;
    H_CALLABLE void loadAlpha(const std::string& path);
  private:
    bool m_hasAlpha;
    int m_width;
    int m_height;
    int m_channels;
    ETextureType m_type;
    float* m_data;
    cudaArray_t m_deviceData;
    cudaTextureObject_t m_deviceTextureObj;
    uint16_t m_pathLength;
    char* m_path;

    STexture_DeviceResource* m_deviceResource;

    H_CALLABLE static bool isTiff(const std::string& path);
  };

  inline glm::vec3 CTexture::operator()(float x, float y) const {
    float4 value = tex2D<float4>(m_deviceTextureObj, x, y);
    return glm::vec3(value.x, value.y, value.z);
  }

  DH_CALLABLE inline int CTexture::width() const {
    return m_width;
  }

  DH_CALLABLE inline int CTexture::height() const {
    return m_height;
  }

  DH_CALLABLE inline int CTexture::channels() const {
    return m_channels;
  }

  H_CALLABLE inline const float* CTexture::data() const {
    return m_data;
  }

  inline bool CTexture::hasAlpha() const {
    return m_hasAlpha;
  }

  inline std::string CTexture::path() const {
    return std::string(m_path, m_pathLength);
  }

  inline ETextureType CTexture::type() const {
    return m_type;
  }
}
#endif // !TEXTURE_HPP
