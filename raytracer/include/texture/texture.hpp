#ifndef TEXTURE_HPP
#define TEXTURE_HPP
#include <string>
#include "utility/qualifiers.hpp"
#include <glm/glm.hpp>
namespace rt {
  struct STexture_DeviceResource {
    float* d_data;
  };

  class CTexture {
  public:
    DH_CALLABLE CTexture();
    H_CALLABLE CTexture(const std::string& path);
    DH_CALLABLE ~CTexture();
    DH_CALLABLE glm::vec3 operator()(float x, float y) const;

    DH_CALLABLE int width() const;
    DH_CALLABLE int height() const;
    DH_CALLABLE int channels() const;
    DH_CALLABLE const float* data() const;

    H_CALLABLE void allocateDeviceMemory();
    H_CALLABLE CTexture copyToDevice() const;
    H_CALLABLE void freeDeviceMemory() const;
  private:
    int m_width;
    int m_height;
    int m_channels;
    float* m_data;

    STexture_DeviceResource* m_deviceResource;
  };

  DH_CALLABLE inline int CTexture::width() const {
    return m_width;
  }

  DH_CALLABLE inline int CTexture::height() const {
    return m_height;
  }

  DH_CALLABLE inline int CTexture::channels() const {
    return m_channels;
  }

  DH_CALLABLE inline const float* CTexture::data() const {
    return m_data;
  }
}
#endif // !TEXTURE_HPP
