#ifndef TEXTURE_HPP
#define TEXTURE_HPP
#include <string>
#include "utility/qualifiers.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
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
    DH_CALLABLE bool hasAlpha() const;

    H_CALLABLE void allocateDeviceMemory();
    H_CALLABLE CTexture copyToDevice() const;
    H_CALLABLE void freeDeviceMemory() const;
    H_CALLABLE void loadAlpha(const std::string& path);
  private:
    int m_width;
    int m_height;
    int m_channels;
    bool m_hasAlpha;
    float* m_data;

    STexture_DeviceResource* m_deviceResource;
  };

  inline glm::vec3 CTexture::operator()(float x, float y) const {
    x = glm::fract(x);
    y = glm::fract(y);
    int lowerRowIndex = glm::clamp(int(y * (m_height - 2)), 0, m_height - 2);
    int upperRowIndex = lowerRowIndex + 1;

    int lowerColumnIndex = glm::clamp(int(x * (m_width - 2)), 0, m_width - 2);
    int upperColumnIndex = lowerColumnIndex + 1;

    glm::vec3 lowerRowInterpolation = glm::make_vec3(&m_data[lowerRowIndex * m_width * m_channels + lowerColumnIndex * m_channels]) * (upperColumnIndex - x * (m_width - 2)) +
      glm::make_vec3(&m_data[lowerRowIndex * m_width * m_channels + upperColumnIndex * m_channels]) * (x * (m_width - 2) - lowerColumnIndex);
    glm::vec3 upperRowInterpolation = glm::make_vec3(&m_data[upperRowIndex * m_width * m_channels + lowerColumnIndex * m_channels]) * (upperColumnIndex - x * (m_width - 2)) +
      glm::make_vec3(&m_data[upperRowIndex * m_width * m_channels + upperColumnIndex * m_channels]) * (x * (m_width - 2) - lowerColumnIndex);

    return lowerRowInterpolation * (upperRowIndex - y * (m_height - 2)) + upperRowInterpolation * (y * (m_height - 2) - lowerRowIndex);
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

  DH_CALLABLE inline const float* CTexture::data() const {
    return m_data;
  }

  inline bool CTexture::hasAlpha() const {
    return m_hasAlpha;
  }
}
#endif // !TEXTURE_HPP
