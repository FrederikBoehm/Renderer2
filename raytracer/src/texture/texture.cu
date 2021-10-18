#include "texture/texture.hpp"

#include <glm/gtc/type_ptr.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

namespace rt {
  CTexture::CTexture(): m_width(0), m_height(0), m_channels(0), m_data(nullptr), m_deviceResource(nullptr) {

  }

  CTexture::CTexture(const std::string& path): m_deviceResource(nullptr) {
    m_data = stbi_loadf(path.c_str(), &m_width, &m_height, &m_channels, 0);
  }

  CTexture::~CTexture() {
#ifndef __CUDA_ARCH__
    //delete[] m_data; // TODO: clean up object
#endif
    if (m_deviceResource) {
      freeDeviceMemory();
      delete m_deviceResource;
    }
  }

  glm::vec3 CTexture::operator()(float x, float y) const {
    int lowerRowIndex = y * (m_height-1);
    int upperRowIndex = lowerRowIndex + 1;

    int lowerColumnIndex = x * (m_width-1);
    int upperColumnIndex = lowerColumnIndex + 1;

    glm::vec3 lowerRowInterpolation = glm::make_vec3(&m_data[lowerRowIndex * m_width * m_channels + lowerColumnIndex * m_channels]) * (upperColumnIndex - x * (m_width - 1)) +
                                      glm::make_vec3(&m_data[lowerRowIndex * m_width * m_channels + upperColumnIndex * m_channels]) * (x * (m_width - 1) - lowerColumnIndex);
    glm::vec3 upperRowInterpolation = glm::make_vec3(&m_data[upperRowIndex * m_width * m_channels + lowerColumnIndex * m_channels]) * (upperColumnIndex - x * (m_width - 1)) +
                                      glm::make_vec3(&m_data[upperRowIndex * m_width * m_channels + upperColumnIndex * m_channels]) * (x * (m_width - 1) - lowerColumnIndex);

    return lowerRowInterpolation * (upperRowIndex - y * (m_height - 1)) + upperRowInterpolation * (y * (m_height - 1) - lowerRowIndex);
  }

  void CTexture::allocateDeviceMemory() {
    if (m_deviceResource) {
      freeDeviceMemory();
      delete m_deviceResource;
    }
    m_deviceResource = new STexture_DeviceResource;
    cudaMalloc(&m_deviceResource->d_data, sizeof(float) * m_width * m_height * m_channels);
  }

  CTexture CTexture::copyToDevice() const {
    if (m_deviceResource) {
      cudaMemcpy(m_deviceResource->d_data, m_data, sizeof(float) * m_width * m_height * m_channels, cudaMemcpyHostToDevice);
    }

    CTexture t;
    t.m_width = m_width;
    t.m_height = m_height;
    t.m_channels = m_channels;
    t.m_data = m_deviceResource->d_data;
    return t;
  }

  void CTexture::freeDeviceMemory() const {
    if (m_deviceResource) {
      cudaFree(m_deviceResource->d_data);
    }
  }
}