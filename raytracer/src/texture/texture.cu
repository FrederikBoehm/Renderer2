#include "texture/texture.hpp"

#include <glm/gtc/type_ptr.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#include <iostream>

namespace rt {
  CTexture::CTexture(): m_width(0), m_height(0), m_channels(0), m_hasAlpha(false), m_data(nullptr), m_deviceResource(nullptr) {

  }

  CTexture::CTexture(const std::string& path): m_channels(3), m_deviceResource(nullptr) {
    int numChannels;
    m_data = stbi_loadf(path.c_str(), &m_width, &m_height, &numChannels, 3);
    m_hasAlpha = numChannels == 4;
    if (!m_data) {
      std::cerr << "Failed to load texture from " << path << std::endl;
    }
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
    t.m_hasAlpha = m_hasAlpha;
    t.m_data = m_deviceResource->d_data; // TODO: Add nullptr check
    return t;
  }

  void CTexture::freeDeviceMemory() const {
    if (m_deviceResource) {
      cudaFree(m_deviceResource->d_data);
    }
  }

  void CTexture::loadAlpha(const std::string& path) {
    int numChannels;
    int forcedChannels = 4;
    float* imgData = stbi_loadf(path.c_str(), &m_width, &m_height, &numChannels, forcedChannels);
    if (!imgData) {
      std::cerr << "Failed to load texture from " << path << std::endl;
    }
    m_channels = 3;
    m_data = static_cast<float*>(malloc(3 * m_width * m_height * sizeof(float)));
    for (size_t i = 0; i < m_width * m_height; ++i) {
      float alpha = imgData[i * forcedChannels + (forcedChannels - 1)];
      m_data[3 * i + 0] = alpha;
      m_data[3 * i + 1] = alpha;
      m_data[3 * i + 2] = alpha;
    }
    stbi_image_free(imgData);
  }
}