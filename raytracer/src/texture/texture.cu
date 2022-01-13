#include "texture/texture.hpp"

#include <glm/gtc/type_ptr.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#include <iostream>
#include "utility/debugging.hpp"

namespace rt {
  CTexture::CTexture() : m_hasAlpha(false), m_width(0), m_height(0), m_channels(0), m_pathLength(0), m_path(nullptr), m_data(nullptr), m_deviceData(nullptr), m_deviceResource(nullptr) {

  }

  float srgbToLinear(float srgb) {
    if (srgb <= 0.04045f) {
      return srgb / 12.92f;
    }
    else {
      return std::pow((srgb + 0.055f) / 1.055f, 2.4f);
    }
  }

  CTexture::CTexture(const std::string& path, ETextureType type): m_channels(4), m_type(type), m_deviceData(nullptr), m_deviceTextureObj(NULL), m_deviceResource(nullptr) {
    m_pathLength = path.size();
    m_path = new char[m_pathLength];
    memcpy(m_path, path.data(), m_pathLength);
    int numChannels;
    bool hdr = stbi_is_hdr(path.c_str());
    if (hdr) {
      float* data = stbi_loadf(path.c_str(), &m_width, &m_height, &numChannels, 3);
      m_data = static_cast<float*>(malloc(sizeof(float) * m_width * m_height * 4));
      for (size_t i = 0; i < m_width * m_height; ++i) {
        m_data[4 * i + 0] = data[3 * i + 0];
        m_data[4 * i + 1] = data[3 * i + 1];
        m_data[4 * i + 2] = data[3 * i + 2];
        m_data[4 * i + 3] = 1.f;
      }
    }
    else {
      uint8_t* data = static_cast<uint8_t*>(stbi_load(path.c_str(), &m_width, &m_height, &numChannels, 3));
      m_data = static_cast<float*>(malloc(sizeof(float) * m_width * m_height * 4));
      for (size_t i = 0; i < m_width * m_height; ++i) {
        m_data[4 * i + 0] = data[3 * i + 0] / 255.f;
        m_data[4 * i + 1] = data[3 * i + 1] / 255.f;
        m_data[4 * i + 2] = data[3 * i + 2] / 255.f;
        m_data[4 * i + 3] = 1.f;
      }
    }
    m_hasAlpha = numChannels == 4;
    if (!m_data) {
      std::cerr << "Failed to load texture from " << path << std::endl;
    }
  }

  CTexture::~CTexture() {
#ifndef __CUDA_ARCH__
    //delete[] m_data; // TODO: clean up object
#endif
    delete[] m_path;
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
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    cudaMallocArray(&m_deviceResource->d_data, &channelDesc, m_width, m_height);
  }

  CTexture CTexture::copyToDevice() const {
    if (m_deviceResource) {
      const size_t spitch = m_width * sizeof(float) * 4;
      cudaMemcpy2DToArray(m_deviceResource->d_data, 0, 0, m_data, spitch, m_width * sizeof(float) * 4, m_height, cudaMemcpyHostToDevice); // TODO: Can we ensure alignment without vec4 per pixel?
      
      cudaResourceDesc resDesc;
      memset(&resDesc, 0, sizeof(cudaResourceDesc));
      resDesc.resType = cudaResourceTypeArray;
      resDesc.res.array.array = m_deviceResource->d_data;

      cudaTextureDesc texDesc;
      memset(&texDesc, 0, sizeof(cudaTextureDesc));
      texDesc.addressMode[0] = cudaAddressModeWrap;
      texDesc.addressMode[1] = cudaAddressModeWrap;
      texDesc.filterMode = cudaFilterModeLinear;
      texDesc.readMode = cudaReadModeElementType;
      texDesc.normalizedCoords = 1;

      cudaCreateTextureObject(&m_deviceResource->d_texObj, &resDesc, &texDesc, NULL);
    }


    CTexture t;
    t.m_width = m_width;
    t.m_height = m_height;
    t.m_channels = m_channels;
    t.m_hasAlpha = m_hasAlpha;
    t.m_data = nullptr;
    t.m_deviceData = m_deviceResource ? m_deviceResource->d_data : nullptr;
    t.m_deviceTextureObj = m_deviceResource ? m_deviceResource->d_texObj : NULL;
    t.m_pathLength = 0;
    t.m_path = nullptr;
    return t;
  }

  void CTexture::freeDeviceMemory() const {
    if (m_deviceResource) {
      CUDA_ASSERT(cudaDestroyTextureObject(m_deviceResource->d_texObj));
      CUDA_ASSERT(cudaFreeArray(m_deviceResource->d_data));
    }
  }

  void CTexture::loadAlpha(const std::string& path) {
    int numChannels;
    int forcedChannels = 4;
    float* imgData = stbi_loadf(path.c_str(), &m_width, &m_height, &numChannels, forcedChannels);
    if (!imgData) {
      std::cerr << "Failed to load texture from " << path << std::endl;
    }
    m_channels = 4;
    m_pathLength = path.size();
    m_path = new char[m_pathLength];
    memcpy(m_path, path.data(), m_pathLength);
    m_type = ALPHA;
    m_data = static_cast<float*>(malloc(4 * m_width * m_height * sizeof(float)));
    for (size_t i = 0; i < m_width * m_height; ++i) {
      float alpha = imgData[i * forcedChannels + (forcedChannels - 1)];
      m_data[4 * i + 0] = alpha;
      m_data[4 * i + 1] = alpha;
      m_data[4 * i + 2] = alpha;
      m_data[4 * i + 3] = 1.f;
    }
    stbi_image_free(imgData);
  }
}