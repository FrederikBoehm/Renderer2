#include "texture/texture.hpp"

#include <glm/gtc/type_ptr.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#include <iostream>
#include "utility/debugging.hpp"

extern "C" {
#include <libtiff/libtiff/tiffio.h>
}

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
    if (CTexture::isTiff(path)) {
      TIFF* tif = TIFFOpen(path.c_str(), "r");
      if (!tif) {
        delete m_path;
        throw std::runtime_error("Failed to load texture from " + path);
      }
      TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &m_width);
      TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &m_height);
      size_t numPixels = m_width * m_height;
      uint32_t* data = (uint32_t*)_TIFFmalloc(numPixels * sizeof(uint32_t));
      m_data = static_cast<float*>(malloc(sizeof(float) * numPixels * 4));
      if (data != NULL) {
        if (TIFFReadRGBAImageOriented(tif, m_width, m_height, data, ORIENTATION_TOPLEFT, 0)) {
          for (size_t i = 0; i < numPixels; ++i) {
            uint8_t* red = reinterpret_cast<uint8_t*>(data + i);
            uint8_t* green = red + 1;
            uint8_t* blue = green + 1;
            uint8_t* alpha = blue + 1;
            m_data[4 * i + 0] = *red / 255.f;
            m_data[4 * i + 1] = *green / 255.f;
            m_data[4 * i + 2] = *blue / 255.f;
            m_data[4 * i + 3] = 1.f;
            m_hasAlpha = std::round(*alpha / 255.f) == 0.f; // We don't support fractional alphas
          }
        }
        _TIFFfree(data);
      }
      TIFFClose(tif);
    }
    else {
      int numChannels;
      bool hdr = stbi_is_hdr(path.c_str());
      if (hdr) {
        float* data = stbi_loadf(path.c_str(), &m_width, &m_height, &numChannels, 3);
        if (!data) {
          delete m_path;
          throw std::runtime_error("Failed to load texture from " + path);
        }
        size_t numPixels = m_width * m_height;
        m_data = static_cast<float*>(malloc(sizeof(float) * numPixels * 4));
        for (size_t i = 0; i < numPixels; ++i) {
          m_data[4 * i + 0] = data[3 * i + 0];
          m_data[4 * i + 1] = data[3 * i + 1];
          m_data[4 * i + 2] = data[3 * i + 2];
          m_data[4 * i + 3] = 1.f;
        }
        stbi_image_free(data);
      }
      else {
        uint8_t* data = static_cast<uint8_t*>(stbi_load(path.c_str(), &m_width, &m_height, &numChannels, 3));
        if (!data) {
          delete m_path;
          throw std::runtime_error("Failed to load texture from " + path);
        }
        size_t numPixels = m_width * m_height;
        m_data = static_cast<float*>(malloc(sizeof(float) * numPixels * 4));
        for (size_t i = 0; i < numPixels; ++i) {
          m_data[4 * i + 0] = data[3 * i + 0] / 255.f;
          m_data[4 * i + 1] = data[3 * i + 1] / 255.f;
          m_data[4 * i + 2] = data[3 * i + 2] / 255.f;
          m_data[4 * i + 3] = 1.f;
        }
        stbi_image_free(data);
      }
      m_hasAlpha = numChannels == 4;
    }
    if (!m_data) {
      std::cerr << "Failed to load texture from " << path << std::endl;
    }
  }

  CTexture::CTexture(const CTexture& texture):
    m_hasAlpha(texture.m_hasAlpha),
    m_width(texture.m_width),
    m_height(texture.m_height),
    m_channels(texture.m_channels),
    m_pathLength(texture.m_pathLength),
    m_path((char*)malloc(texture.m_pathLength)),
    m_data((float*)malloc(sizeof(float) * m_width * m_height * 4)),
    m_deviceData(nullptr),
    m_deviceResource(nullptr),
    m_deviceTextureObj(NULL) {
    memcpy(m_path, texture.m_path, m_pathLength);
    memcpy(m_data, texture.m_data, sizeof(float) * m_width * m_height * 4);
  }

  CTexture::CTexture(CTexture&& texture) :
    m_hasAlpha(std::move(texture.m_hasAlpha)),
    m_width(std::move(texture.m_width)),
    m_height(std::move(texture.m_height)),
    m_channels(std::move(texture.m_channels)),
    m_pathLength(std::move(texture.m_pathLength)),
    m_path(std::exchange(texture.m_path, nullptr)),
    m_data(std::exchange(texture.m_data, nullptr)),
    m_deviceData(std::exchange(texture.m_deviceData, nullptr)),
    m_deviceResource(std::exchange(texture.m_deviceResource, nullptr)),
    m_deviceTextureObj(std::exchange(texture.m_deviceTextureObj, NULL)) {

  }

  CTexture::~CTexture() {
    delete[] m_data;
    delete[] m_path;
    if (m_deviceResource) {
      freeDeviceMemory();
      delete m_deviceResource;
    }
  }

  CTexture& CTexture::operator=(CTexture&& texture) {
    m_hasAlpha = std::move(texture.m_hasAlpha);
    m_width = std::move(texture.m_width);
    m_height = std::move(texture.m_height);
    m_channels = std::move(texture.m_channels);
    m_pathLength = std::move(texture.m_pathLength);
    m_path = std::exchange(texture.m_path, nullptr);
    m_data = std::exchange(texture.m_data, nullptr);
    m_deviceData = std::exchange(texture.m_deviceData, nullptr);
    m_deviceResource = std::exchange(texture.m_deviceResource, nullptr);
    m_deviceTextureObj = std::exchange(texture.m_deviceTextureObj, NULL);
    
    return *this;
  }

  bool CTexture::isTiff(const std::string& path) {
    std::string tif = ".tif";
    std::string tiff = ".tiff";
    std::string tifSubstring = path.substr(path.size() - tif.size());
    std::string tiffSubstring = path.substr(path.size() - tiff.size());
    return tifSubstring == tif || tiffSubstring == tiff;
  }
  

  void CTexture::allocateDeviceMemory() {
    if (m_deviceResource) {
      freeDeviceMemory();
      delete m_deviceResource;
    }
    m_deviceResource = new STexture_DeviceResource;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    CUDA_ASSERT(cudaMallocArray(&m_deviceResource->d_data, &channelDesc, m_width, m_height));

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

    CUDA_ASSERT(cudaCreateTextureObject(&m_deviceResource->d_texObj, &resDesc, &texDesc, NULL));
  }

  CTexture CTexture::copyToDevice() const {
    if (m_deviceResource) {
      const size_t spitch = m_width * sizeof(float) * 4;
      CUDA_ASSERT(cudaMemcpy2DToArray(m_deviceResource->d_data, 0, 0, m_data, spitch, m_width * sizeof(float) * 4, m_height, cudaMemcpyHostToDevice));
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
    if (CTexture::isTiff(path)) {
      TIFF* tif = TIFFOpen(path.c_str(), "r");
      if (!tif) {
        throw std::runtime_error("Failed to load texture from " + path);
      }
      m_channels = 4;
      m_pathLength = path.size();
      m_path = new char[m_pathLength];
      memcpy(m_path, path.data(), m_pathLength);
      m_type = ALPHA;

      TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &m_width);
      TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &m_height);
      size_t numPixels = m_width * m_height;
      uint32_t* data = (uint32_t*)_TIFFmalloc(numPixels * sizeof(uint32_t));
      m_data = static_cast<float*>(malloc(sizeof(float) * numPixels * 4));
      if (data != NULL) {
        if (TIFFReadRGBAImageOriented(tif, m_width, m_height, data, ORIENTATION_TOPLEFT, 0)) {
          for (size_t i = 0; i < numPixels; ++i) {
            uint8_t* alpha = reinterpret_cast<uint8_t*>(data + i) + 3;
            float alphaF = std::round(*alpha / 255.f);
            m_data[4 * i + 0] = alphaF;
            m_data[4 * i + 1] = alphaF;
            m_data[4 * i + 2] = alphaF;
            m_data[4 * i + 3] = 1.f;
          }
        }
        _TIFFfree(data);
      }
      TIFFClose(tif);
    }
    else {
      int numChannels;
      int forcedChannels = 4;
      float* imgData = stbi_loadf(path.c_str(), &m_width, &m_height, &numChannels, forcedChannels);
      if (!imgData) {
        throw std::runtime_error("Failed to load texture from " + path);
      }
      m_channels = 4;
      m_pathLength = path.size();
      m_path = new char[m_pathLength];
      memcpy(m_path, path.data(), m_pathLength);
      m_type = ALPHA;
      size_t numPixels = m_width * m_height;
      m_data = static_cast<float*>(malloc(4 * numPixels * sizeof(float)));
      for (size_t i = 0; i < numPixels; ++i) {
        float alpha = imgData[i * forcedChannels + (forcedChannels - 1)];
        m_data[4 * i + 0] = alpha;
        m_data[4 * i + 1] = alpha;
        m_data[4 * i + 2] = alpha;
        m_data[4 * i + 3] = 1.f;
      }
      stbi_image_free(imgData);
    }

  }
}