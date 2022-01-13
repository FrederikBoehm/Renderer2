#define _USE_MATH_DEFINES
#include <math.h>
#include "scene/environmentmap.hpp"
#include "utility/debugging.hpp"

namespace rt {
  CEnvironmentMap::CEnvironmentMap() {

  }

  CEnvironmentMap::CEnvironmentMap(const std::string& path):
    m_texture(path),
    m_dist(CDistribution2D(m_texture.width(), m_texture.height(), radiancePerPixel(m_texture))) {
  }

  std::vector<float> CEnvironmentMap::radiancePerPixel(const CTexture& texture) const {
    std::vector<float> radiance;
    radiance.reserve(texture.width() * texture.height());
    for (size_t i = 0; i < texture.channels() * texture.height() * texture.width(); i += texture.channels()) {
      float pixelRadiance = 0.f;
      for (uint8_t j = 0; j < 3; ++j) {
        pixelRadiance += texture.data()[i + j];
      }
      uint16_t row = i / (texture.channels() * texture.width());
      float sinTheta = glm::sin(M_PI * float(row + 0.5f) / (float)texture.height());
      radiance.push_back(pixelRadiance * sinTheta);
    }
    return radiance;
  }

  

  void CEnvironmentMap::allocateDeviceMemory() {
    m_texture.allocateDeviceMemory();
    m_dist.allocateDeviceMemory();
  }

  void CEnvironmentMap::copyToDevice(CEnvironmentMap* dst) {
    CEnvironmentMap envMap;
    envMap.m_texture = m_texture.copyToDevice();
    envMap.m_dist = m_dist.copyToDevice();
    CUDA_ASSERT(cudaMemcpy(dst, &envMap, sizeof(envMap), cudaMemcpyHostToDevice));
  }

  void CEnvironmentMap::freeDeviceMemory() {
    m_texture.freeDeviceMemory();
    m_dist.freeDeviceMemory();
  }
}