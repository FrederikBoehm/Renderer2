#define _USE_MATH_DEFINES
#include <math.h>
#include "scene/environmentmap.hpp"

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
      for (uint8_t j = 0; j < texture.channels(); ++j) {
        pixelRadiance += texture.data()[i + j];
      }
      uint16_t row = i / (texture.channels() * texture.width());
      float sinTheta = glm::sin(M_PI * float(row + 0.5f) / (float)texture.height());
      radiance.push_back(pixelRadiance * sinTheta);
    }
    return radiance;
  }

  glm::vec3 CEnvironmentMap::sample(CSampler& sampler, glm::vec3* direction, float* pdf) const {
    glm::vec2 texelCoord = m_dist.sampleContinuous(sampler, pdf);
    float phi = texelCoord.x * 2 * M_PI;
    float theta = texelCoord.y * M_PI;
    float sinTheta = glm::sin(theta);
    if (sinTheta == 0.0f) {
      *pdf = 0.f;
    }
    else {
      *pdf /= (sinTheta * 2 * M_PI * M_PI);
    }
    *direction = glm::vec3(glm::sin(phi) * sinTheta, glm::cos(theta), glm::cos(phi) * sinTheta);
    direction->z = (direction->z);
    return m_texture(texelCoord.x, texelCoord.y);
  }

  glm::vec3 CEnvironmentMap::le(const glm::vec3& direction, float* pdf) const {
    float theta = glm::acos(glm::clamp(direction.y, -1.f, 1.f));
    float p = glm::atan(direction.x, direction.z);
    float phi = p < 0 ? (p + 2 * M_PI) : p;

    float x = phi / (2 * M_PI);
    float y = theta / M_PI;
    glm::vec2 pos = glm::vec2(x, y);
    float sinTheta = glm::sin(theta);
    *pdf = sinTheta == 0 ? 0 : m_dist.pdf(pos) / (2 * M_PI * M_PI * sinTheta);
    return m_texture(x, y);
  }

  void CEnvironmentMap::allocateDeviceMemory() {
    m_texture.allocateDeviceMemory();
    m_dist.allocateDeviceMemory();
  }

  void CEnvironmentMap::copyToDevice(CEnvironmentMap* dst) {
    CEnvironmentMap envMap;
    envMap.m_texture = m_texture.copyToDevice();
    envMap.m_dist = m_dist.copyToDevice();
    cudaMemcpy(dst, &envMap, sizeof(envMap), cudaMemcpyHostToDevice);
  }

  void CEnvironmentMap::freeDeviceMemory() {
    m_texture.freeDeviceMemory();
    m_dist.freeDeviceMemory();
  }
}