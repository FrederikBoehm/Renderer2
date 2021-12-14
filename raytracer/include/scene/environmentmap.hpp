#ifndef ENVIRONMENTMAP_HPP
#define ENVIRONMENTMAP_HPP

#include "texture/texture.hpp"
#include "sampling/distribution_2d.hpp"

#include <string>
#include <glm/glm.hpp>
namespace rt {
  class CEnvironmentMap {
  public:
    H_CALLABLE CEnvironmentMap();
    H_CALLABLE CEnvironmentMap(const std::string& path);

    H_CALLABLE std::vector<float> radiancePerPixel(const CTexture& texture) const;
    DH_CALLABLE glm::vec3 sample(CSampler& sampler, glm::vec3* direction, float* pdf) const;
    DH_CALLABLE glm::vec3 le(const glm::vec3& direction, float* pdf) const;

    H_CALLABLE void allocateDeviceMemory();
    H_CALLABLE void copyToDevice(CEnvironmentMap* dst);
    H_CALLABLE void freeDeviceMemory();

    CTexture m_texture;
  private:
    CDistribution2D m_dist;
  };

  inline glm::vec3 CEnvironmentMap::sample(CSampler& sampler, glm::vec3* direction, float* pdf) const {
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

  inline glm::vec3 CEnvironmentMap::le(const glm::vec3& direction, float* pdf) const {
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
}

#endif