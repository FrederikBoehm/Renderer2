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
}