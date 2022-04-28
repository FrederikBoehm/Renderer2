#ifndef DEVICE_GRID_BRICK_HPP
#define DEVICE_GRID_BRICK_HPP
#include "utility/qualifiers.hpp"
#include <glm/glm.hpp>
namespace rt {
  struct SDeviceBrickGridResources {
    cudaArray_t m_indirectionData;
    cudaTextureObject_t m_indirectionObj;
    cudaArray_t m_rangeData;
    cudaTextureObject_t m_rangeObj;
    cudaArray_t m_atlasData;
    cudaTextureObject_t m_atlasObj;
  };

  class CDeviceBrickGrid {
    friend class CHostBrickGrid;
  public:
    CDeviceBrickGrid() = default;

    D_CALLABLE float lookupDensity(const glm::vec3& ipos, const glm::vec3& random, size_t* numLookups) const;
    D_CALLABLE float lookupMajorant(const glm::vec3& ipos, int mip, size_t* numLookups) const;

  private:
    cudaArray_t m_indirectionData;
    cudaTextureObject_t m_indirectionObj;
    glm::uvec3 m_indirectionSize;
    cudaMipmappedArray_t m_rangeData;
    cudaTextureObject_t m_rangeObj;
    glm::uvec3 m_rangeSize;
    glm::uvec3 m_rangeSize1;
    cudaArray_t m_atlasData;
    cudaTextureObject_t m_atlasObj;
    glm::uvec3 m_atlasSize;

    const float m_volDensityScale = 1.f;

    D_CALLABLE float lookupDensity(const glm::vec3& ipos, size_t* numLookups) const;
    D_CALLABLE float lookupDensityBrick(const glm::vec3& ipos, size_t* numLookups) const;
  };
}

#endif // !DEVICE_GRID_BRICK_HPP
