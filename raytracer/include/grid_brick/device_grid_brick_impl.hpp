#include "device_grid_brick.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "common.hpp"
#include <cmath>
#include <glm/glm.hpp>

namespace rt {
  inline float CDeviceBrickGrid::lookupDensity(const glm::vec3& ipos, const glm::vec3& random) const {
    return lookupDensity(ipos + random - 0.5f);
  }

  inline float CDeviceBrickGrid::lookupDensity(const glm::vec3& ipos) const {
    return m_volDensityScale * lookupDensityBrick(ipos);
  }

  inline float CDeviceBrickGrid::lookupDensityBrick(const glm::vec3& ipos) const {
    const glm::ivec3 iipos = glm::ivec3(glm::floor(ipos));
    const glm::ivec3 brick = iipos >> 3;
    const glm::uvec3 ptr = decode_ptr(tex3D<unsigned int>(m_indirectionObj, (brick.x + 0.5f) / m_indirectionSize.x, (brick.y + 0.5f) / m_indirectionSize.y, (brick.z + 0.5f) / m_indirectionSize.z));
    const float2 range = tex3DLod<float2>(m_rangeObj, (brick.x + 0.5f) / m_rangeSize.x, (brick.y + 0.5f) / m_rangeSize.y, (brick.z + 0.5f) / m_rangeSize.z, 0); // Seems like CUDA doesn't support half2 lookups
    glm::vec3 atlasPos = glm::ivec3(ptr << 3u) + (iipos & 7);
    atlasPos = (atlasPos + 0.5f) / glm::vec3(m_atlasSize);
    const unsigned char value_unorm = tex3D<unsigned char>(m_atlasObj, atlasPos.x, atlasPos.y, atlasPos.z);
    return decode_voxel(value_unorm, glm::vec2(range.x, range.y));
  }

  inline float CDeviceBrickGrid::lookupMajorant(const glm::vec3& ipos, int mip) const {
    const glm::ivec3 brick = glm::ivec3(glm::floor(ipos)) >> 3;
    return m_volDensityScale * tex3DLod<float2>(m_rangeObj, (brick.x + 0.5f) / m_rangeSize.x, (brick.y + 0.5f) / m_rangeSize.y, (brick.z + 0.5f) / m_rangeSize.z, mip).y; // Seems like CUDA doesn't support half2 lookups
  }

}