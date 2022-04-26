#ifndef GRID_BRICK_COMMON_HPP
#define GRID_BRICK_COMMON_HPP
#include <cstdint>
#include <glm/glm.hpp>
#include "utility/qualifiers.hpp"

// Copyright (c) Nikolai Hofmann
// https://git9.cs.fau.de/ul40ovyj/voldata/-/blob/dev/src/grid_brick.cpp

namespace rt {
  // ----------------------------------------------
// constants

  static const uint32_t BRICK_SIZE = 8;
  static const uint32_t BITS_PER_AXIS = 10;
  static const uint32_t MAX_BRICKS = 1 << BITS_PER_AXIS;
  static const uint32_t VOXELS_PER_BRICK = BRICK_SIZE * BRICK_SIZE * BRICK_SIZE;
  static const uint32_t NUM_MIPMAPS = 3;

  // ----------------------------------------------
  // encoding helpers

  DH_CALLABLE inline uint32_t encode_range(float x, float y) {
    return uint32_t(glm::detail::toFloat16(x)) | (uint32_t(glm::detail::toFloat16(y)) << 16);
  }

  DH_CALLABLE inline glm::vec2 decode_range(uint32_t data) {
    return glm::vec2(glm::detail::toFloat32(data & 0xFFFFu), glm::detail::toFloat32(data >> 16));
  }

  DH_CALLABLE inline uint32_t encode_ptr(const glm::uvec3& ptr) {
#ifndef __CUDA_ARCH__
    assert(ptr.x < MAX_BRICKS && ptr.y < MAX_BRICKS && ptr.z < MAX_BRICKS);
#endif
    return (glm::clamp(ptr.x, 0u, MAX_BRICKS - 1) << (2 + 2 * BITS_PER_AXIS)) |
      (glm::clamp(ptr.y, 0u, MAX_BRICKS - 1) << (2 + 1 * BITS_PER_AXIS)) |
      (glm::clamp(ptr.z, 0u, MAX_BRICKS - 1) << (2 + 0 * BITS_PER_AXIS));
  }

  DH_CALLABLE inline glm::uvec3 decode_ptr(uint32_t data) {
    return glm::uvec3((data >> (2 + 2 * BITS_PER_AXIS)) & (MAX_BRICKS - 1),
      (data >> (2 + 1 * BITS_PER_AXIS)) & (MAX_BRICKS - 1),
      (data >> (2 + 0 * BITS_PER_AXIS)) & (MAX_BRICKS - 1));
  }

  DH_CALLABLE inline uint8_t encode_voxel(float value, const glm::vec2& range) {
    const float value_norm = glm::clamp((value - range.x) / (range.y - range.x), 0.f, 1.f);
    return uint8_t(glm::round(255 * value_norm));
  }

  DH_CALLABLE inline float decode_voxel(uint8_t data, const glm::vec2& range) {
    return range.x + data * (1.f / 255.f) * (range.y - range.x);
  }

  DH_CALLABLE inline glm::uvec3 div_round_up(const glm::uvec3& num, const glm::uvec3& denom) {
    return glm::ceil(glm::vec3(num) / glm::vec3(denom));
  }
}
#endif