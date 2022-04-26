#ifndef BUF3D_HPP
#define BUF3D_HPP

#include <glm/glm.hpp>

// Copyright (c) Nikolai Hofmann
// https://git9.cs.fau.de/ul40ovyj/voldata/-/blob/dev/include/voldata/buf3d.h

namespace rt {
  template <typename T> class Buf3D {
  public:
    Buf3D(const glm::uvec3& stride = glm::uvec3(0)) : stride(stride), data(size_t(stride.x) * stride.y * stride.z) {}

    inline T& operator[](const glm::uvec3& at) { return data[to_idx(at)]; }
    inline const T& operator[](const glm::uvec3& at) const { return data[to_idx(at)]; }

    inline glm::uvec3 size() const { return stride; }

    inline void prune(size_t slices) {
      this->stride.z = slices;
      data.resize(stride.x * stride.y * stride.z);
    }

    inline void resize(const glm::uvec3& stride) {
      this->stride = stride;
      data.resize(size_t(stride.x) * stride.y * stride.z);
    }

    inline size_t to_idx(const glm::uvec3& coord) const {
      return coord.z * stride.x * stride.y + coord.y * stride.x + coord.x;
    }

    inline glm::uvec3 to_coord(size_t idx) const {
      return glm::uvec3(idx % stride.x, (idx / stride.x) % stride.y, idx / (stride.x * stride.y));
    }

    // data
    glm::uvec3 stride;
    std::vector<T> data;
  };
}
#endif