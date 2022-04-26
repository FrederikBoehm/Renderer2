#include <grid_brick/serialization.hpp>

#include <fstream>

#include <cereal/types/utility.hpp>
#include <cereal/types/atomic.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/archives/portable_binary.hpp>

// Copyright (c) Nikolai Hofmann
// https://git9.cs.fau.de/ul40ovyj/voldata/-/blob/dev/src/serialization.cpp

// external serialization functions
namespace cereal {
  // glm types
  template <class Archive> void serialize(Archive& archive, glm::vec2& v) { archive(v.x, v.y); }
  template <class Archive> void serialize(Archive& archive, glm::ivec2& v) { archive(v.x, v.y); }
  template <class Archive> void serialize(Archive& archive, glm::uvec2& v) { archive(v.x, v.y); }
  template <class Archive> void serialize(Archive& archive, glm::vec3& v) { archive(v.x, v.y, v.z); }
  template <class Archive> void serialize(Archive& archive, glm::ivec3& v) { archive(v.x, v.y, v.z); }
  template <class Archive> void serialize(Archive& archive, glm::uvec3& v) { archive(v.x, v.y, v.z); }
  template <class Archive> void serialize(Archive& archive, glm::vec4& v) { archive(v.x, v.y, v.z, v.w); }
  template <class Archive> void serialize(Archive& archive, glm::ivec4& v) { archive(v.x, v.y, v.z, v.w); }
  template <class Archive> void serialize(Archive& archive, glm::uvec4& v) { archive(v.x, v.y, v.z, v.w); }
  template <class Archive> void serialize(Archive& archive, glm::mat3& m) { archive(m[0], m[1], m[2]); }
  template <class Archive> void serialize(Archive& archive, glm::mat4& m) { archive(m[0], m[1], m[2], m[3]); }
}

namespace rt {
  // buf3d
  template <class Archive, typename T> void serialize(Archive& archive, Buf3D<T>& buf) {
    archive(buf.stride, buf.data);
  }

  // brick grid
  template <class Archive> void serialize(Archive& archive, CHostBrickGrid& grid) {
    archive(grid.transform, grid.n_bricks, grid.min_maj, grid.brick_counter, grid.indirection, grid.range, grid.atlas, grid.range_mipmaps);
  }

  // general write func
  template <typename T> void write(const T& data, const fs::path& path) {
    std::ofstream file(path, std::ios::binary);
    cereal::PortableBinaryOutputArchive archive(file);
    archive(data);
    std::cout << path << " written." << std::endl;
  }

  void write_grid(const std::shared_ptr<CHostGrid>& grid, const fs::path& path) {
    if (CHostBrickGrid* brick = dynamic_cast<CHostBrickGrid*>(grid.get()))
      write<CHostBrickGrid>(*brick, path);
    else
      throw std::runtime_error("Unsupported grid type!");
  }

  //std::shared_ptr<CHostBrickGrid> load_brick_grid(const fs::path& path) {
  //  std::ifstream file(path, std::ios::binary);
  //  cereal::PortableBinaryInputArchive archive(file);
  //  std::shared_ptr<CHostBrickGrid> grid = std::make_shared<CHostBrickGrid>();
  //  archive(*grid.get());
  //  return grid;
  //}
  CHostBrickGrid* load_brick_grid(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    cereal::PortableBinaryInputArchive archive(file);
    //std::shared_ptr<CHostBrickGrid> grid = std::make_shared<CHostBrickGrid>();
    CHostBrickGrid* grid = new CHostBrickGrid();
    archive(*grid);
    return grid;
  }
}
