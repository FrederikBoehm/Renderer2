#ifndef SERIALIZATION_HPP
#define SERIALIZATION_HPP
#include "host_grid_brick.hpp"

#include <memory>
#include <filesystem>
namespace fs = std::filesystem;

// Copyright (c) Nikolai Hofmann
// https://git9.cs.fau.de/ul40ovyj/voldata/-/blob/dev/include/voldata/serialization.h

namespace rt {

  void write_grid(const std::shared_ptr<CHostGrid>& grid, const fs::path& path);
  //std::shared_ptr<CHostBrickGrid> load_brick_grid(const fs::path& path);
  CHostBrickGrid* load_brick_grid(const fs::path& path);

} 

#endif