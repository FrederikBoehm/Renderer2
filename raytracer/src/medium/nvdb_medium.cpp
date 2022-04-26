#include "medium/nvdb_medium.hpp"
#include "grid_brick/host_grid_brick.hpp"
#include "grid_brick/serialization.hpp"
#include <filesystem>

namespace rt {
  CHostBrickGrid* CNVDBMedium::loadBrickGrid(const std::string& path) {
    return load_brick_grid(std::filesystem::path(path + ".brickgrid"));
  }

  void CNVDBMedium::brickGridAllocateDeviceMemory() {
    m_hostBrickGrid->allocateDeviceMemory();
  }

  void CNVDBMedium::brickGridCopyToDevice(CDeviceBrickGrid* dst) const {
    m_hostBrickGrid->copyToDevice(dst);
  }

  void CNVDBMedium::brickGridFreeDeviceMemory() const {
    m_hostBrickGrid->freeDeviceMemory();
  }
}