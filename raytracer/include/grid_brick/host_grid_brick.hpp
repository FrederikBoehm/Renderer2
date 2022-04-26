#ifndef GRID_BRICK_HPP
#define GRID_BRICK_HPP
#include <memory>
#include <glm/glm.hpp>
#include <string>
#include <vector>
#include <memory>
#include <atomic>
#include "buf3d.hpp"
#include "filtering/openvdb_data.hpp"
#include <cuda_runtime.h>

// Copyright (c) Nikolai Hofmann
// https://git9.cs.fau.de/ul40ovyj/voldata/-/blob/dev/include/voldata/grid.h
// https://git9.cs.fau.de/ul40ovyj/voldata/-/blob/dev/include/voldata/grid_brick.h
namespace rt {
  class CDeviceBrickGrid;

  class CHostGrid {
  public:
    CHostGrid();
    CHostGrid(const glm::mat4& t);
    virtual ~CHostGrid() {}

    // grid interface
    virtual float lookup(const glm::uvec3& ipos) const = 0;                 // index-space grid lookup
    virtual std::pair<float, float> minorant_majorant() const = 0;          // global minorant and majorant
    virtual glm::uvec3 index_extent() const = 0;                            // max of index space voxel AABB, origin always (0, 0, 0)
    virtual size_t num_voxels() const = 0;                                  // number of (active) voxels in this grid
    virtual size_t size_bytes() const = 0;                                  // required bytes to store this grid

    // convenience operators and functions
    virtual std::string to_string(const std::string& indent = "") const;      // string representation
    inline float operator[](const glm::uvec3& ipos) const { return lookup(ipos); };

    // data
    glm::mat4 transform;                                                    // grid transformation (i.e. model matrix)
  };
  std::ostream& operator<<(std::ostream& out, const CHostGrid& grid);

  class CHostBrickGrid : public CHostGrid {
    struct SDeviceResource {
      cudaArray_t d_indirectionData;
      cudaTextureObject_t d_indirectionObj;
      cudaMipmappedArray_t d_rangeData;
      cudaTextureObject_t d_rangeObj;
      cudaArray_t d_atlasData;
      cudaTextureObject_t d_atlasObj;
    };
  public:
    CHostBrickGrid();
    CHostBrickGrid(const filter::SOpenvdbData& grid, const glm::uvec3& numVoxels);
    //CHostBrickGrid(const std::shared_ptr<CHostGrid>& grid);
    virtual ~CHostBrickGrid();

    float lookup(const glm::uvec3& ipos) const;
    std::pair<float, float> minorant_majorant() const;
    glm::uvec3 index_extent() const;
    size_t num_voxels() const;
    size_t size_bytes() const;
    virtual std::string to_string(const std::string& indent = "") const override;
    void allocateDeviceMemory();
    void copyToDevice(CDeviceBrickGrid* dst) const;
    void freeDeviceMemory() const;

    // data
    glm::uvec3 n_bricks;
    std::pair<float, float> min_maj;
    std::atomic<size_t> brick_counter;
    Buf3D<uint32_t> indirection;                    // 3x 10bits uint: (ptr_x, ptr_y, ptr_z, 2bit unused)
    Buf3D<uint32_t> range;                          // 2x float16: (minorant, majorant)
    Buf3D<uint8_t> atlas;                           // 512x uint8_t: 8x8x8 normalized brick data
    std::vector<Buf3D<uint32_t>> range_mipmaps;     // 3x float16 min/max mipmas of range data
    SDeviceResource* m_deviceResource;
  };
}

#endif // !GRID_BRICK_HPP

