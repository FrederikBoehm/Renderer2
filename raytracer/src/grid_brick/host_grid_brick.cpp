#include "grid_brick/host_grid_brick.hpp"
#include <sstream>
#include <glm/gtx/string_cast.hpp>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <execution>
#include "filtering/filtered_data.hpp"
#include "utility/debugging.hpp"
#include "grid_brick/device_grid_brick.hpp"
#include "grid_brick/common.hpp"

// Copyright (c) Nikolai Hofmann
// https://git9.cs.fau.de/ul40ovyj/voldata/-/blob/dev/src/grid.cpp
// https://git9.cs.fau.de/ul40ovyj/voldata/-/blob/dev/src/grid_brick.cpp

namespace rt {
  CHostGrid::CHostGrid() : transform(glm::mat4(1)) {}

  CHostGrid::CHostGrid(const glm::mat4& t) : transform(t) {

  }


  std::string CHostGrid::to_string(const std::string& indent) const {
    std::stringstream out;
    const glm::uvec3 ibb_max = index_extent();
    out << indent << "AABB (index-space): " << glm::to_string(glm::uvec3(0)) << " / " << glm::to_string(ibb_max) << std::endl;
    const auto[min, maj] = minorant_majorant();
    out << indent << "minorant: " << min << ", majorant: " << maj << std::endl;
    const size_t active = num_voxels(), dense = size_t(ibb_max.x)*ibb_max.y*ibb_max.z;
    out << indent << "active voxels: " << active / 1000 << "k / " << dense / 1000 << "k (" << uint32_t(std::round(100 * active / float(dense))) << "%)" << std::endl;
    out << indent << "transform: " << glm::to_string(transform) << std::endl;
    out << indent << "memory: " << (size_bytes() / 100000) / 10.f << " MB";
    return out.str();
  }

  std::ostream& operator<<(std::ostream& out, const CHostGrid& CHostGrid) { return out << CHostGrid.to_string(); }



  

  CHostBrickGrid::CHostBrickGrid() : CHostGrid(), n_bricks(0), min_maj({ 0, 0 }), brick_counter(0), m_deviceResource(nullptr) {}

  std::pair<float, float> getMinorantMajorant(const filter::SOpenvdbData& grid, const glm::uvec3& numVoxels) {
    float min = FLT_MAX;
    float max = -FLT_MAX;
    for (int x = 0; x < numVoxels.x; ++x) {
      for (int y = 0; y < numVoxels.y; ++y) {
        for (int z = 0; z < numVoxels.z; ++z) {
          const filter::SFilteredDataCompact& data = reinterpret_cast<const filter::SFilteredDataCompact&>(grid.accessor.getValue(openvdb::Coord(x, y, z)));
          min = std::min(data.density, min);
          max = std::max(data.density, max);
        }
      }
    }
    return { min, max };
  }

  CHostBrickGrid::CHostBrickGrid(const filter::SOpenvdbData& grid, const glm::uvec3& numVoxels) :
    //CHostGrid(grid),
    CHostGrid(glm::mat4(1.f)),
    n_bricks(div_round_up(div_round_up(numVoxels, glm::uvec3(BRICK_SIZE)), glm::uvec3(1u << NUM_MIPMAPS)) * 1u << NUM_MIPMAPS),
    min_maj(getMinorantMajorant(grid, numVoxels)),
    m_deviceResource(nullptr)
  {
    // allocate buffers
    if (glm::any(glm::greaterThanEqual(n_bricks, glm::uvec3(MAX_BRICKS))))
      throw std::runtime_error(std::string("exceeded max brick count of ") + std::to_string(MAX_BRICKS));
    indirection.resize(n_bricks);
    range.resize(n_bricks);
    atlas.resize(n_bricks * BRICK_SIZE);

    // construct brick grid
    brick_counter = 0;
    std::vector<int> slices(n_bricks.z);
    std::iota(slices.begin(), slices.end(), 0);
    std::for_each(std::execution::par_unseq, slices.begin(), slices.end(), [&](int bz) {
      Vec4DGrid::Accessor accessor = grid.grid->getAccessor();
      for (size_t by = 0; by < n_bricks.y; ++by) {
        for (size_t bx = 0; bx < n_bricks.x; ++bx) {
          // store empty brick
          const glm::uvec3 brick = glm::uvec3(bx, by, bz);
          indirection[brick] = 0;
          // compute local range over dilated brick
          float local_min = FLT_MAX, local_max = -FLT_MAX;
          for (int z = -1; z < int(BRICK_SIZE) + 1; ++z) {
            for (int y = -1; y < int(BRICK_SIZE) + 1; ++y) {
              for (int x = -1; x < int(BRICK_SIZE) + 1; ++x) {
                glm::uvec3 coord = glm::uvec3(glm::ivec3(brick * BRICK_SIZE) + glm::ivec3(x, y, z));
                const filter::SFilteredDataCompact& data = reinterpret_cast<const filter::SFilteredDataCompact&>(accessor.getValue(openvdb::Coord(coord.x, coord.y, coord.z)));
                local_min = std::min(local_min, data.density);
                local_max = std::max(local_max, data.density);
              }
            }
          }
          // store range but skip pointer and atlas for empty bricks
          range[brick] = encode_range(local_min, local_max);
          if (local_max == local_min) continue;
          // allocate memory for brick
          const size_t id = brick_counter.fetch_add(1, std::memory_order_relaxed);
          const glm::uvec3 ptr = indirection.to_coord(id);
          // store pointer (offset)
          indirection[brick] = encode_ptr(ptr);
          // store brick data
          const glm::vec2 local_range = decode_range(range[brick]);
          for (size_t z = 0; z < BRICK_SIZE; ++z) {
            for (size_t y = 0; y < BRICK_SIZE; ++y) {
              for (size_t x = 0; x < BRICK_SIZE; ++x) {
                glm::uvec3 coord = brick * BRICK_SIZE + glm::uvec3(x, y, z);
                const filter::SFilteredDataCompact& data = reinterpret_cast<const filter::SFilteredDataCompact&>(accessor.getValue(openvdb::Coord(coord.x, coord.y, coord.z)));
                atlas[ptr * BRICK_SIZE + glm::uvec3(x, y, z)] = encode_voxel(data.density, local_range);
              }
            }
          }
        }
      }
    });

    // prune atlas in z dimension
    atlas.prune(BRICK_SIZE * std::round(std::ceil(brick_counter / float(n_bricks.x * n_bricks.y))));

    // generate min/max mipmaps of range texture
    range_mipmaps.resize(NUM_MIPMAPS);
    for (uint32_t i = 0; i < NUM_MIPMAPS; ++i) {
      const glm::uvec3 mip_size = glm::uvec3(n_bricks / (1u << (i + 1u)));
      range_mipmaps[i].resize(mip_size);
      auto& source = i == 0 ? range : range_mipmaps[i - 1];
      slices.resize(mip_size.z);
      std::iota(slices.begin(), slices.end(), 0);
      std::for_each(std::execution::par_unseq, slices.begin(), slices.end(), [&](int bz) {
        for (size_t by = 0; by < mip_size.y; ++by) {
          for (size_t bx = 0; bx < mip_size.x; ++bx) {
            const glm::uvec3 brick = glm::uvec3(bx, by, bz);
            float range_min = FLT_MAX, range_max = -FLT_MAX;
            for (uint32_t z = 0; z < 2; ++z) {
              for (uint32_t y = 0; y < 2; ++y) {
                for (uint32_t x = 0; x < 2; ++x) {
                  const glm::uvec3 source_at = 2u * brick + glm::uvec3(x, y, z);
                  const glm::vec2 curr = decode_range(source[source_at]);
                  range_min = std::min(range_min, curr.x);
                  range_max = std::max(range_max, curr.y);
                }
              }
            }
            range_mipmaps[i][brick] = encode_range(range_min, range_max);
          }
        }
      });
    }
  }

  //CHostBrickGrid::CHostBrickGrid(const std::shared_ptr<CHostGrid>& CHostGrid) : CHostBrickGrid(*CHostGrid) {}

  CHostBrickGrid::~CHostBrickGrid() {}

  float CHostBrickGrid::lookup(const glm::uvec3& ipos) const {
    const glm::uvec3 brick = ipos >> 3u;
    const glm::uvec3 ptr = decode_ptr(indirection[brick]);
    const glm::vec2 minmax = decode_range(range[brick]);
    const glm::uvec3 voxel = (ptr << 3u) + glm::uvec3(ipos & 7u);
    return decode_voxel(atlas[voxel], minmax);
  }

  std::pair<float, float> CHostBrickGrid::minorant_majorant() const { return min_maj; }

  glm::uvec3 CHostBrickGrid::index_extent() const { return n_bricks * BRICK_SIZE; }

  size_t CHostBrickGrid::num_voxels() const { return brick_counter * VOXELS_PER_BRICK; }

  size_t CHostBrickGrid::size_bytes() const {
    const size_t dense_bricks = n_bricks.x * n_bricks.y * n_bricks.z;
    const size_t size_indirection = sizeof(uint32_t) * dense_bricks;
    const size_t size_range = sizeof(uint32_t) * dense_bricks;
    const size_t size_atlas = sizeof(uint8_t) * brick_counter * VOXELS_PER_BRICK;
    size_t size_mipmaps = 0;
    for (const auto& mip : range_mipmaps)
      size_mipmaps += sizeof(uint32_t) * mip.stride.x * mip.stride.y * mip.stride.z;
    return size_indirection + size_range + size_atlas + size_mipmaps;
  }

  std::string CHostBrickGrid::to_string(const std::string& indent) const {
    std::stringstream out;
    out << CHostGrid::to_string(indent) << std::endl;
    out << indent << "voxel dim: " << glm::to_string(index_extent()) << std::endl;
    out << indent << "brick dim: " << glm::to_string(n_bricks) << std::endl;
    const size_t bricks_allocd = brick_counter, bricks_capacity = atlas.size().x * atlas.size().y * atlas.size().z / VOXELS_PER_BRICK;
    out << indent << "bricks in atlas: " << bricks_allocd << " / " << bricks_capacity << " (" << uint32_t(std::round(100 * bricks_allocd / float(bricks_capacity))) << "%)" << std::endl;
    out << indent << "atlas dim: " << glm::to_string(atlas.size()) << std::endl;
    return out.str();
  }

  void CHostBrickGrid::allocateDeviceMemory() {
    if (m_deviceResource) {
      freeDeviceMemory();
      delete m_deviceResource;
    }
    m_deviceResource = new SDeviceResource;


    // Indirection texture
    cudaChannelFormatDesc channelDescIndirection = cudaCreateChannelDesc<uint1>();
    cudaExtent extentIndirection{ indirection.stride.x, indirection.stride.y, indirection.stride.z };
    CUDA_ASSERT(cudaMalloc3DArray(&m_deviceResource->d_indirectionData, &channelDescIndirection, extentIndirection));

    cudaResourceDesc indirectionResDesc = {};
    indirectionResDesc.resType = cudaResourceTypeArray;
    indirectionResDesc.res.array.array = m_deviceResource->d_indirectionData;
    cudaTextureDesc indirectionTexDesc = {};
    indirectionTexDesc.addressMode[0] = cudaAddressModeBorder;
    indirectionTexDesc.addressMode[1] = cudaAddressModeBorder;
    indirectionTexDesc.addressMode[2] = cudaAddressModeBorder;
    indirectionTexDesc.filterMode = cudaFilterModePoint;
    indirectionTexDesc.readMode = cudaReadModeElementType;
    indirectionTexDesc.normalizedCoords = 1;
    CUDA_ASSERT(cudaCreateTextureObject(&m_deviceResource->d_indirectionObj, &indirectionResDesc, &indirectionTexDesc, NULL));

    //Range texture
    cudaChannelFormatDesc rangeChannelDesc = cudaCreateChannelDescHalf2();
    cudaExtent rangeExtent{ range.stride.x, range.stride.y, range.stride.z };
    CUDA_ASSERT(cudaMallocMipmappedArray(&m_deviceResource->d_rangeData, &rangeChannelDesc, rangeExtent, NUM_MIPMAPS + 1));
    cudaResourceDesc rangeResDesc = {};
    rangeResDesc.resType = cudaResourceTypeMipmappedArray;
    rangeResDesc.res.mipmap.mipmap = m_deviceResource->d_rangeData;
    cudaTextureDesc rangeTexDesc = {};
    rangeTexDesc.addressMode[0] = cudaAddressModeBorder;
    rangeTexDesc.addressMode[1] = cudaAddressModeBorder;
    rangeTexDesc.addressMode[2] = cudaAddressModeBorder;
    rangeTexDesc.filterMode = cudaFilterModePoint;
    rangeTexDesc.readMode = cudaReadModeElementType;
    rangeTexDesc.normalizedCoords = 1;
    rangeTexDesc.mipmapFilterMode = cudaFilterModePoint;
    rangeTexDesc.minMipmapLevelClamp = 0;
    rangeTexDesc.maxMipmapLevelClamp = NUM_MIPMAPS;
    CUDA_ASSERT(cudaCreateTextureObject(&m_deviceResource->d_rangeObj, &rangeResDesc, &rangeTexDesc, NULL));

    // Atlas texture
    cudaChannelFormatDesc channelDescAtlas = cudaCreateChannelDesc<unsigned char>();
    cudaExtent extentAtlas{ atlas.stride.x, atlas.stride.y, atlas.stride.z };
    CUDA_ASSERT(cudaMalloc3DArray(&m_deviceResource->d_atlasData, &channelDescAtlas, extentAtlas));

    cudaResourceDesc atlasResDesc = {};
    atlasResDesc.resType = cudaResourceTypeArray;
    atlasResDesc.res.array.array = m_deviceResource->d_atlasData;
    cudaTextureDesc atlasTexDesc = {};
    atlasTexDesc.addressMode[0] = cudaAddressModeBorder;
    atlasTexDesc.addressMode[1] = cudaAddressModeBorder;
    atlasTexDesc.addressMode[2] = cudaAddressModeBorder;
    atlasTexDesc.filterMode = cudaFilterModePoint;
    atlasTexDesc.readMode = cudaReadModeElementType;
    atlasTexDesc.normalizedCoords = 1;
    CUDA_ASSERT(cudaCreateTextureObject(&m_deviceResource->d_atlasObj, &atlasResDesc, &atlasTexDesc, NULL));

  }

  void CHostBrickGrid::copyToDevice(CDeviceBrickGrid* dst) const {
    if (m_deviceResource) {
      cudaMemcpy3DParms indirectionParams = {};
      indirectionParams.srcPtr = make_cudaPitchedPtr((void*)indirection.data.data(), indirection.stride.x * sizeof(unsigned int), indirection.stride.x, indirection.stride.y);
      indirectionParams.dstArray = m_deviceResource->d_indirectionData;
      indirectionParams.extent = make_cudaExtent(indirection.stride.x, indirection.stride.y, indirection.stride.z);
      indirectionParams.kind = cudaMemcpyHostToDevice;
      CUDA_ASSERT(cudaMemcpy3D(&indirectionParams));


      cudaArray_t defaultLevel;
      CUDA_ASSERT(cudaGetMipmappedArrayLevel(&defaultLevel, m_deviceResource->d_rangeData, 0));
      cudaMemcpy3DParms rangeParms = {};
      rangeParms.srcPtr = make_cudaPitchedPtr((void*)range.data.data(), range.stride.x * sizeof(float), range.stride.x, range.stride.y);
      rangeParms.dstArray = defaultLevel;
      rangeParms.extent = make_cudaExtent(range.stride.x, range.stride.y, range.stride.z);
      rangeParms.kind = cudaMemcpyHostToDevice;
      CUDA_ASSERT(cudaMemcpy3D(&rangeParms));
      for (size_t i = 0; i < range_mipmaps.size(); ++i) {
        cudaArray_t mipmapArray;
        CUDA_ASSERT(cudaGetMipmappedArrayLevel(&mipmapArray, m_deviceResource->d_rangeData, i + 1));
        cudaMemcpy3DParms mipmapParms = {};
        const rt::Buf3D<uint32_t>& mipmap = range_mipmaps[i];
        mipmapParms.srcPtr = make_cudaPitchedPtr((void*)mipmap.data.data(), mipmap.stride.x * sizeof(float), mipmap.stride.x, mipmap.stride.y);
        mipmapParms.dstArray = mipmapArray;
        mipmapParms.extent = make_cudaExtent(mipmap.stride.x, mipmap.stride.y, mipmap.stride.z);
        rangeParms.kind = cudaMemcpyHostToDevice;
        CUDA_ASSERT(cudaMemcpy3D(&mipmapParms));
      }


      cudaMemcpy3DParms atlasParms = {};
      atlasParms.srcPtr = make_cudaPitchedPtr((void*)atlas.data.data(), atlas.stride.x * sizeof(unsigned char), atlas.stride.x, atlas.stride.y);
      atlasParms.dstArray = m_deviceResource->d_atlasData;
      atlasParms.extent = make_cudaExtent(atlas.stride.x, atlas.stride.y, atlas.stride.z);
      atlasParms.kind = cudaMemcpyHostToDevice;
      CUDA_ASSERT(cudaMemcpy3D(&atlasParms));

      CDeviceBrickGrid deviceGrid;
      deviceGrid.m_indirectionData = m_deviceResource->d_indirectionData;
      deviceGrid.m_indirectionObj = m_deviceResource->d_indirectionObj;
      deviceGrid.m_indirectionSize = indirection.stride;
      deviceGrid.m_rangeData = m_deviceResource->d_rangeData;
      deviceGrid.m_rangeObj = m_deviceResource->d_rangeObj;
      deviceGrid.m_rangeSize = range.stride;
      deviceGrid.m_rangeSize1 = range_mipmaps[0].stride;
      deviceGrid.m_atlasData = m_deviceResource->d_atlasData;
      deviceGrid.m_atlasObj = m_deviceResource->d_atlasObj;
      deviceGrid.m_atlasSize = atlas.stride;
      CUDA_ASSERT(cudaMemcpy(dst, &deviceGrid, sizeof(CDeviceBrickGrid), cudaMemcpyHostToDevice));
    }
  }

  void CHostBrickGrid::freeDeviceMemory() const {
    if (m_deviceResource) {
      CUDA_ASSERT(cudaDestroyTextureObject(m_deviceResource->d_atlasObj));
      CUDA_ASSERT(cudaFreeArray(m_deviceResource->d_atlasData));
      CUDA_ASSERT(cudaDestroyTextureObject(m_deviceResource->d_rangeObj));
      CUDA_ASSERT(cudaFreeMipmappedArray(m_deviceResource->d_rangeData));
      CUDA_ASSERT(cudaDestroyTextureObject(m_deviceResource->d_indirectionObj));
      CUDA_ASSERT(cudaFreeArray(m_deviceResource->d_indirectionData));
    }
  }
}