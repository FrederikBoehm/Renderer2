#include "medium/nvdb_medium.hpp"
#include <nanovdb/NanoVDB.h>
#include "utility/functions.hpp"
#include "intersect/ray.hpp"
#include "sampling/sampler.hpp"
#include "scene/interaction.hpp"
#include <glm/gtx/transform.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include "medium/sggx_phase_function.hpp"
#include "utility/debugging.hpp"
#include "backend/rt_backend.hpp"
#include <optix/optix_stubs.h>
#include "medium/phase_function_impl.hpp"
#include "medium/medium_impl.hpp"
#include "backend/build_optix_accel.hpp"
#include "filtering/filtered_data.hpp"

namespace rt {
  CNVDBMedium::CNVDBMedium(const std::string& path, const glm::vec3& sigma_a, const glm::vec3& sigma_s, float g):
    CMedium(EMediumType::NVDB_MEDIUM),
    m_pathLength(path.size()),
    m_path((char*)malloc(path.size())),
    m_isHostObject(true),
    m_handle(getHandle(path)),
    m_gridType(m_handle->gridType()),
    m_deviceAabb(NULL),
    m_sigma_a(sigma_a),
    m_sigma_s(sigma_s),
    m_phase(new CHenyeyGreensteinPhaseFunction(g)),
    m_sigma_t(glm::max(sigma_a.x + sigma_s.x, glm::max(sigma_a.y + sigma_s.y, sigma_a.z + sigma_s.z))),
    m_deviceGasBuffer(NULL),
    m_deviceResource(nullptr) {
    
    init(path);
  }

  CNVDBMedium::CNVDBMedium(const std::string& path, const glm::vec3& sigma_a, const glm::vec3& sigma_s, const SSGGXDistributionParameters& sggxDiffuse, const SSGGXDistributionParameters& sggxSpecular, const glm::vec3& worldPos, const glm::vec3& n, const glm::vec3& scaling) :
    CMedium(EMediumType::NVDB_MEDIUM),
    m_isHostObject(true),
    m_handle(getHandle(path)),
    m_gridType(m_handle->gridType()),
    m_deviceAabb(NULL),
    m_sigma_a(sigma_a),
    m_sigma_s(sigma_s),
    m_phase(new CSGGXPhaseFunction(sggxDiffuse, sggxSpecular)),
    m_sigma_t(glm::max(sigma_a.x + sigma_s.x, glm::max(sigma_a.y + sigma_s.y, sigma_a.z + sigma_s.z))),
    m_deviceResource(nullptr) {
    nanovdb::CoordBBox box;
    switch (m_gridType) {
    case nanovdb::GridType::Float:
      m_grid = m_handle->grid<float>();
      m_invMaxDensity = 1.f / getMaxValue(m_grid);
      box = m_grid->indexBBox();
      m_worldBB = m_grid->worldBBox();
      m_size = getMediumSize(m_grid->worldBBox(), m_grid->voxelSize());
      if (m_grid->activeVoxelCount() == 0) {
        m_ibbMin = glm::ivec3(0);
        m_ibbMax = glm::ivec3(0);
      }
      else {
        m_ibbMin = glm::ivec3(box.min().x(), box.min().y(), box.min().z());
        m_ibbMax = glm::ivec3(box.max().x(), box.max().y(), box.max().z());
      }
      m_indexToModel = getIndexToModelTransformation(m_grid->map(), m_ibbMin, m_size);
      m_modelToIndex = glm::inverse(glm::mat4(m_indexToModel));
      break;
    case nanovdb::GridType::Vec4d:
      m_vec4grid = m_handle->grid<nanovdb::Vec4d>();
      m_invMaxDensity = 1.f / getMaxValue(m_vec4grid);
      box = m_vec4grid->indexBBox();
      m_worldBB = m_vec4grid->worldBBox();
      m_size = getMediumSize(m_vec4grid->worldBBox(), m_vec4grid->voxelSize());
      if (m_grid->activeVoxelCount() == 0) {
        m_ibbMin = glm::ivec3(0);
        m_ibbMax = glm::ivec3(0);
      }
      else {
        m_ibbMin = glm::ivec3(box.min().x(), box.min().y(), box.min().z());
        m_ibbMax = glm::ivec3(box.max().x(), box.max().y(), box.max().z());
      }
      m_indexToModel = getIndexToModelTransformation(m_vec4grid->map(), m_ibbMin, m_size);
      m_modelToIndex = glm::inverse(glm::mat4(m_indexToModel));
      break;
    }
  }

  CNVDBMedium::CNVDBMedium(const std::string& path, const glm::vec3& sigma_a, const glm::vec3& sigma_s, float diffuseRoughness, float specularRoughness) :
    CMedium(EMediumType::NVDB_MEDIUM),
    m_pathLength(path.size()),
    m_path((char*)malloc(path.size())),
    m_isHostObject(true),
    m_handle(getHandle(path)),
    m_gridType(m_handle->gridType()),
    m_deviceAabb(NULL),
    m_sigma_a(sigma_a),
    m_sigma_s(sigma_s),
    m_phase(new CSGGXPhaseFunction(diffuseRoughness, specularRoughness)),
    m_sigma_t(glm::max(sigma_a.x + sigma_s.x, glm::max(sigma_a.y + sigma_s.y, sigma_a.z + sigma_s.z))),
    m_deviceGasBuffer(NULL),
    m_deviceResource(nullptr) {

    init(path);
  }

  CNVDBMedium::CNVDBMedium() :
    CMedium(EMediumType::NVDB_MEDIUM),
    m_pathLength(0),
    m_path(nullptr),
    m_isHostObject(true),
    m_handle(nullptr),
    m_worldBB(),
    m_deviceAabb(NULL),
    m_size(0),
    m_indexToModel(1.f),
    m_modelToIndex(1.f),
    m_sigma_a(0.f),
    m_sigma_s(0.f),
    m_phase(nullptr),
    m_ibbMin(0),
    m_ibbMax(0),
    m_sigma_t(0.f),
    m_invMaxDensity(0.f),
    m_deviceResource(nullptr) {

  }

  CNVDBMedium::CNVDBMedium(CNVDBMedium&& medium) :
    CMedium(std::move(medium.type())),
    m_pathLength(std::move(medium.m_pathLength)),
    m_path(std::exchange(medium.m_path, nullptr)),
    m_isHostObject(std::move(medium.m_isHostObject)),
    m_handle(std::exchange(medium.m_handle, nullptr)),
    m_gridType(std::move(medium.m_gridType)),
    m_worldBB(std::move(medium.m_worldBB)),
    m_deviceAabb(std::exchange(medium.m_deviceAabb, NULL)),
    m_size(std::move(medium.m_size)),
    m_indexToModel(std::move(medium.m_indexToModel)),
    m_modelToIndex(std::move(medium.m_modelToIndex)),
    m_sigma_a(std::move(medium.m_sigma_a)),
    m_sigma_s(std::move(medium.m_sigma_s)),
    m_phase(std::exchange(medium.m_phase, nullptr)),
    m_ibbMin(std::move(medium.m_ibbMin)),
    m_ibbMax(std::move(medium.m_ibbMax)),
    m_sigma_t(std::move(medium.m_sigma_t)),
    m_invMaxDensity(std::move(medium.m_invMaxDensity)),
    m_deviceResource(std::exchange(medium.m_deviceResource, nullptr)) {
    switch (m_gridType) {
    case nanovdb::GridType::Float:
      m_grid = std::exchange(medium.m_grid, nullptr);
      break;
    case nanovdb::GridType::Vec4d:
      m_vec4grid = std::exchange(medium.m_vec4grid, nullptr);
      break;
    }
  }

  CNVDBMedium::~CNVDBMedium() {
    if (m_isHostObject) {
      delete m_path;
      delete m_handle;
      delete m_phase;
    }
  }

  void CNVDBMedium::init(const std::string& path) {
    memcpy(m_path, path.data(), path.size());

    nanovdb::CoordBBox box;
    switch (m_gridType) {
    case nanovdb::GridType::Float:
      m_grid = m_handle->grid<float>();
      m_invMaxDensity = 1.f / getMaxValue(m_grid);
      box = m_grid->indexBBox();
      m_worldBB = m_grid->worldBBox();
      m_size = getMediumSize(m_grid->worldBBox(), m_grid->voxelSize());
      if (m_grid->activeVoxelCount() == 0) {
        m_ibbMin = glm::ivec3(0);
        m_ibbMax = glm::ivec3(0);
      }
      else {
        m_ibbMin = glm::ivec3(box.min().x(), box.min().y(), box.min().z());
        m_ibbMax = glm::ivec3(box.max().x(), box.max().y(), box.max().z());
      }
      m_indexToModel = getIndexToModelTransformation(m_grid->map(), m_ibbMin, m_size);
      m_modelToIndex = glm::inverse(glm::mat4(m_indexToModel));
      break;
    case nanovdb::GridType::Vec4d:
      m_vec4grid = m_handle->grid<nanovdb::Vec4d>();
      m_invMaxDensity = 1.f / getMaxValue(m_vec4grid);
      box = m_vec4grid->indexBBox();
      m_worldBB = m_vec4grid->worldBBox();
      m_size = getMediumSize(m_vec4grid->worldBBox(), m_vec4grid->voxelSize());
      if (m_grid->activeVoxelCount() == 0) {
        m_ibbMin = glm::ivec3(0);
        m_ibbMax = glm::ivec3(0);
      }
      else {
        m_ibbMin = glm::ivec3(box.min().x(), box.min().y(), box.min().z());
        m_ibbMax = glm::ivec3(box.max().x(), box.max().y(), box.max().z());
      }
      m_indexToModel = getIndexToModelTransformation(m_vec4grid->map(), m_ibbMin, m_size);
      m_modelToIndex = glm::inverse(glm::mat4(m_indexToModel));
      break;
    }
  }


  CNVDBMedium& CNVDBMedium::operator=(const CNVDBMedium&& medium) {
    return *this;
  }

  

  void CNVDBMedium::allocateDeviceMemory() {
    if (m_deviceResource) {
      freeDeviceMemory();
      delete m_deviceResource;
    }

    m_deviceResource = new DeviceResource();
    switch (m_phase->type()) {
    case EPhaseFunction::HENYEY_GREENSTEIN:
      CUDA_ASSERT(cudaMalloc(&m_deviceResource->d_phase, sizeof(CHenyeyGreensteinPhaseFunction)));
      break;
    case EPhaseFunction::SGGX:
      CUDA_ASSERT(cudaMalloc(&m_deviceResource->d_phase, sizeof(CSGGXPhaseFunction)));
      break;
    }
  }

  CNVDBMedium CNVDBMedium::copyToDevice() const {
    m_handle->deviceUpload();

    
    CNVDBMedium medium;
    medium.m_isHostObject = false;
    medium.m_handle = this->m_handle;
    medium.m_gridType = this->m_gridType;
    switch (m_gridType) {
    case nanovdb::GridType::Float:
      medium.m_grid = m_handle->deviceGrid<float>();
      break;
    case nanovdb::GridType::Vec4d:
      medium.m_vec4grid = m_handle->deviceGrid<nanovdb::Vec4d>();
      break;
    }
    if (!medium.m_grid) {
      fprintf(stderr, "GridHandle does not contain a valid device grid");
    }
    if (m_deviceResource) {

      medium.m_phase = m_deviceResource->d_phase;
      switch (m_phase->type()) {
      case EPhaseFunction::HENYEY_GREENSTEIN:
        CUDA_ASSERT(cudaMemcpy(m_deviceResource->d_phase, this->m_phase, sizeof(CHenyeyGreensteinPhaseFunction), cudaMemcpyHostToDevice));
        break;
      case EPhaseFunction::SGGX:
        CUDA_ASSERT(cudaMemcpy(m_deviceResource->d_phase, this->m_phase, sizeof(CSGGXPhaseFunction), cudaMemcpyHostToDevice));
        break;
      }
    }
    else {
      fprintf(stderr, "No device resource for CNVDBMedium");
    }
    medium.m_size = this->m_size;
    medium.m_indexToModel = this->m_indexToModel;
    medium.m_modelToIndex = this->m_modelToIndex;
    medium.m_sigma_a = this->m_sigma_a;
    medium.m_sigma_s = this->m_sigma_s;
    medium.m_sigma_t = this->m_sigma_t;
    medium.m_invMaxDensity = this->m_invMaxDensity;
    medium.m_deviceResource = nullptr;
    medium.m_ibbMin = m_ibbMin;
    medium.m_ibbMax = m_ibbMax;
    medium.m_worldBB = m_worldBB;
    
    return medium;
  }

  void CNVDBMedium::freeDeviceMemory() const {
    if (m_deviceResource) {
      CUDA_ASSERT(cudaFree(m_deviceResource->d_phase));
    }
    CUDA_ASSERT(cudaFree(reinterpret_cast<void*>(m_deviceAabb)));
    CUDA_ASSERT(cudaFree(reinterpret_cast<void*>(m_deviceGasBuffer)));
  }

  void CNVDBMedium::buildOptixAccel() {
    if (!m_deviceAabb) {
      CUDA_ASSERT(cudaMalloc(reinterpret_cast<void**>(&m_deviceAabb), sizeof(OptixAabb)));
      CUDA_ASSERT(cudaMemcpy(reinterpret_cast<void*>(m_deviceAabb), &m_worldBB, sizeof(OptixAabb), cudaMemcpyHostToDevice));
    }

    OptixBuildInput buildInput;
    buildInput = {};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    buildInput.customPrimitiveArray.aabbBuffers = &m_deviceAabb;
    OptixGeometryFlags flags[] = { OPTIX_GEOMETRY_FLAG_NONE };
    buildInput.customPrimitiveArray.flags = reinterpret_cast<unsigned int*>(flags);
    buildInput.customPrimitiveArray.numSbtRecords = 1;
    buildInput.customPrimitiveArray.numPrimitives = 1;
    buildInput.customPrimitiveArray.sbtIndexOffsetBuffer = 0;
    buildInput.customPrimitiveArray.sbtIndexOffsetSizeInBytes = 0;
    buildInput.customPrimitiveArray.primitiveIndexOffset = 0;
    rt::buildOptixAccel(buildInput, &m_traversableHandle, &m_deviceGasBuffer);
  }

  glm::ivec3 CNVDBMedium::getMediumSize(const nanovdb::BBox<nanovdb::Vec3R>& boundingBox, const nanovdb::Vec3R& voxelSize) {
    nanovdb::Vec3R size = boundingBox.dim() / voxelSize;
    return glm::ivec3(size[0], size[1], size[2]);
  }

  float CNVDBMedium::getMaxValue(const nanovdb::NanoGrid<float>* grid) {
    float min = 0.f;
    float max = 0.f;
    grid->tree().extrema(min, max);
    return max;
  }

  float CNVDBMedium::getMaxValue(const nanovdb::NanoGrid<nanovdb::Vec4d>* grid) {
    const nanovdb::CoordBBox& iBB = grid->indexBBox();
    const nanovdb::Coord& minValues = iBB.min();
    const nanovdb::Coord& maxValues = iBB.max();
    nanovdb::DefaultReadAccessor<nanovdb::Vec4d> readAccessor = grid->getAccessor();
    float maxDensity = 0.f;
    for (int32_t x = minValues.x(); x <= maxValues.x(); ++x) {
      for (int32_t y = minValues.y(); y <= maxValues.y(); ++y) {
        for (int32_t z = minValues.z(); z <= maxValues.z(); ++z) {
          nanovdb::Vec4d value = readAccessor.getValue(nanovdb::Coord(x, y, z));
          maxDensity = fmaxf(reinterpret_cast<filter::SFilteredDataCompact&>(value).density, maxDensity);
        }
      }
    }
    return maxDensity;
  }

  glm::mat4 CNVDBMedium::getIndexToModelTransformation(const nanovdb::Map& map, const glm::ivec3& ibbMin, const glm::ivec3& size) {
    glm::mat4 nanoIndexToWorld(map.mMatF[0], map.mMatF[3], map.mMatF[6], 0.f,
                               map.mMatF[1], map.mMatF[4], map.mMatF[7], 0.f,
                               map.mMatF[2], map.mMatF[5], map.mMatF[8], 0.f,
                               map.mVecF[0], map.mVecF[1], map.mVecF[2], 1.f); // [IdxMin, IdxMax] to world space
    glm::mat4 indexToNano((float)size[0], 0.f, 0.f, 0.f,
                          0.f, (float)size[1], 0.f, 0.f,
                          0.f, 0.f, (float)size[2], 0.f,
                          ibbMin.x, ibbMin.y, ibbMin.z, 1.f); // [0, 1] to [IdxMin, IdxMax] (Nanovdb index space
    return nanoIndexToWorld * indexToNano;
  }

  nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>* CNVDBMedium::getHandle(const std::string& path) {
    nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>* handle = nullptr;
    try {
      handle = new nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>(nanovdb::io::readGrid<nanovdb::CudaDeviceBuffer>(path));
    }
    catch (const std::exception& e) {
      fprintf(stderr, "Couldn't load nvdb file: %s", e.what());
      handle = nullptr;
    }
    return handle;
  }

  OptixProgramGroup CNVDBMedium::getOptixProgramGroup() const {
    return CRTBackend::instance()->programGroups().m_hitVolume;
  }

  std::string CNVDBMedium::path() const {
    return std::string(m_path, m_pathLength);
  }

  OptixTraversableHandle CNVDBMedium::getOptixHandle() const {
    return m_traversableHandle;
  }
}