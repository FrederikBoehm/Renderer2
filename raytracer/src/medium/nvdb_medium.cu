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

namespace rt {
  CNVDBMedium::CNVDBMedium(const std::string& path, const glm::vec3& sigma_a, const glm::vec3& sigma_s, float g):
    CMedium(EMediumType::NVDB_MEDIUM),
    m_isHostObject(true),
    m_handle(getHandle(path)),
    m_grid(m_handle->grid<float>()),
    m_readAccessor(new nanovdb::DefaultReadAccessor<float>(m_grid->getAccessor())),
    m_deviceAabb(NULL),
    m_size(getMediumSize(m_grid->worldBBox(), m_grid->voxelSize())),
    m_mediumToWorld(getMediumToWorldTransformation(m_grid->worldBBox())),
    m_worldToMedium(glm::inverse(m_mediumToWorld)),
    m_sigma_a(sigma_a),
    m_sigma_s(sigma_s),
    m_phase(new CHenyeyGreensteinPhaseFunction(g)),
    m_sigma_t(sigma_a.z + sigma_s.z),
    m_invMaxDensity(1.f / getMaxValue(m_grid)),
    m_deviceResource(nullptr) {
  }

  CNVDBMedium::CNVDBMedium(const std::string& path, const glm::vec3& sigma_a, const glm::vec3& sigma_s, const SSGGXDistributionParameters& sggxDiffuse, const SSGGXDistributionParameters& sggxSpecular) :
    CMedium(EMediumType::NVDB_MEDIUM),
    m_isHostObject(true),
    m_handle(getHandle(path)),
    m_grid(m_handle->grid<float>()),
    m_readAccessor(new nanovdb::DefaultReadAccessor<float>(m_grid->getAccessor())),
    m_deviceAabb(NULL),
    m_size(getMediumSize(m_grid->worldBBox(), m_grid->voxelSize())),
    m_mediumToWorld(getMediumToWorldTransformation(m_grid->worldBBox())),
    m_worldToMedium(glm::inverse(m_mediumToWorld)),
    m_sigma_a(sigma_a),
    m_sigma_s(sigma_s),
    m_phase(new CSGGXPhaseFunction(sggxDiffuse, sggxSpecular)),
    m_sigma_t(sigma_a.z + sigma_s.z),
    m_invMaxDensity(1.f / getMaxValue(m_grid)),
    m_deviceResource(nullptr) {

  }

  CNVDBMedium::CNVDBMedium(const std::string& path, const glm::vec3& sigma_a, const glm::vec3& sigma_s, float diffuseRoughness, float specularRoughness) :
    CMedium(EMediumType::NVDB_MEDIUM),
    m_isHostObject(true),
    m_handle(getHandle(path)),
    m_grid(m_handle->grid<float>()),
    m_readAccessor(new nanovdb::DefaultReadAccessor<float>(m_grid->getAccessor())),
    m_deviceAabb(NULL),
    m_size(getMediumSize(m_grid->worldBBox(), m_grid->voxelSize())),
    m_mediumToWorld(getMediumToWorldTransformation(m_grid->worldBBox())),
    m_worldToMedium(glm::inverse(m_mediumToWorld)),
    m_sigma_a(sigma_a),
    m_sigma_s(sigma_s),
    m_phase(new CSGGXPhaseFunction(diffuseRoughness, specularRoughness)),
    m_sigma_t(sigma_a.z + sigma_s.z),
    m_invMaxDensity(1.f / getMaxValue(m_grid)),
    m_deviceResource(nullptr) {

  }

  CNVDBMedium::CNVDBMedium() :
    CMedium(EMediumType::NVDB_MEDIUM),
    m_isHostObject(true),
    m_handle(nullptr),
    m_grid(nullptr),
    m_readAccessor(nullptr),
    m_deviceAabb(NULL),
    m_size(0),
    m_mediumToWorld(1.f),
    m_worldToMedium(1.f),
    m_sigma_a(0.f),
    m_sigma_s(0.f),
    m_phase(nullptr),
    m_sigma_t(0.f),
    m_invMaxDensity(0.f),
    m_deviceResource(nullptr) {

  }

  CNVDBMedium::CNVDBMedium(CNVDBMedium&& medium) :
    CMedium(std::move(medium.type())),
    m_isHostObject(std::move(medium.m_isHostObject)),
    m_handle(std::exchange(medium.m_handle, nullptr)),
    m_grid(std::exchange(medium.m_grid, nullptr)),
    m_readAccessor(std::exchange(medium.m_readAccessor, nullptr)),
    m_deviceAabb(std::exchange(medium.m_deviceAabb, NULL)),
    m_size(std::move(medium.m_size)),
    m_mediumToWorld(std::move(medium.m_mediumToWorld)),
    m_worldToMedium(std::move(medium.m_worldToMedium)),
    m_sigma_a(std::move(medium.m_sigma_a)),
    m_sigma_s(std::move(medium.m_sigma_s)),
    m_phase(std::exchange(medium.m_phase, nullptr)),
    m_sigma_t(std::move(medium.m_sigma_t)),
    m_invMaxDensity(std::move(medium.m_invMaxDensity)),
    m_deviceResource(std::exchange(medium.m_deviceResource, nullptr)) {
  }

  CNVDBMedium::~CNVDBMedium() {
    if (m_isHostObject) {
      delete m_readAccessor;
      delete m_handle;
      delete m_phase;
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
    CUDA_ASSERT(cudaMalloc(&m_deviceResource->d_readAccessor, sizeof(nanovdb::DefaultReadAccessor<float>)));
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
    medium.m_grid = m_handle->deviceGrid<float>();
    if (!medium.m_grid) {
      fprintf(stderr, "GridHandle does not contain a valid device grid");
    }
    if (m_deviceResource) {
      medium.m_readAccessor = m_deviceResource->d_readAccessor;
      CUDA_ASSERT(cudaMemcpy(m_deviceResource->d_readAccessor, this->m_readAccessor, sizeof(nanovdb::DefaultReadAccessor<float>), cudaMemcpyHostToDevice));

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
      medium.m_readAccessor = nullptr;
      fprintf(stderr, "No device resource for CNVDBMedium");
    }
    medium.m_size = this->m_size;
    medium.m_mediumToWorld = this->m_mediumToWorld;
    medium.m_worldToMedium = this->m_worldToMedium;
    medium.m_sigma_a = this->m_sigma_a;
    medium.m_sigma_s = this->m_sigma_s;
    medium.m_sigma_t = this->m_sigma_t;
    medium.m_invMaxDensity = this->m_invMaxDensity;
    medium.m_deviceResource = nullptr;
    
    return medium;
  }

  void CNVDBMedium::freeDeviceMemory() const {
    if (m_deviceResource) {
      CUDA_ASSERT(cudaFree(m_deviceResource->d_readAccessor));
      CUDA_ASSERT(cudaFree(m_deviceResource->d_phase));
    }
    CUDA_ASSERT(cudaFree(reinterpret_cast<void*>(m_deviceAabb)));
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

  glm::mat4 CNVDBMedium::getMediumToWorld(const nanovdb::Map& map) {
    glm::mat4 mediumToWorld(map.mMatF[0], map.mMatF[3], map.mMatF[6], 0.f,
                            map.mMatF[1], map.mMatF[4], map.mMatF[7], 0.f,
                            map.mMatF[2], map.mMatF[5], map.mMatF[8], 0.f,
                            map.mVecF[0], map.mVecF[1], map.mVecF[2], 1.f);
    return mediumToWorld;
  }

  glm::mat4 CNVDBMedium::getMediumToWorldTransformation(const nanovdb::BBox<nanovdb::Vec3R>& boundingBox) {
    nanovdb::Vec3R size = boundingBox.dim();
    nanovdb::Vec3R pos = (boundingBox.max() + boundingBox.min()) / 2.f;
    glm::mat4 scaling = glm::scale(glm::vec3(size[0], size[1], size[2]));
    glm::mat4 translation1 = glm::translate(glm::vec3(-0.5f)); // Move corner to origin
    glm::mat4 translation2 = glm::translate(glm::vec3(pos[0], pos[1], pos[2]));
    return translation2 * scaling;
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

  SBuildInputWrapper CNVDBMedium::getOptixBuildInput() {
    if (!m_deviceAabb) {
      const nanovdb::BBoxR& bbox = m_grid->worldBBox();
      nanovdb::Vec3R boundsMin = bbox.min();
      nanovdb::Vec3R boundsMax = bbox.max() + m_grid->voxelSize();
      OptixAabb aabb{ boundsMin[0], boundsMin[1], boundsMin[2], boundsMax[0], boundsMax[1], boundsMax[2] }; // TODO: Make bounding box rotatable
      CUDA_ASSERT(cudaMalloc(reinterpret_cast<void**>(&m_deviceAabb), sizeof(OptixAabb)));
      CUDA_ASSERT(cudaMemcpy(reinterpret_cast<void*>(m_deviceAabb), &aabb, sizeof(OptixAabb), cudaMemcpyHostToDevice));
    }

    SBuildInputWrapper wrapper;
    wrapper.flags.push_back(OPTIX_GEOMETRY_FLAG_NONE);

    wrapper.buildInput = {};
    wrapper.buildInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    wrapper.buildInput.customPrimitiveArray.aabbBuffers = &m_deviceAabb;
    wrapper.buildInput.customPrimitiveArray.flags = wrapper.flags.data();
    wrapper.buildInput.customPrimitiveArray.numSbtRecords = 1;
    wrapper.buildInput.customPrimitiveArray.numPrimitives = 1;
    wrapper.buildInput.customPrimitiveArray.sbtIndexOffsetBuffer = 0;
    wrapper.buildInput.customPrimitiveArray.sbtIndexOffsetSizeInBytes = 0;
    wrapper.buildInput.customPrimitiveArray.primitiveIndexOffset = 0;

    return wrapper;
  }

  OptixProgramGroup CNVDBMedium::getOptixProgramGroup() const {
    return CRTBackend::instance()->programGroups().m_hitVolume;
  }
}