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

namespace rt {
  CNVDBMedium::CNVDBMedium(const std::string& path, const glm::vec3& sigma_a, const glm::vec3& sigma_s, float g):
    CMedium(EMediumType::NVDB_MEDIUM),
    m_isHostObject(true),
    m_handle(getHandle(path)),
    m_grid(m_handle->grid<float>()),
    m_readAccessor(new nanovdb::DefaultReadAccessor<float>(m_grid->getAccessor())),
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

  CNVDBMedium::CNVDBMedium(const std::string& path, const glm::vec3& sigma_a, const glm::vec3& sigma_s, SSGGXDistributionParameters& sggxParameters) :
    CMedium(EMediumType::NVDB_MEDIUM),
    m_isHostObject(true),
    m_handle(getHandle(path)),
    m_grid(m_handle->grid<float>()),
    m_readAccessor(new nanovdb::DefaultReadAccessor<float>(m_grid->getAccessor())),
    m_size(getMediumSize(m_grid->worldBBox(), m_grid->voxelSize())),
    m_mediumToWorld(getMediumToWorldTransformation(m_grid->worldBBox())),
    m_worldToMedium(glm::inverse(m_mediumToWorld)),
    m_sigma_a(sigma_a),
    m_sigma_s(sigma_s),
    m_phase(new CSGGXPhaseFunction(sggxParameters)),
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
#ifndef __CUDA_ARCH__
    if (m_isHostObject) {
      freeDeviceMemory();
      if (m_readAccessor) {
        delete m_readAccessor;
      }
      if (m_handle) {
        delete m_handle;
      }
      if (m_phase) {
        delete m_phase;
      }
    }
#endif
  }

  CNVDBMedium& CNVDBMedium::operator=(const CNVDBMedium&& medium) {
    return *this;
  }

  float CNVDBMedium::density(const glm::vec3& p, const nanovdb::DefaultReadAccessor<float>& accessor) const {
    glm::vec3 pSamples(p.x * m_size.x - 0.5f, p.y * m_size.y - 0.5f, p.z * m_size.z - 0.5f);
    glm::ivec3 pi = glm::floor(pSamples);
    glm::vec3 d = pSamples - (glm::vec3)pi;
    int x = pi.x;
    int y = pi.y;
    int z = pi.z;

    float d00 = interpolate(d.x, D(pi, accessor), D(pi + glm::ivec3(1, 0, 0), accessor));
    float d10 = interpolate(d.x, D(pi + glm::ivec3(0, 1, 0), accessor), D(pi + glm::ivec3(1, 1, 0), accessor));
    float d01 = interpolate(d.x, D(pi + glm::ivec3(0, 0, 1), accessor), D(pi + glm::ivec3(1, 0, 1), accessor));
    float d11 = interpolate(d.x, D(pi + glm::ivec3(0, 1, 1), accessor), D(pi + glm::ivec3(1, 1, 1), accessor));

    float d0 = interpolate(d.y, d00, d10);
    float d1 = interpolate(d.y, d01, d11);

    return interpolate(d.z, d0, d1);
  }

  float CNVDBMedium::D(const glm::ivec3& p, const nanovdb::DefaultReadAccessor<float>& accessor) const {
    glm::vec3 pCopy = p;
    if (!insideExclusive(p, glm::ivec3(-m_size.x / 2.f, -m_size.y / 2.f, -m_size.z / 2.f), glm::ivec3(m_size.x / 2.f, m_size.y / 2.f, m_size.z / 2.f))) {
      return 0.f;
    }
    nanovdb::Coord coord(p.x, p.y, p.z);
    return accessor.getValue(coord);
  }

  glm::vec3 CNVDBMedium::sample(const CRay& rayWorld, CSampler& sampler, SInteraction* mi) const {
    const CRay ray = rayWorld.transform(m_worldToMedium);
    const CRay rayWorldCopy = rayWorld;
    nanovdb::DefaultReadAccessor<float> accessor(m_grid->getAccessor());

    float t = 0.f;
    while (true) {
      t -= glm::log(1.f - sampler.uniformSample01()) * m_invMaxDensity / m_sigma_t;
      if (t >= ray.m_t_max) {
        break;
      }
      float d = density(ray.m_origin + t * ray.m_direction, accessor);
      if (d * m_invMaxDensity > sampler.uniformSample01()) {
        ray.m_t_max = t;
        CRay rayWorldNew = ray.transform(m_mediumToWorld);
        SHitInformation hitInfo = { true, rayWorldNew.m_origin + rayWorldNew.m_t_max * rayWorldNew.m_direction, glm::vec3(0.f), rayWorldNew.m_t_max };
        *mi = { hitInfo, nullptr, nullptr, this };
        return m_sigma_s / m_sigma_t;
      }
    }
    return glm::vec3(1.f);
  }

  glm::vec3 CNVDBMedium::tr(const CRay& rayWorld, CSampler& sampler) const {
    const CRay rayWorldCopy = rayWorld;
    const CRay ray = rayWorld.transform(m_worldToMedium);
    nanovdb::DefaultReadAccessor<float> accessor(m_grid->getAccessor());
    float tr = 1.f;
    float t = 0.f;
    while (true) {
      t -= glm::log(1.f - sampler.uniformSample01()) * m_invMaxDensity / m_sigma_t;
      if (t >= ray.m_t_max) {
        break;
      }

      float d = density(ray.m_origin + t * ray.m_direction, accessor);
      tr *= 1.f - glm::max(0.f, d * m_invMaxDensity);
      
      if (tr < 1.f) {
        float p = 1.f - tr;
        if (sampler.uniformSample01() < p) {
          return glm::vec3(0.f);
        }
        else {
          tr /= 1.f - p;
        }
      }
    }
    return glm::vec3(tr);
  }

  void CNVDBMedium::allocateDeviceMemory() {
    if (m_deviceResource) {
      freeDeviceMemory();
      delete m_deviceResource;
    }

    m_deviceResource = new DeviceResource();
    cudaMalloc(&m_deviceResource->d_readAccessor, sizeof(nanovdb::DefaultReadAccessor<float>));
    switch (m_phase->type()) {
    case EPhaseFunction::HENYEY_GREENSTEIN:
      cudaMalloc(&m_deviceResource->d_phase, sizeof(CHenyeyGreensteinPhaseFunction));
      break;
    case EPhaseFunction::SGGX:
      cudaMalloc(&m_deviceResource->d_phase, sizeof(CSGGXPhaseFunction));
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
      cudaMemcpy(m_deviceResource->d_readAccessor, this->m_readAccessor, sizeof(nanovdb::DefaultReadAccessor<float>), cudaMemcpyHostToDevice);

      medium.m_phase = m_deviceResource->d_phase;
      switch (m_phase->type()) {
      case EPhaseFunction::HENYEY_GREENSTEIN:
        cudaMemcpy(m_deviceResource->d_phase, this->m_phase, sizeof(CHenyeyGreensteinPhaseFunction), cudaMemcpyHostToDevice);
        break;
      case EPhaseFunction::SGGX:
        cudaMemcpy(m_deviceResource->d_phase, this->m_phase, sizeof(CSGGXPhaseFunction), cudaMemcpyHostToDevice);
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
      cudaFree(m_deviceResource->d_readAccessor);
      cudaFree(m_deviceResource->d_phase);
    }
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
}