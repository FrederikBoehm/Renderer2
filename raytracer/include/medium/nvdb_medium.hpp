#ifndef NVDB_MEDIUM_HPP
#define NVDB_MEDIUM_HPP
#include "utility/qualifiers.hpp"
#include <string>

#include <nanovdb/util/IO.h>
#include <nanovdb/util/CudaDeviceBuffer.h>
#include <glm/glm.hpp>
#include "medium.hpp"
#include <optix/optix_types.h>

#include "scene/types.hpp"
namespace rt {
  class CRay;
  class CSampler;
  struct SInteraction;
  class CPhaseFunction;
  struct SSGGXDistributionParameters;

  class CNVDBMedium : public CMedium {
    struct DeviceResource {
      nanovdb::DefaultReadAccessor<float>* d_readAccessor = nullptr;
      CPhaseFunction* d_phase = nullptr;
    };

  public:
    H_CALLABLE CNVDBMedium(const std::string& path, const glm::vec3& sigma_a, const glm::vec3& sigma_s, float g);
    H_CALLABLE CNVDBMedium(const std::string& path, const glm::vec3& sigma_a, const glm::vec3& sigma_s, const SSGGXDistributionParameters& sggxDiffuse, const SSGGXDistributionParameters& sggxSpecular);
    H_CALLABLE CNVDBMedium(const std::string& path, const glm::vec3& sigma_a, const glm::vec3& sigma_s, float diffuseRoughness, float specularRoughness);
    H_CALLABLE CNVDBMedium();
    H_CALLABLE CNVDBMedium(const CNVDBMedium& medium) = delete;
    H_CALLABLE CNVDBMedium(CNVDBMedium&& medium);

    H_CALLABLE CNVDBMedium& operator=(const CNVDBMedium&& medium);
    H_CALLABLE CNVDBMedium& operator=(const CNVDBMedium& medium) = delete;

    H_CALLABLE ~CNVDBMedium();

    H_CALLABLE void allocateDeviceMemory();
    H_CALLABLE CNVDBMedium copyToDevice() const;
    H_CALLABLE void freeDeviceMemory() const;

    DH_CALLABLE float density(const glm::vec3& p, const nanovdb::DefaultReadAccessor<float>& accessor) const;
    DH_CALLABLE float D(const glm::ivec3& p, const nanovdb::DefaultReadAccessor<float>& accessor) const;
    D_CALLABLE glm::vec3 sample(const CRay& rayWorld, CSampler& sampler, SInteraction* mi) const;
    D_CALLABLE glm::vec3 tr(const CRay& ray, CSampler& sampler) const;
    D_CALLABLE glm::vec3 normal(const glm::vec3& p, CSampler& sampler) const;
    D_CALLABLE glm::vec3 normal(const glm::vec3& p, const nanovdb::DefaultReadAccessor<float>& accessor) const;

    DH_CALLABLE const CPhaseFunction& phase() const;

    DH_CALLABLE const nanovdb::NanoGrid<float>* grid() const;

    H_CALLABLE SBuildInputWrapper getOptixBuildInput();
    H_CALLABLE OptixProgramGroup getOptixProgramGroup() const;

  private:
    bool m_isHostObject;
    nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>* m_handle;
    const nanovdb::NanoGrid<float>* m_grid;
    const nanovdb::DefaultReadAccessor<float>* m_readAccessor;
    CUdeviceptr m_deviceAabb;
    glm::ivec3 m_size;
    glm::vec3 m_sigma_a;
    glm::vec3 m_sigma_s;
    glm::mat4 m_mediumToWorld;
    glm::mat4 m_worldToMedium;
    CPhaseFunction* m_phase;


    float m_sigma_t;
    float m_invMaxDensity;

    DeviceResource* m_deviceResource;

    H_CALLABLE static glm::ivec3 getMediumSize(const nanovdb::BBox<nanovdb::Vec3R>& boundingBox, const nanovdb::Vec3R& voxelSize);
    H_CALLABLE static float getMaxValue(const nanovdb::NanoGrid<float>* grid);
    H_CALLABLE static glm::mat4 getMediumToWorld(const nanovdb::Map& map);
    H_CALLABLE static glm::mat4 getMediumToWorldTransformation(const nanovdb::BBox<nanovdb::Vec3R>& boundingBox);
    H_CALLABLE static nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>* getHandle(const std::string& path);
    
  };

  

  
}
#endif