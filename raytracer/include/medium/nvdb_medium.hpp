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
#include "intersect/aabb.hpp"
#include "filtering/filtered_data.hpp"
namespace rt {
  class CRay;
  class CSampler;
  struct SInteraction;
  class CPhaseFunction;
  struct SSGGXDistributionParameters;

  class CNVDBMedium : public CMedium {
    struct DeviceResource {
      CPhaseFunction* d_phase = nullptr;
    };

  public:
    H_CALLABLE CNVDBMedium(const std::string& path, const glm::vec3& sigma_a, const glm::vec3& sigma_s, float g, const glm::vec3& worldPos, const glm::vec3& n, const glm::vec3& scaling);
    H_CALLABLE CNVDBMedium(const std::string& path, const glm::vec3& sigma_a, const glm::vec3& sigma_s, const SSGGXDistributionParameters& sggxDiffuse, const SSGGXDistributionParameters& sggxSpecular, const glm::vec3& worldPos, const glm::vec3& n, const glm::vec3& scaling);
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

    template <typename TReadAccessor>
    DH_CALLABLE float density(const glm::vec3& p, const TReadAccessor& accessor) const;
    template <typename TReadAccessor>
    DH_CALLABLE filter::SFilteredData filteredData(const glm::vec3& p, const TReadAccessor& accessor) const;
    DH_CALLABLE float D(const glm::ivec3& p, const nanovdb::DefaultReadAccessor<float>& accessor) const;
    DH_CALLABLE float D(const glm::ivec3& p, const nanovdb::DefaultReadAccessor<nanovdb::Vec4d>& accessor) const;
    DH_CALLABLE filter::SFilteredData getValue(const glm::ivec3& p, const nanovdb::DefaultReadAccessor<nanovdb::Vec4d>& accessor) const;
    DH_CALLABLE filter::SFilteredData getValue(const glm::ivec3& p, const nanovdb::DefaultReadAccessor<float>& accessor) const;
    D_CALLABLE glm::vec3 sample(const CRay& rayWorld, CSampler& sampler, SInteraction* mi) const;
    D_CALLABLE glm::vec3 tr(const CRay& ray, CSampler& sampler) const;
    D_CALLABLE glm::vec3 normal(const glm::vec3& p, CSampler& sampler) const;

    DH_CALLABLE const CPhaseFunction& phase() const;

    DH_CALLABLE const SAABB& worldBB() const;

    H_CALLABLE OptixProgramGroup getOptixProgramGroup() const;
    H_CALLABLE void buildOptixAccel();
    H_CALLABLE OptixTraversableHandle getOptixHandle() const;

    H_CALLABLE std::string path() const;

  private:
    uint16_t m_pathLength;
    char* m_path;
    bool m_isHostObject;
    nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>* m_handle;
    union {
      const nanovdb::NanoGrid<float>* m_grid;
      const nanovdb::NanoGrid<nanovdb::Vec4d>* m_vec4grid;
    };
    nanovdb::GridType m_gridType;
    SAABB m_worldBB;
    CUdeviceptr m_deviceAabb;
    glm::ivec3 m_size;
    glm::vec3 m_sigma_a;
    glm::vec3 m_sigma_s;
    glm::mat4x3 m_indexToModel;
    glm::mat4x3 m_modelToIndex;
    CPhaseFunction* m_phase;
    glm::ivec3 m_ibbMin;
    glm::ivec3 m_ibbMax;


    float m_sigma_t;
    float m_invMaxDensity;

    OptixTraversableHandle m_traversableHandle;
    CUdeviceptr m_deviceGasBuffer;

    DeviceResource* m_deviceResource;

    H_CALLABLE static glm::ivec3 getMediumSize(const nanovdb::BBox<nanovdb::Vec3R>& boundingBox, const nanovdb::Vec3R& voxelSize);
    H_CALLABLE static float getMaxValue(const nanovdb::NanoGrid<float>* grid);
    H_CALLABLE static float getMaxValue(const nanovdb::NanoGrid<nanovdb::Vec4d>* grid);
    H_CALLABLE static glm::mat4 getIndexToModelTransformation(const nanovdb::Map& map, const glm::ivec3& ibbMin, const glm::ivec3& size);
    H_CALLABLE static nanovdb::GridHandle<nanovdb::CudaDeviceBuffer>* getHandle(const std::string& path);
    
    template <typename TReadAccessor>
    D_CALLABLE glm::vec3 normal(const glm::vec3& p, const TReadAccessor& accessor) const;

    template <typename TReadAccessor>
    D_CALLABLE glm::vec3 sampleInternal(const CRay& rayWorld, CSampler& sampler, SInteraction* mi, const TReadAccessor& accessor) const;

    template <typename TReadAccessor>
    D_CALLABLE glm::vec3 trInternal(const CRay& ray, CSampler& sampler, const TReadAccessor& accessor) const;
  };

  

  
}
#endif