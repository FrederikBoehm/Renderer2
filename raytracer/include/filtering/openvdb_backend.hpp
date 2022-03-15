#ifndef OPENVDB_BACKEND_HPP
#define OPENVDB_BACKEND_HPP
#include <vector>
#include "intersect/aabb.hpp"
#include <nanovdb/util/GridHandle.h>
#include "filtering/launch_params.hpp"
#include "filtered_data.hpp"
namespace filter {

  struct SOpenvdbBackendConfig {
    std::vector<rt::SAABB> modelSpaceBoundingBoxes;
    std::vector<rt::SAABB> worldSpaceBoundingBoxes;
    glm::mat4x3 worldToModel;
    float voxelSize;
  };


  struct SOpenvdbData;
  class COpenvdbBackend {
  public:
    static COpenvdbBackend* instance();

    void init(const SOpenvdbBackendConfig& config);
    void setValues(const std::vector<SFilteredDataCompact>& filteredData, const glm::ivec3& numVoxels);
    nanovdb::GridHandle<nanovdb::HostBuffer> getNanoGridHandle() const;
    void writeToFile(const nanovdb::GridHandle<nanovdb::HostBuffer>& gridHandle, const char* directory, const char* fileName) const;
    SFilterLaunchParams setupGrid(const glm::vec3& voxelSize);

    const glm::ivec3& numVoxelsMajorant() const;
  private:
    static COpenvdbBackend* s_instance;

    SOpenvdbData* m_data;
    glm::vec3 m_minModel;
    glm::vec3 m_maxModel;
    glm::vec3 m_minWorld;
    glm::vec3 m_maxWorld;
    glm::mat4x3 m_worldToModel;
    glm::ivec3 m_numVoxelsMajorant;


    COpenvdbBackend();
    ~COpenvdbBackend();

    COpenvdbBackend(const COpenvdbBackend&) = delete; // Singleton
    COpenvdbBackend(COpenvdbBackend&&) = delete; // Singleton
    COpenvdbBackend& operator=(COpenvdbBackend&) = delete; // Singleton
    COpenvdbBackend& operator=(COpenvdbBackend&&) = delete; // Singleton

  };

  inline const glm::ivec3& COpenvdbBackend::numVoxelsMajorant() const {
    return m_numVoxelsMajorant;
  }
}
#endif