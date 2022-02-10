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
    glm::ivec3 numVoxels = glm::ivec3(100);
  };


  struct SOpenvdbData;
  class COpenvdbBackend {
  public:
    static COpenvdbBackend* instance();

    void init(const SOpenvdbBackendConfig& config);
    void setValues(const std::vector<SFilteredData>& filteredData);
    nanovdb::GridHandle<nanovdb::HostBuffer> getNanoGridHandle() const;
    void writeToFile(const nanovdb::GridHandle<nanovdb::HostBuffer>& gridHandle, const char* directory, const char* fileName) const;

    const SFilterLaunchParams& launchParams() const;
  private:
    static COpenvdbBackend* s_instance;

    SOpenvdbData* m_data;
    SFilterLaunchParams m_launchParams;


    COpenvdbBackend();
    ~COpenvdbBackend();

    COpenvdbBackend(const COpenvdbBackend&) = delete; // Singleton
    COpenvdbBackend(COpenvdbBackend&&) = delete; // Singleton
    COpenvdbBackend& operator=(COpenvdbBackend&) = delete; // Singleton
    COpenvdbBackend& operator=(COpenvdbBackend&&) = delete; // Singleton

  };

  inline const SFilterLaunchParams& COpenvdbBackend::launchParams() const {
    return m_launchParams;
  }
}
#endif