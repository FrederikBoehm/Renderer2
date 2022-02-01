#ifndef OPENVDB_BACKEND_HPP
#define OPENVDB_BACKEND_HPP
#include <vector>
#include "intersect/aabb.hpp"
#include <nanovdb/util/GridHandle.h>
namespace filter {

  struct SOpenvdbBackendConfig {
    std::vector<rt::SAABB> boundingBoxes;
    glm::ivec3 numVoxels = glm::ivec3(100);
  };
  

  struct SOpenvdbData;
  class COpenvdbBackend {
  public:
    static COpenvdbBackend* instance();

    void init(const SOpenvdbBackendConfig& config);
    void setValues();
    nanovdb::GridHandle<nanovdb::HostBuffer> getNanoGridHandle();
    void writeToFile(const nanovdb::GridHandle<nanovdb::HostBuffer>& gridHandle, const char* directory, const char* fileName);
  private:
    static COpenvdbBackend* s_instance;

    SOpenvdbData* m_data;

  
    COpenvdbBackend();
    ~COpenvdbBackend();

    COpenvdbBackend(const COpenvdbBackend&) = delete; // Singleton
    COpenvdbBackend(COpenvdbBackend&&) = delete; // Singleton
    COpenvdbBackend& operator=(COpenvdbBackend&) = delete; // Singleton
    COpenvdbBackend& operator=(COpenvdbBackend&&) = delete; // Singleton

  };
}
#endif