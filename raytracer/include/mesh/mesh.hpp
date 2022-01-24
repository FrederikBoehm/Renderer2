#ifndef MESH_HPP
#define MESH_HPP
#include <glm/glm.hpp>
#include <vector>
#include "utility/qualifiers.hpp"
#include <optix/optix_types.h>
#include "scene/types.hpp"
#include <string>
namespace rt {
  struct SMeshDeviceResource {
    glm::vec3* d_vbo;
    glm::uvec3* d_ibo;
    glm::vec3* d_normals;
    glm::vec2* d_tcs;
  };

  class CMesh {
  public:
    H_CALLABLE CMesh(const std::string& path, size_t submeshId, const std::vector<glm::vec3>& vbo, const std::vector<glm::uvec3>& ibo, const std::vector<glm::vec3>& normals, const std::vector<glm::vec2>& tcs);
    H_CALLABLE CMesh();
    H_CALLABLE CMesh(CMesh&& mesh);
    H_CALLABLE ~CMesh();
    H_CALLABLE void allocateDeviceMemory();
    H_CALLABLE CMesh copyToDevice();
    H_CALLABLE void freeDeviceMemory();
    H_CALLABLE OptixProgramGroup getOptixProgramGroup() const;
    H_CALLABLE void buildOptixAccel();
    H_CALLABLE OptixTraversableHandle getOptixHandle() const;

    DH_CALLABLE const glm::vec3* vbo() const;
    DH_CALLABLE const glm::uvec3* ibo() const;
    DH_CALLABLE const glm::vec3* normals() const;
    DH_CALLABLE const glm::vec2* tcs() const;
    H_CALLABLE std::string path() const;
    H_CALLABLE size_t submeshId() const;
  private:
    uint16_t m_pathLength;
    char* m_path;
    size_t m_submeshId;
    size_t m_numVertices;
    glm::vec3* m_vbo;
    size_t m_numIndices;
    glm::uvec3* m_ibo;
    glm::vec3* m_normals;
    glm::vec2* m_tcs;
    size_t m_numTcs;
    bool m_deviceObject;

    OptixTraversableHandle m_traversableHandle;
    CUdeviceptr m_deviceGasBuffer;

    SMeshDeviceResource* m_deviceResource;

    H_CALLABLE void initBuffers(const std::vector<glm::vec3>& vbo, const std::vector<glm::uvec3>& ibo, const std::vector<glm::vec3>& normals, const std::vector<glm::vec2>& tcs);
  };

  inline const glm::vec3* CMesh::vbo() const {
    return m_vbo;
  }

  inline const glm::uvec3* CMesh::ibo() const {
    return m_ibo;
  }

  inline const glm::vec3* CMesh::normals() const {
    return m_normals;
  }

  inline const glm::vec2* CMesh::tcs() const {
    return m_tcs;
  }

  inline OptixTraversableHandle CMesh::getOptixHandle() const {
    return m_traversableHandle;
  }

  inline std::string CMesh::path() const {
    return std::string(m_path, m_pathLength);
  }

  inline size_t CMesh::submeshId() const {
    return m_submeshId;
  }
}
#endif // !MESH_HPP
