#ifndef MESH_HPP
#define MESH_HPP
#include <glm/glm.hpp>
#include <vector>
#include "utility/qualifiers.hpp"
#include <optix/optix_types.h>
#include "scene/types.hpp"
namespace rt {
  struct SMeshDeviceResource {
    glm::vec3* d_vbo;
    glm::uvec3* d_ibo;
    glm::vec3* d_normals;
  };

  class CMesh {
  public:
    H_CALLABLE CMesh(const std::vector<glm::vec3>& vbo, const std::vector<glm::uvec3>& ibo, const std::vector<glm::vec3>& normals, const glm::vec3& bbMin, const glm::vec3& bbMax);
    H_CALLABLE CMesh();
    H_CALLABLE CMesh(CMesh&& mesh);
    H_CALLABLE ~CMesh();
    H_CALLABLE void allocateDeviceMemory();
    H_CALLABLE CMesh copyToDevice();
    H_CALLABLE void freeDeviceMemory();
    H_CALLABLE SBuildInputWrapper getOptixBuildInput();
    H_CALLABLE OptixProgramGroup getOptixProgramGroup() const;
  private:
    size_t m_numVertices;
    glm::vec3* m_vbo;
    size_t m_numIndices;
    glm::uvec3* m_ibo;
    glm::vec3* m_normals;
    OptixAabb m_aabb;
    OptixAabb* m_deviceAabb;
    bool m_deviceObject;
    CUdeviceptr m_deviceVertices;
    CUdeviceptr m_deviceIndices;

    SMeshDeviceResource* m_deviceResource;
  };
}
#endif // !MESH_HPP
