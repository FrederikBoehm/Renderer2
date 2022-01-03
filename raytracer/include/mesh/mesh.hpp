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
    glm::vec2* d_tcs;
  };

  class CMesh {
  public:
    H_CALLABLE CMesh(const std::vector<glm::vec3>& vbo, const std::vector<glm::uvec3>& ibo, const std::vector<glm::vec3>& normals, const std::vector<glm::vec2>& tcs);
    H_CALLABLE CMesh(const std::vector<glm::vec3>& vbo, const std::vector<glm::uvec3>& ibo, const std::vector<glm::vec3>& normals, const std::vector<glm::vec2>& tcs, const glm::vec3& worldPos, const glm::vec3& normal, const glm::vec3& scaling);
    H_CALLABLE CMesh();
    H_CALLABLE CMesh(CMesh&& mesh);
    H_CALLABLE ~CMesh();
    H_CALLABLE void allocateDeviceMemory();
    H_CALLABLE CMesh copyToDevice();
    H_CALLABLE void freeDeviceMemory();
    H_CALLABLE SBuildInputWrapper getOptixBuildInput();
    H_CALLABLE OptixProgramGroup getOptixProgramGroup() const;

    DH_CALLABLE const glm::vec3* vbo() const;
    DH_CALLABLE const glm::uvec3* ibo() const;
    DH_CALLABLE const glm::vec3* normals() const;
    DH_CALLABLE const glm::vec2* tcs() const;
    DH_CALLABLE const glm::mat4& modelToWorld() const;
    DH_CALLABLE const glm::mat4& worldToModel() const;
  private:
    size_t m_numVertices;
    glm::vec3* m_vbo;
    size_t m_numIndices;
    glm::uvec3* m_ibo;
    glm::vec3* m_normals;
    glm::vec2* m_tcs;
    size_t m_numTcs;
    glm::mat4 m_modelToWorld;
    glm::mat4 m_worldToModel;
    bool m_deviceObject;

    SMeshDeviceResource* m_deviceResource;

    H_CALLABLE void initBuffers(const std::vector<glm::vec3>& vbo, const std::vector<glm::uvec3>& ibo, const std::vector<glm::vec3>& normals, const std::vector<glm::vec2>& tcs);
    H_CALLABLE static glm::mat4 getModelToWorldTransform(const glm::vec3& worldPos, const glm::vec3& normal, const glm::vec3& scaling);
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

  inline const glm::mat4& CMesh::modelToWorld() const {
    return m_modelToWorld;
  }

  inline const glm::mat4& CMesh::worldToModel() const {
    return m_worldToModel;
  }
}
#endif // !MESH_HPP
