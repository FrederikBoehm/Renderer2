#include "mesh/mesh.hpp"
#include "utility/debugging.hpp"
#include <backend/rt_backend.hpp>
#include "utility/functions.hpp"
#include "backend/build_optix_accel.hpp"
namespace rt {
  CMesh::CMesh(const std::string& path, size_t submeshId, const std::vector<glm::vec3>& vbo, const std::vector<glm::uvec3>& ibo, const std::vector<glm::vec3>& normals, const std::vector<glm::vec2>& tcs, const SAABB& aabb):
    m_pathLength(path.size()),
    m_path((char*)malloc(path.size())),
    m_submeshId(submeshId),
    m_deviceObject(false),
    m_traversableHandle(NULL),
    m_deviceGasBuffer(NULL),
    m_aabb(aabb),
    m_deviceResource(nullptr) {
    initBuffers(vbo, ibo, normals, tcs);
    memcpy(m_path, path.data(), path.size());
  }

  CMesh::CMesh() :
    m_pathLength(0),
    m_path(nullptr),
    m_submeshId(0),
    m_numVertices(0),
    m_vbo(nullptr),
    m_numIndices(0),
    m_ibo(nullptr),
    m_normals(nullptr),
    m_tcs(nullptr),
    m_deviceObject(false),
    m_traversableHandle(NULL),
    m_deviceGasBuffer(NULL),
    m_aabb(),
    m_deviceResource(nullptr) {

  }

  CMesh::CMesh(CMesh&& mesh):
    m_pathLength(std::move(mesh.m_pathLength)),
    m_path(std::exchange(mesh.m_path, nullptr)),
    m_submeshId(std::move(mesh.m_submeshId)),
    m_numVertices(std::move(mesh.m_numVertices)),
    m_vbo(std::exchange(mesh.m_vbo, nullptr)),
    m_numIndices(std::move(mesh.m_numIndices)),
    m_ibo(std::exchange(mesh.m_ibo, nullptr)),
    m_normals(std::exchange(mesh.m_normals, nullptr)),
    m_tcs(std::exchange(mesh.m_tcs, nullptr)),
    m_deviceObject(std::move(mesh.m_deviceObject)),
    m_traversableHandle(std::exchange(mesh.m_traversableHandle, NULL)),
    m_deviceGasBuffer(std::exchange(mesh.m_deviceGasBuffer, NULL)),
    m_aabb(std::move(mesh.m_aabb)),
    m_deviceResource(std::move(mesh.m_deviceResource)) {

  }

  CMesh::~CMesh() {
    if (!m_deviceObject) {
      free(m_path);
      m_path = nullptr;
      free(m_vbo);
      m_vbo = nullptr;
      free(m_ibo);
      m_ibo = nullptr;
      free(m_normals);
      m_normals = nullptr;
      free(m_tcs);
      m_tcs = nullptr;
    }
  }

  void CMesh::initBuffers(const std::vector<glm::vec3>& vbo, const std::vector<glm::uvec3>& ibo, const std::vector<glm::vec3>& normals, const std::vector<glm::vec2>& tcs) {
    m_numVertices = vbo.size();
    size_t vboBytes = sizeof(glm::vec3) * vbo.size();
    m_vbo = static_cast<glm::vec3*>(malloc(vboBytes));
    m_numIndices = ibo.size();
    size_t iboBytes = sizeof(glm::uvec3) * ibo.size();
    m_ibo = static_cast<glm::uvec3*>(malloc(iboBytes));
    size_t normalsBytes = sizeof(glm::vec3) * normals.size();
    m_normals = static_cast<glm::vec3*>(malloc(normalsBytes));
    if (tcs.size() > 0) {
      size_t tcsBytes = sizeof(glm::vec2) * tcs.size();
      m_tcs = static_cast<glm::vec2*>(malloc(tcsBytes));
      memcpy(m_tcs, tcs.data(), tcsBytes);
    }
    else {
      m_tcs = nullptr;
    }

    memcpy(m_vbo, vbo.data(), vboBytes);
    memcpy(m_ibo, ibo.data(), iboBytes);
    memcpy(m_normals, normals.data(), normalsBytes);
  }

  void CMesh::allocateDeviceMemory() {
    if (m_deviceResource) {
      freeDeviceMemory();
      delete m_deviceResource;
    }
    m_deviceResource = new SMeshDeviceResource();
    CUDA_ASSERT(cudaMalloc(&m_deviceResource->d_vbo, sizeof(glm::vec3) * m_numVertices));
    CUDA_ASSERT(cudaMalloc(&m_deviceResource->d_ibo, sizeof(glm::uvec3) * m_numIndices));
    CUDA_ASSERT(cudaMalloc(&m_deviceResource->d_normals, sizeof(glm::vec3) * m_numVertices));
    if (m_tcs) {
      CUDA_ASSERT(cudaMalloc(&m_deviceResource->d_tcs, sizeof(glm::vec2) * m_numVertices));
    }
    else {
      m_deviceResource->d_tcs = nullptr;
    }
  }

  CMesh CMesh::copyToDevice() {
    CUDA_ASSERT(cudaMemcpy(m_deviceResource->d_vbo, m_vbo, sizeof(glm::vec3) * m_numVertices, cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(m_deviceResource->d_ibo, m_ibo, sizeof(glm::uvec3) * m_numIndices, cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(m_deviceResource->d_normals, m_normals, sizeof(glm::vec3) * m_numVertices, cudaMemcpyHostToDevice));
    if (m_tcs) {
        CUDA_ASSERT(cudaMemcpy(m_deviceResource->d_tcs, m_tcs, sizeof(glm::vec2) * m_numVertices, cudaMemcpyHostToDevice));
    }

    CMesh deviceMesh;
    deviceMesh.m_numVertices = m_numVertices;
    deviceMesh.m_vbo = m_deviceResource->d_vbo;
    deviceMesh.m_numIndices = m_numIndices;
    deviceMesh.m_ibo = m_deviceResource->d_ibo;
    deviceMesh.m_normals = m_deviceResource->d_normals;
    deviceMesh.m_tcs = m_deviceResource->d_tcs;
    deviceMesh.m_aabb = m_aabb;
    deviceMesh.m_deviceObject = true;
    return deviceMesh;
  }

  void CMesh::freeDeviceMemory() {
    if (m_deviceResource) {
      CUDA_ASSERT(cudaFree(m_deviceResource->d_vbo));
      CUDA_ASSERT(cudaFree(m_deviceResource->d_ibo));
      CUDA_ASSERT(cudaFree(m_deviceResource->d_normals));
      CUDA_ASSERT(cudaFree(m_deviceResource->d_tcs));
    }
  }

  void CMesh::buildOptixAccel() {
    if (m_deviceResource) {
      CUDA_ASSERT(cudaMemcpy(reinterpret_cast<void*>(m_deviceResource->d_vbo), m_vbo, sizeof(glm::vec3) * m_numVertices, cudaMemcpyHostToDevice)); // TODO: Find way to copy vertices and indices only once
      CUDA_ASSERT(cudaMemcpy(reinterpret_cast<void*>(m_deviceResource->d_ibo), m_ibo, sizeof(glm::uvec3) * m_numIndices, cudaMemcpyHostToDevice));
    }
    else {
      fprintf(stderr, "[ERROR] CMesh::getOptixBuildInput: vertex buffer and index buffer not allocated on device.\n");
    }

    OptixBuildInput buildInput;
    buildInput = {};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    buildInput.triangleArray.vertexStrideInBytes = sizeof(glm::vec3);
    buildInput.triangleArray.numVertices = m_numVertices;
    buildInput.triangleArray.vertexBuffers = reinterpret_cast<CUdeviceptr*>(&m_deviceResource->d_vbo);
    buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    buildInput.triangleArray.numIndexTriplets = m_numIndices;
    buildInput.triangleArray.indexBuffer = reinterpret_cast<CUdeviceptr>(m_deviceResource->d_ibo);
    buildInput.triangleArray.indexStrideInBytes = sizeof(glm::uvec3);
    buildInput.triangleArray.numSbtRecords = 1;
    buildInput.triangleArray.sbtIndexOffsetBuffer = 0;
    buildInput.triangleArray.sbtIndexOffsetSizeInBytes = 0;
    buildInput.triangleArray.primitiveIndexOffset = 0;
    OptixGeometryFlags flags[] = { OPTIX_GEOMETRY_FLAG_NONE };
    buildInput.triangleArray.flags = reinterpret_cast<unsigned int*>(flags);
    rt::buildOptixAccel(buildInput, &m_traversableHandle, &m_deviceGasBuffer);

  }

  OptixProgramGroup CMesh::getOptixProgramGroup() const {
    return CRTBackend::instance()->programGroups().m_hitMesh;
  }

}