#include "mesh/mesh.hpp"
#include "utility/debugging.hpp"
#include <backend/rt_backend.hpp>
namespace rt {
  CMesh::CMesh(const std::vector<glm::vec3>& vbo, const std::vector<glm::uvec3>& ibo, const std::vector<glm::vec3>& normals, const glm::vec3& bbMin, const glm::vec3& bbMax):
    m_deviceObject(false),
    m_deviceResource(nullptr) {
    CUDA_LOG_ERROR_STATE();
    m_numVertices = vbo.size();
    size_t vboBytes = sizeof(glm::vec3) * vbo.size();
    m_vbo = static_cast<glm::vec3*>(malloc(vboBytes));
    m_numIndices = ibo.size();
    size_t iboBytes = sizeof(glm::uvec3) * ibo.size();
    m_ibo = static_cast<glm::uvec3*>(malloc(iboBytes));
    size_t normalsBytes = sizeof(glm::vec3) * normals.size();
    m_normals = static_cast<glm::vec3*>(malloc(normalsBytes));

    uint64_t vboAdress = reinterpret_cast<uint64_t>(m_vbo);
    uint64_t iboAdress = reinterpret_cast<uint64_t>(m_ibo);
    size_t maxIndex = 0;
    for (auto index : ibo) {
      size_t triangleMax = glm::max(index.x, glm::max(index.y, index.z));
      maxIndex = glm::max(triangleMax, maxIndex);
    }

    glm::vec3 lastVertex = vbo.data()[maxIndex];

    memcpy(m_vbo, vbo.data(), vboBytes);
    memcpy(m_ibo, ibo.data(), iboBytes);
    memcpy(m_normals, normals.data(), normalsBytes);
    m_aabb = { bbMin.x, bbMin.y, bbMin.z, bbMax.x, bbMax.y, bbMax.z };

    //std::vector<glm::vec3> vbo2;
    //vbo2.push_back(glm::vec3(10.f, 0.f, 0.f));
    //vbo2.push_back(glm::vec3(0.f, 10.f, 0.f));
    //vbo2.push_back(glm::vec3(0.f, 0.f, 10.f));
    //std::vector<glm::uvec3> ibo2;
    //ibo2.push_back(glm::uvec3(0, 1, 2));
    //m_numVertices = vbo2.size();
    //size_t vboBytes = sizeof(glm::vec3) * vbo2.size();
    //m_vbo = static_cast<glm::vec3*>(malloc(vboBytes));
    //m_numIndices = ibo2.size();
    //size_t iboBytes = sizeof(glm::uvec3) * ibo2.size();
    //m_ibo = static_cast<glm::uvec3*>(malloc(iboBytes));
    //size_t normalsBytes = sizeof(glm::vec3) * normals.size();
    //m_normals = static_cast<glm::vec3*>(malloc(normalsBytes));

    //uint64_t vboAdress = reinterpret_cast<uint64_t>(m_vbo);
    //uint64_t iboAdress = reinterpret_cast<uint64_t>(m_ibo);
    //size_t maxIndex = 0;
    //for (auto index : ibo2) {
    //  size_t triangleMax = glm::max(index.x, glm::max(index.y, index.z));
    //  maxIndex = glm::max(triangleMax, maxIndex);
    //}

    //glm::vec3 lastVertex = vbo2.data()[maxIndex];

    //memcpy(m_vbo, vbo2.data(), vboBytes);
    //memcpy(m_ibo, ibo2.data(), iboBytes);
    //memcpy(m_normals, normals.data(), normalsBytes);
    //m_aabb = { bbMin.x, bbMin.y, bbMin.z, bbMax.x, bbMax.y, bbMax.z };
    CUDA_LOG_ERROR_STATE();
  }

  CMesh::CMesh():
    m_numVertices(0),
    m_vbo(nullptr),
    m_numIndices(0),
    m_ibo(nullptr),
    m_normals(nullptr),
    m_aabb{ 0.f, 0.f, 0.f, 0.f, 0.f, 0.f },
    m_deviceAabb(nullptr),
    m_deviceObject(false),
    m_deviceResource(nullptr) {

  }

  CMesh::CMesh(CMesh&& mesh):
    m_numVertices(std::move(mesh.m_numVertices)),
    m_vbo(std::exchange(mesh.m_vbo, nullptr)),
    m_numIndices(std::move(mesh.m_numIndices)),
    m_ibo(std::exchange(mesh.m_ibo, nullptr)),
    m_normals(std::exchange(mesh.m_normals, nullptr)),
    m_aabb(std::move(mesh.m_aabb)),
    m_deviceAabb(std::move(mesh.m_deviceAabb)),
    m_deviceObject(std::move(mesh.m_deviceObject)),
    m_deviceResource(std::move(mesh.m_deviceResource)) {

  }

  CMesh::~CMesh() {
    if (!m_deviceObject) {
      free(m_vbo);
      m_vbo = nullptr;
      free(m_ibo);
      m_ibo = nullptr;
      free(m_normals);
      m_normals = nullptr;
    }
  }

  void CMesh::allocateDeviceMemory() {
    if (m_deviceResource) {
      freeDeviceMemory();
      delete m_deviceResource;
    }
    m_deviceResource = new SMeshDeviceResource();
    cudaMalloc(&m_deviceResource->d_vbo, sizeof(glm::vec3) * m_numVertices + 128);
    cudaMalloc(&m_deviceResource->d_ibo, sizeof(glm::uvec3) * m_numIndices + 128);
    cudaMalloc(&m_deviceResource->d_normals, sizeof(glm::vec3) * m_numVertices);
    CUDA_LOG_ERROR_STATE();
  }

  CMesh CMesh::copyToDevice() {
    CUDA_ASSERT(cudaMemcpy(m_deviceResource->d_vbo, m_vbo, sizeof(glm::vec3) * m_numVertices, cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(m_deviceResource->d_ibo, m_ibo, sizeof(glm::uvec3) * m_numIndices, cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(m_deviceResource->d_normals, m_normals, sizeof(glm::vec3) * m_numVertices, cudaMemcpyHostToDevice));

    CMesh deviceMesh;
    deviceMesh.m_numVertices = m_numVertices;
    deviceMesh.m_vbo = m_deviceResource->d_vbo;
    deviceMesh.m_numIndices = m_numIndices;
    deviceMesh.m_ibo = m_deviceResource->d_ibo;
    deviceMesh.m_normals = m_deviceResource->d_normals;
    deviceMesh.m_aabb = m_aabb;
    deviceMesh.m_deviceObject = true;
    return deviceMesh;
  }

  void CMesh::freeDeviceMemory() {
    if (m_deviceResource) {
      cudaFree(m_deviceResource->d_vbo);
      cudaFree(m_deviceResource->d_ibo);
      cudaFree(m_deviceResource->d_normals);
    }
  }

  SBuildInputWrapper CMesh::getOptixBuildInput() {
    //if (!m_deviceAabb) {
      //CUDA_ASSERT(cudaMalloc(reinterpret_cast<void**>(&m_deviceAabb), sizeof(OptixAabb)));
      //CUDA_ASSERT(cudaMemcpy(reinterpret_cast<void*>(m_deviceAabb), &m_aabb, sizeof(OptixAabb), cudaMemcpyHostToDevice));
    //}
    if (m_deviceResource) {
      CUDA_ASSERT(cudaMemcpy(reinterpret_cast<void*>(m_deviceResource->d_vbo), m_vbo, sizeof(glm::vec3) * m_numVertices, cudaMemcpyHostToDevice)); // TODO: Find way to copy vertices and indices only once
      CUDA_ASSERT(cudaMemcpy(reinterpret_cast<void*>(m_deviceResource->d_ibo), m_ibo, sizeof(glm::uvec3) * m_numIndices, cudaMemcpyHostToDevice));
    }
    else {
      fprintf(stderr, "[ERROR] CMesh::getOptixBuildInput: vertex buffer and index buffer not allocated on device.\n");
    }
    CUDA_LOG_ERROR_STATE();
    m_deviceVertices = reinterpret_cast<CUdeviceptr>(m_deviceResource->d_vbo);
    m_deviceIndices = reinterpret_cast<CUdeviceptr>(m_deviceResource->d_ibo);
    SBuildInputWrapper wrapper;
    wrapper.flags.push_back(OPTIX_GEOMETRY_FLAG_NONE);

    wrapper.buildInput = {};
    wrapper.buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    wrapper.buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    wrapper.buildInput.triangleArray.vertexStrideInBytes = sizeof(glm::vec3);
    wrapper.buildInput.triangleArray.numVertices = m_numVertices;
    wrapper.buildInput.triangleArray.vertexBuffers = reinterpret_cast<CUdeviceptr*>(&m_deviceResource->d_vbo);
    wrapper.buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    wrapper.buildInput.triangleArray.numIndexTriplets = m_numIndices;
    wrapper.buildInput.triangleArray.indexBuffer = reinterpret_cast<CUdeviceptr>(m_deviceResource->d_ibo);
    //CUdeviceptr iboPtr = reinterpret_cast<CUdeviceptr>(m_deviceResource->d_ibo);
    wrapper.buildInput.triangleArray.indexStrideInBytes = sizeof(glm::uvec3);
    wrapper.buildInput.triangleArray.numSbtRecords = 1;
    wrapper.buildInput.triangleArray.sbtIndexOffsetBuffer = 0;
    wrapper.buildInput.triangleArray.sbtIndexOffsetSizeInBytes = 0;
    wrapper.buildInput.triangleArray.primitiveIndexOffset = 0;
    wrapper.buildInput.triangleArray.flags = wrapper.flags.data();
    return wrapper;

  }

  OptixProgramGroup CMesh::getOptixProgramGroup() const {
    return CRTBackend::instance()->programGroups().m_hitMesh;
  }
}