#include "shapes/cuboid.hpp"
#include "intersect/hit_information.hpp"
#include "utility/functions.hpp"
#include "utility/debugging.hpp"
#include "backend/rt_backend.hpp"

namespace rt {
  CCuboid::CCuboid(const glm::vec3& worldPos, const glm::vec3& dimensions, const glm::vec3& normal) :
    CShape(EShape::CUBOID, worldPos, normal),
    m_dimensions(dimensions),
    m_faces{
      CRectangle(worldPos + glm::vec3(dimensions.x * 0.5f, 0.f, 0.f), glm::vec2(dimensions.y, dimensions.z), glm::vec3(1.f, 0.f, 0.f)),
      CRectangle(worldPos + glm::vec3(-dimensions.x * 0.5f, 0.f, 0.f), glm::vec2(dimensions.y, dimensions.z), glm::vec3(-1.f, 0.f, 0.f)),
      CRectangle(worldPos + glm::vec3(0.f, dimensions.y * 0.5f, 0.f), glm::vec2(dimensions.x, dimensions.z), glm::vec3(0.f, 1.f, 0.f)),
      CRectangle(worldPos + glm::vec3(0.f, -dimensions.y * 0.5f, 0.f), glm::vec2(dimensions.x, dimensions.z), glm::vec3(0.f, -1.f, 0.f)),
      CRectangle(worldPos + glm::vec3(0.f, 0.f, dimensions.z * 0.5f), glm::vec2(dimensions.x, dimensions.y), glm::vec3(0.f, 0.f, 1.f)),
      CRectangle(worldPos + glm::vec3(0.f, 0.f, -dimensions.z * 0.5f), glm::vec2(dimensions.x, dimensions.y), glm::vec3(0.f, 0.f, -1.f)),
    } {
    }

  OptixAabb CCuboid::getAABB() const {
    return OptixAabb{ m_worldPos.x - m_dimensions.x * 0.5f,
                      m_worldPos.y - m_dimensions.y * 0.5f,
                      m_worldPos.z - m_dimensions.z * 0.5f,
                      m_worldPos.x + m_dimensions.x * 0.5f,
                      m_worldPos.y + m_dimensions.y * 0.5f,
                      m_worldPos.z + m_dimensions.z * 0.5f }; // TODO: account for rotation
  }

  SBuildInputWrapper CCuboid::getOptixBuildInput() {
    if (!m_deviceAabb) {
      OptixAabb aabb = getAABB();
      CUDA_ASSERT(cudaMalloc(reinterpret_cast<void**>(&m_deviceAabb), sizeof(OptixAabb)));
      CUDA_ASSERT(cudaMemcpy(reinterpret_cast<void*>(m_deviceAabb), &aabb, sizeof(OptixAabb), cudaMemcpyHostToDevice));
    }

    //uint32_t aabbInputFlags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
    SBuildInputWrapper wrapper;
    wrapper.flags.push_back(OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT);
    //OptixBuildInput buildInput = {};
    wrapper.buildInput = {};
    wrapper.buildInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    wrapper.buildInput.customPrimitiveArray.aabbBuffers = &m_deviceAabb;
    wrapper.buildInput.customPrimitiveArray.flags = wrapper.flags.data();
    wrapper.buildInput.customPrimitiveArray.numSbtRecords = 1;
    wrapper.buildInput.customPrimitiveArray.numPrimitives = 1;
    wrapper.buildInput.customPrimitiveArray.sbtIndexOffsetBuffer = 0;
    wrapper.buildInput.customPrimitiveArray.sbtIndexOffsetSizeInBytes = 0;
    wrapper.buildInput.customPrimitiveArray.primitiveIndexOffset = 0;

    return wrapper;
  }

  OptixProgramGroup CCuboid::getOptixProgramGroup() const {
    return CRTBackend::instance()->programGroups().m_hitSurface;
  }
}