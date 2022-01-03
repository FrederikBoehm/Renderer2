#include <glm/glm.hpp>
#include <iostream>

#include "shapes/sphere.hpp"
#include "intersect/ray.hpp"
#include "backend/rt_backend.hpp"
#include "utility/debugging.hpp"

namespace rt {
  Sphere::Sphere() :CShape(EShape::SPHERE), m_radius(1.0f) {

  }

  Sphere::Sphere(float radius) : CShape(EShape::SPHERE), m_radius(radius) {
  }

  Sphere::Sphere(const glm::vec3& worldPos, float radius, const glm::vec3& normal) :
    CShape(EShape::SPHERE, worldPos, normal),
    m_radius(radius) {

  }

  OptixAabb Sphere::getAABB() const {
    glm::vec3 min(m_worldPos.x - m_radius, m_worldPos.y - m_radius, m_worldPos.z - m_radius);
    glm::vec3 max(m_worldPos.x + m_radius, m_worldPos.y + m_radius, m_worldPos.z + m_radius);

    return OptixAabb{ min.x, min.y, min.z, max.x, max.y, max.z };
  }

  SBuildInputWrapper Sphere::getOptixBuildInput() {
    if (!m_deviceAabb) {
      OptixAabb aabb = getAABB();
      CUDA_ASSERT(cudaMalloc(reinterpret_cast<void**>(&m_deviceAabb), sizeof(OptixAabb)));
      CUDA_ASSERT(cudaMemcpy(reinterpret_cast<void*>(m_deviceAabb), &aabb, sizeof(OptixAabb), cudaMemcpyHostToDevice));
    }

    SBuildInputWrapper wrapper;
    wrapper.flags.push_back(OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT);

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

  OptixProgramGroup Sphere::getOptixProgramGroup() const {
    return CRTBackend::instance()->programGroups().m_hitSurface;
  }
}