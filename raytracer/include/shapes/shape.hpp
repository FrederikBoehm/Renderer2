#ifndef SHAPE_HPP
#define SHAPE_HPP

#include "utility/qualifiers.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <optix/optix_types.h>
#include "scene/types.hpp"

//#include "intersect/ray.hpp"

namespace rt {
  enum EShape {
    SPHERE,
    CIRCLE,
    CUBOID,
    RECTANGLE
  };

  class SurfaceInteraction;
  class Ray;

  class CShape {
    friend struct SSharedMemoryInitializer;
  public:
    DH_CALLABLE EShape shape() const;

    DH_CALLABLE virtual ~CShape();
    H_CALLABLE SBuildInputWrapper getOptixBuildInput();
    H_CALLABLE OptixProgramGroup getOptixProgramGroup() const;

  protected:
    DH_CALLABLE CShape(EShape shape, const glm::vec3& worldPos, const glm::vec3& normal);
    DH_CALLABLE CShape(EShape shape);

    const EShape m_shape;
    glm::vec3 m_worldPos;
    glm::vec3 m_normal;
    glm::mat4 m_modelToWorld;
    glm::mat4 m_worldToModel;
    CUdeviceptr m_deviceAabb;
    OptixProgramGroup m_optixProgramGroup;

  private:
    DH_CALLABLE glm::mat4 getRotation(const glm::vec3& normal);
  };

  inline CShape::CShape(EShape shape, const glm::vec3& worldPos, const glm::vec3& normal) :
    m_shape(shape),
    m_worldPos(worldPos),
    m_normal(normal),
    m_modelToWorld(glm::translate(glm::mat4(1.0f), worldPos) * getRotation(normal)),
    m_worldToModel(glm::inverse(m_modelToWorld)),
    m_deviceAabb(NULL) {

  }

  inline CShape::CShape(EShape shape) :
    m_shape(shape),
    m_worldPos(glm::vec3(0.0f)),
    m_normal(glm::vec3(0.0f, 1.0f, 0.0f)),
    m_modelToWorld(glm::mat4(1.0f)),
    m_worldToModel(glm::mat4(1.0f)),
    m_deviceAabb(NULL) {

  }

  inline CShape::~CShape() {
#ifndef __CUDA_ARCH__
    cudaFree((void*)m_deviceAabb);
#endif
  }

  inline glm::mat4 CShape::getRotation(const glm::vec3& normal) {
    float cos = glm::dot(normal, glm::vec3(0.0f, 1.0f, 0.0f));
    if (cos == 1.0f) {
      return glm::mat4(1.0f);
    }
    else if (cos == -1.0f) {
      return glm::mat4(glm::mat3(-1.0f));
    }
    else {
      float angle = glm::acos(glm::dot(normal, glm::vec3(0.0f, 1.0f, 0.0f)));
      glm::vec3 rotationAxis = glm::normalize(glm::cross(normal, glm::vec3(0.0f, 1.0f, 0.0f)));
      return glm::rotate(glm::mat4(1.0f), angle, rotationAxis);
    }

  }

  inline EShape CShape::shape() const {
    return m_shape;
  }
}
#endif // !SHAPE_HPP
