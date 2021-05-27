#ifndef SHAPE_HPP
#define SHAPE_HPP

#include "cuda_runtime.h"

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

//#include "intersect/ray.hpp"

namespace rt {
  class SurfaceInteraction;
  class Ray;

  class Shape {
  public:
    __device__ __host__ virtual SurfaceInteraction intersect(const Ray& ray) const = 0;

  protected:
    //__device__ __host__ Shape(float* modelToWorld, float* worldToModel, float* worldPos);
    __device__ __host__ Shape(const glm::vec3& worldPos);
    __device__ __host__ Shape();
    __device__ __host__ virtual ~Shape();

    glm::vec3 m_worldPos;
    glm::mat4 m_modelToWorld;
    glm::mat4 m_worldToModel;

  };

  //inline Shape::Shape(float* modelToWorld, float* worldToModel, float* worldPos) :
  //  m_modelToWorld(glm::make_mat4(modelToWorld)), m_worldToModel(glm::make_mat4(worldToModel)), m_worldPos(glm::make_vec3(worldPos)) {

  //}

  inline Shape::Shape(const glm::vec3& worldPos) :
    m_worldPos(worldPos),
    m_modelToWorld(glm::translate(glm::mat4(1.0f), worldPos)),
    m_worldToModel(glm::inverse(m_modelToWorld)) {

  }

  inline Shape::Shape() :
    m_modelToWorld(glm::mat4(1.0f)), m_worldToModel(glm::mat4(1.0f)), m_worldPos(glm::vec3(0.0f)) {

  }

  inline Shape::~Shape() {
  }
}
#endif // !SHAPE_HPP
