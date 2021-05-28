#ifndef RAY_HPP
#define RAY_HPP

#include "cuda_runtime.h"

#include <glm/glm.hpp>
namespace rt {
  class Ray {
  public:
    __device__ __host__ Ray(const glm::vec3& origin, const glm::vec3& direction);
    const glm::vec3 m_origin;
    const glm::vec3 m_direction;
    __device__ __host__ Ray transform(const glm::mat4& worldToModel) const;
  };

  inline Ray::Ray(const glm::vec3& origin, const glm::vec3& direction) :
    m_origin(origin), m_direction(direction) {

  }
  inline Ray Ray::transform(const glm::mat4& transformMatrix) const {
    glm::vec3 origin = glm::vec3(transformMatrix * glm::vec4(m_origin, 1.0f));
    glm::vec3 direction = glm::normalize(glm::vec3(transformMatrix * glm::vec4(m_direction, 0.0f)));
    Ray r(origin, direction);
    return r;
  }
}

#endif