#include "intersect/ray.hpp"

namespace rt {
  Ray::Ray(const glm::vec3& origin, const glm::vec3& direction) :
    m_origin(origin), m_direction(direction) {

  }

  // Change basis of ray
  Ray Ray::transform(const glm::mat4& transformMatrix) const {
    glm::vec3 origin = glm::vec3(transformMatrix * glm::vec4(m_origin, 1.0f));
    glm::vec3 direction = glm::normalize(glm::vec3(transformMatrix * glm::vec4(m_direction, 0.0f)));
    Ray r(origin, direction);
    return r;
  }
}