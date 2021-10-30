#include "intersect/ray.hpp"

namespace rt {
  CRay::CRay(const glm::vec3& origin, const glm::vec3& direction, float t_max) :
    m_origin(origin), m_direction(glm::normalize(direction)), m_t_max(t_max) {

  }

  // Change basis of ray
  CRay CRay::transform(const glm::mat4& transformMatrix) const {
    glm::vec3 origin = glm::vec3(transformMatrix * glm::vec4(m_origin, 1.0f));
    glm::vec3 endpoint = glm::vec3(transformMatrix * (glm::vec4(m_origin, 1.0f) + m_t_max * glm::vec4(m_direction, 0.f)));
    glm::vec3 direction = glm::normalize(glm::vec3(transformMatrix * glm::vec4(m_direction, 0.0f)));
    CRay r(origin, direction, glm::length(endpoint - origin));
    return r;
  }

  CRay CRay::spawnRay(const glm::vec3& start, const glm::vec3& end) {
    glm::vec3 dir = end - start;
    float t = glm::length(dir);
    dir /= t;
    return CRay(start + OFFSET * dir, dir, t);
  }
}