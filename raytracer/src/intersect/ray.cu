#include "intersect/ray.hpp"

#include "utility/functions.hpp"

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

  CRay CRay::robustTransform(const glm::mat4& worldToModel, const glm::vec3& offsetDir) const {
    CRay r = transform(worldToModel);
    return r.offsetRayOrigin(offsetDir);
  }

  CRay CRay::spawnRay(const glm::vec3& start, const glm::vec3& end) {
    glm::vec3 dir = end - start;
    glm::vec3 offsetted = offsetRayOrigin(start, glm::normalize(end - start));
    glm::vec3 newDir = end - offsetted;
    float t = glm::length(newDir);
    dir /= t;
    return CRay(offsetted, dir, t);
  }

  glm::vec3 CRay::offsetRayOrigin(const glm::vec3& p, const glm::vec3& offsetDir) {
    float d = glm::dot(glm::abs(offsetDir), glm::vec3(CRay::OFFSET));
    glm::vec3 offset = d * offsetDir;
    glm::vec3 po = p + offset;
    for (uint8_t i = 0; i < 3; ++i) {
      if (offset[i] > 0) {
        po[i] = nextFloatUp(po[i]);
      }
      else if (offset[i] < 0) {
        po[i] = nextFloatDown(po[i]);
      }
    }
    return po;
  }

  CRay CRay::offsetRayOrigin(const glm::vec3& offsetDir) {
    glm::vec3 end = m_origin + m_t_max * m_direction;
    this->m_origin = offsetRayOrigin(this->m_origin, offsetDir);
    this->m_t_max = glm::length(end - m_origin);
    return *this;
  }
}