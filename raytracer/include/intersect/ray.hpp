#ifndef RAY_HPP
#define RAY_HPP

#include "utility/qualifiers.hpp"

#include <glm/glm.hpp>
#include "utility/functions.hpp"
namespace rt {
  class CMediumInstance;

  class CRay {
  public:
    inline static constexpr float OFFSET = 1e-4f;
    inline static constexpr float DEFAULT_TMAX = 1e+12;
    DH_CALLABLE CRay(const glm::vec3& origin, const glm::vec3& direction, float t_max = DEFAULT_TMAX, const CMediumInstance* medium = nullptr);
    DH_CALLABLE CRay();
    glm::vec3 m_origin;
    glm::vec3 m_direction;
    mutable float m_t_max;
    const CMediumInstance* m_medium;
    DH_CALLABLE static glm::vec3 offsetRayOrigin(const glm::vec3& p, const glm::vec3& offsetDir);
    DH_CALLABLE CRay offsetRayOrigin(const glm::vec3& offsetDir);
    DH_CALLABLE CRay transform(const glm::mat4x3& worldToModel) const;
    // Transforms without scaling direction to length 1
    DH_CALLABLE CRay transform2(const glm::mat4x3& worldToModel) const;
    DH_CALLABLE CRay robustTransform(const glm::mat4x3& worldToModel, const glm::vec3& offsetDir) const;
    DH_CALLABLE static CRay spawnRay(const glm::vec3& start, const glm::vec3& end, const CMediumInstance* originMedium = nullptr);
  };

  inline CRay::CRay() :
    m_origin(0.f), m_direction(0.f), m_t_max(0.f), m_medium(nullptr) {

  }

  inline CRay::CRay(const glm::vec3& origin, const glm::vec3& direction, float t_max, const CMediumInstance* medium) :
    m_origin(origin), m_direction(direction), m_t_max(t_max), m_medium(medium) {

  }

  // Change basis of ray
  inline CRay CRay::transform(const glm::mat4x3& transformMatrix) const {
    glm::vec3 origin = transformMatrix * glm::vec4(m_origin, 1.0f);
    glm::vec3 endpoint = transformMatrix * (glm::vec4(m_origin, 1.0f) + m_t_max * glm::vec4(m_direction, 0.f));
    glm::vec3 direction = glm::normalize(transformMatrix * glm::vec4(m_direction, 0.0f));
    CRay r(origin, direction, glm::length(endpoint - origin), m_medium);
    return r;
  }

  inline CRay CRay::transform2(const glm::mat4x3& transformMatrix) const {
    glm::vec3 origin = transformMatrix * glm::vec4(m_origin, 1.0f);
    glm::vec3 endpoint = transformMatrix * (glm::vec4(m_origin, 1.0f) + m_t_max * glm::vec4(m_direction, 0.f));
    glm::vec3 direction = transformMatrix * glm::vec4(m_direction, 0.0f);
    CRay r(origin, direction, glm::length(endpoint - origin) / glm::length(direction), m_medium);
    return r;
  }

  inline CRay CRay::robustTransform(const glm::mat4x3& worldToModel, const glm::vec3& offsetDir) const {
    CRay r = transform(worldToModel);
    return r.offsetRayOrigin(offsetDir);
  }

  inline CRay CRay::spawnRay(const glm::vec3& start, const glm::vec3& end, const CMediumInstance* originMedium) {
    glm::vec3 dir = end - start;
    glm::vec3 offsetted = offsetRayOrigin(start, glm::normalize(end - start));
    glm::vec3 newDir = end - offsetted;
    float t = glm::length(newDir);
    dir /= t;
    return CRay(offsetted, dir, t, originMedium);
  }

  inline glm::vec3 CRay::offsetRayOrigin(const glm::vec3& p, const glm::vec3& offsetDir) {
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

  inline CRay CRay::offsetRayOrigin(const glm::vec3& offsetDir) {
    glm::vec3 end = m_origin + m_t_max * m_direction;
    this->m_origin = offsetRayOrigin(this->m_origin, offsetDir);
    this->m_t_max = glm::length(end - m_origin);
    return *this;
  }

  
}

#endif