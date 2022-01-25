#ifndef AABB_HPP
#define AABB_HPP
#include <glm/glm.hpp>
#include "utility/debugging.hpp"
#include "nanovdb/NanoVDB.h"
namespace rt {
  struct SAABB {

    DH_CALLABLE SAABB transform(const glm::mat4x3& transformation) const {
      glm::vec3 newMin = transformation * glm::vec4(m_min, 1.f);
      glm::vec3 newMax = transformation * glm::vec4(m_max, 1.f);
      return { newMin, newMax };
    }

    DH_CALLABLE SAABB& operator=(const nanovdb::BBoxR& bbox) {
      m_min = glm::vec3(bbox.min()[0], bbox.min()[1], bbox.min()[2]);
      m_max = glm::vec3(bbox.max()[0], bbox.max()[1], bbox.max()[2]);
      return *this;
    }

    glm::vec3 m_min;
    glm::vec3 m_max;

  };

}
#endif // !AABB_HPP
