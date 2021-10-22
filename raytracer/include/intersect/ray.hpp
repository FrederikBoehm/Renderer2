#ifndef RAY_HPP
#define RAY_HPP

#include "utility/qualifiers.hpp"

#include <glm/glm.hpp>
namespace rt {
  class CRay {
  public:
    DH_CALLABLE CRay(const glm::vec3& origin, const glm::vec3& direction, float t_max = 1e+12);
    glm::vec3 m_origin;
    glm::vec3 m_direction;
    mutable float m_t_max;
    DH_CALLABLE CRay transform(const glm::mat4& worldToModel) const;
    DH_CALLABLE static CRay spawnRay(const glm::vec3& start, const glm::vec3& end);
  };

  
}

#endif