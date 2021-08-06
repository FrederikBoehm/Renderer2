#ifndef RAY_HPP
#define RAY_HPP

#include "utility/qualifiers.hpp"

#include <glm/glm.hpp>
namespace rt {
  class Ray {
  public:
    DH_CALLABLE Ray(const glm::vec3& origin, const glm::vec3& direction);
    const glm::vec3 m_origin;
    const glm::vec3 m_direction;
    DH_CALLABLE Ray transform(const glm::mat4& worldToModel) const;
  };

  
}

#endif