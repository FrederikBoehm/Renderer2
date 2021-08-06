#ifndef PLANE_HPP
#define PLANE_HPP

#include "shape.hpp"
#include "intersect/hit_information.hpp"
#include "intersect/ray.hpp"

namespace rt {
  class Plane : public CShape { // Our plane is actually a circle, this simplifies intersection
  public:
    DH_CALLABLE Plane();
    DH_CALLABLE Plane(float radius);
    DH_CALLABLE Plane(const glm::vec3& worldPos, float radius, const glm::vec3& normal);

    DH_CALLABLE SHitInformation intersect(const Ray& ray) const;
  
  private:
    float m_radius;
    glm::vec3 m_normal; // World space normal
  };

  
}

#endif // !PLANE_HPP

