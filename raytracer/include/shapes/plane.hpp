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

  //inline SHitInformation Plane::intersect(const Ray& ray) const {
  //  // TODO: Currently computation is done in world space -> maybe switch to object space

  //  SHitInformation si;
  //  si.hit = false;

  //  float denominator = dot(ray.m_direction, m_normal);

  //  if (denominator != 0.0f) { // We have one hit

  //    float t = glm::dot(m_worldPos - ray.m_origin, m_normal) / denominator;
  //    if (t > 0.0f) {

  //      glm::vec3 intersectionPos = ray.m_origin + t * ray.m_direction;
  //      float distance = glm::length(intersectionPos - m_worldPos);
  //      if (distance < m_radius) {
  //        si.hit = true;
  //        si.pos = ray.m_origin + t * ray.m_direction;
  //        si.normal = m_normal;
  //        si.t = t;
  //      }
  //    }
  //  }

  //  return si;
  //}

  
}

#endif // !PLANE_HPP

