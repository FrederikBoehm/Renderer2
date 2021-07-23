#include "shapes/plane.hpp"

namespace rt {
  Plane::Plane() :
    CShape(EShape::PLANE), m_radius(1.0f), m_normal(glm::vec3(0.0f, 1.0f, 0.0f)) {

  }

  Plane::Plane(float radius) :
    CShape(EShape::PLANE), m_radius(radius), m_normal(glm::vec3(0.0f, 1.0f, 0.0f)) {

  }

  Plane::Plane(const glm::vec3& worldPos, float radius, const glm::vec3& normal) :
    CShape(EShape::PLANE, worldPos),
    m_radius(radius),
    m_normal(normal) {

  }


  SHitInformation Plane::intersect(const Ray& ray) const {
    SHitInformation si;
    si.hit = false;

    float denominator = dot(ray.m_direction, m_normal);

    if (denominator != 0.0f) { // We have one hit

      float t = glm::dot(m_worldPos - ray.m_origin, m_normal) / denominator;
      if (t > 0.0f) {

        glm::vec3 intersectionPos = ray.m_origin + t * ray.m_direction;
        float distance = glm::length(intersectionPos - m_worldPos);
        if (distance < m_radius) {
          si.hit = true;
          si.pos = ray.m_origin + t * ray.m_direction;
          si.normal = m_normal;
          si.t = t;
        }
      }
    }

    return si;
  }
}