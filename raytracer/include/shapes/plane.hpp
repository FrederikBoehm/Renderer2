#include "shape.hpp"
#include "intersect/surface_interaction.hpp"
#include "intersect/ray.hpp"

namespace rt {
  class Plane : public Shape { // Our plane is actually a circle, this simplifies intersection
  public:
    Plane(float radius);
    Plane(const glm::vec3& worldPos, float radius, const glm::vec3& normal);

    __device__ __host__ virtual SurfaceInteraction intersect(const Ray& ray) const override;
  
  private:
    float m_radius;
    glm::vec3 m_normal; // World space normal
  };

  Plane::Plane(float radius):
    Shape(), m_radius(radius), m_normal(glm::vec3(0.0f, 1.0f, 0.0f)) {

  }

  Plane::Plane(const glm::vec3& worldPos, float radius, const glm::vec3& normal):
    Shape(worldPos),
    m_radius(radius),
    m_normal(normal) {

  }

  SurfaceInteraction Plane::intersect(const Ray& ray) const {
    // TODO: Currently computation is done in world space -> maybe switch to object space
    
    SurfaceInteraction si;
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
        }
      }
    }

    return si;
  }

  
}