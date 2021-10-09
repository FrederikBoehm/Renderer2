#include "shapes/plane.hpp"
#include "sampling/sampler.hpp"
#include "scene/interaction.hpp"

namespace rt {
  Plane::Plane() :
    CShape(EShape::PLANE), m_radius(1.0f) {

  }

  Plane::Plane(float radius) :
    CShape(EShape::PLANE), m_radius(radius) {

  }

  Plane::Plane(const glm::vec3& worldPos, float radius, const glm::vec3& normal) :
    CShape(EShape::PLANE, worldPos, normal),
    m_radius(radius) {

  }


  SHitInformation Plane::intersect(const CRay& ray) const {
    SHitInformation si;
    si.hit = false;

    float denominator = dot(ray.m_direction, m_normal);

    if (denominator != 0.0f) { // We have one hit

      float t = glm::dot(m_worldPos - ray.m_origin, m_normal) / denominator;
      if (t > 0.0f && t <= ray.m_t_max) {

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

  glm::vec3 Plane::sample(CSampler& sampler) const {
    glm::vec3 pd = m_radius * sampler.concentricSampleDisk();
    return glm::vec3(m_modelToWorld * glm::vec4(pd, 1.0f));
  }

  float Plane::pdf(const SInteraction& lightHit, const CRay& shadowRay) const {
    float distance = glm::length(lightHit.hitInformation.pos - shadowRay.m_origin);
    return 1 / area();
  }

  float Plane::area() const {
    return m_radius * m_radius * M_PI;
  }
}