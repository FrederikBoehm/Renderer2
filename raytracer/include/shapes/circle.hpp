#ifndef CIRCLE_HPP
#define CIRCLE_HPP

#include "shape.hpp"
#include "intersect/hit_information.hpp"
#include "intersect/ray.hpp"
#include "sampling/sampler.hpp"
#include "scene/interaction.hpp"
#include "scene/types.hpp"

namespace rt {
  class CSampler;
  class SInteraction;

  class CCircle : public CShape {
  public:
    H_CALLABLE CCircle();
    H_CALLABLE CCircle(float radius);
    H_CALLABLE CCircle(const glm::vec3& worldPos, float radius, const glm::vec3& normal);

    DH_CALLABLE SHitInformation intersect(const CRay& ray) const;
    D_CALLABLE glm::vec3 sample(CSampler& sampler) const;
    DH_CALLABLE float pdf(const SInteraction& lightHit, const CRay& shadowRay) const;
    DH_CALLABLE float area() const;

    H_CALLABLE SBuildInputWrapper getOptixBuildInput();
    H_CALLABLE OptixProgramGroup getOptixProgramGroup() const;
  
  private:
    float m_radius;

    H_CALLABLE OptixAabb getAABB() const;
  };

  inline SHitInformation CCircle::intersect(const CRay& ray) const {
    SHitInformation si;
    si.hit = false;

    float denominator = glm::dot(ray.m_direction, m_normal);

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

  inline glm::vec3 CCircle::sample(CSampler& sampler) const {
    glm::vec3 pd = m_radius * sampler.concentricSampleDisk();
    return glm::vec3(m_modelToWorld * glm::vec4(pd, 1.0f));
  }

  inline float CCircle::pdf(const SInteraction& lightHit, const CRay& shadowRay) const {
    float distance = glm::length(lightHit.hitInformation.pos - shadowRay.m_origin);
    return 1 / area();
  }

  inline float CCircle::area() const {
    return m_radius * m_radius * M_PI;
  }

  
}

#endif // !CIRCLE_HPP

