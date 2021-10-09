#ifndef PLANE_HPP
#define PLANE_HPP

#include "shape.hpp"
#include "intersect/hit_information.hpp"
#include "intersect/ray.hpp"

namespace rt {
  class CSampler;
  class SInteraction;

  class Plane : public CShape { // Our plane is actually a circle, this simplifies intersection
  public:
    DH_CALLABLE Plane();
    DH_CALLABLE Plane(float radius);
    DH_CALLABLE Plane(const glm::vec3& worldPos, float radius, const glm::vec3& normal);

    DH_CALLABLE SHitInformation intersect(const CRay& ray) const;
    D_CALLABLE glm::vec3 sample(CSampler& sampler) const;
    DH_CALLABLE float pdf(const SInteraction& lightHit, const CRay& shadowRay) const;
    DH_CALLABLE float area() const;
  
  private:
    float m_radius;
  };

  
}

#endif // !PLANE_HPP

