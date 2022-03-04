#ifndef SPHERE_HXX
#define SPHERE_HXX

#include "utility/qualifiers.hpp"

#include "shape.hpp"
#include "intersect/hit_information.hpp"
#include "intersect/ray.hpp"
namespace rt {
  class Sphere : public CShape {
  public:
    H_CALLABLE Sphere();
    H_CALLABLE Sphere(float radius);
    DH_CALLABLE Sphere(const glm::vec3& worldPos, float radius, const glm::vec3& normal);

    DH_CALLABLE SHitInformation intersect(const CRay& ray) const;

    H_CALLABLE SBuildInputWrapper getOptixBuildInput();
    H_CALLABLE OptixProgramGroup getOptixProgramGroup() const;
    
    H_CALLABLE OptixAabb getAABB() const;
  private:
    float m_radius;

  };

  inline Sphere::Sphere(const glm::vec3& worldPos, float radius, const glm::vec3& normal) :
    CShape(EShape::SPHERE, worldPos, normal),
    m_radius(radius) {

  }

  inline SHitInformation Sphere::intersect(const CRay& ray) const {
    CRay rayModelSpace = ray.transform(m_worldToModel);

    float a = glm::dot(rayModelSpace.m_direction, rayModelSpace.m_direction);
    float b = 2.0f * glm::dot(rayModelSpace.m_direction, rayModelSpace.m_origin);
    float c = glm::dot(rayModelSpace.m_origin, rayModelSpace.m_origin) - m_radius * m_radius;

    SHitInformation si;
    si.hit = false;

    float discriminant = b * b - 4 * a * c;
    if (discriminant == 0.0f) {
      float t = (-b + glm::sqrt(discriminant)) / (2 * a);
      if (t > 0 && t <= ray.m_t_max) { // Intersection in front of ray origin
        si.hit = true;
        glm::vec3 intersectionObjectSpace = rayModelSpace.m_origin + t * rayModelSpace.m_direction;
        si.pos = m_modelToWorld * glm::vec4(intersectionObjectSpace, 1.0f);
        si.normal = glm::normalize(m_modelToWorld * glm::vec4(intersectionObjectSpace, 0.0f));
        si.normalG = si.normal;
        si.tc = glm::vec2(0.f);
        si.t = t;
      }
    }
    else if (discriminant > 0.0f) {
      float sqrtDiscriminant = glm::sqrt(discriminant);
      float denominator = 1 / (2.0f * a);
      float t1 = (-b + sqrtDiscriminant) * denominator;
      float t2 = (-b - sqrtDiscriminant) * denominator;

      float minimum = glm::min(t1, t2);
      float maximum = glm::max(t1, t2);
      if (maximum > 0.0f) {
        if (minimum > 0.0f && minimum <= ray.m_t_max) {
          si.hit = true;
          glm::vec3 intersectionObjectSpace = rayModelSpace.m_origin + minimum * rayModelSpace.m_direction;
          si.pos = m_modelToWorld * glm::vec4(intersectionObjectSpace, 1.0f);
          si.normal = glm::normalize(m_modelToWorld * glm::vec4(intersectionObjectSpace, 0.0f));
          si.normalG = si.normal;
          si.tc = glm::vec2(0.f);
          si.t = minimum;
        }
        else if (minimum <= 0.0f && maximum <= ray.m_t_max) {
          si.hit = true;
          glm::vec3 intersectionObjectSpace = rayModelSpace.m_origin + maximum * rayModelSpace.m_direction;
          si.pos = m_modelToWorld * glm::vec4(intersectionObjectSpace, 1.0f);
          si.normal = glm::normalize(m_modelToWorld * glm::vec4(intersectionObjectSpace, 0.0f));
          si.normalG = si.normal;
          si.tc = glm::vec2(0.f);
          si.t = maximum;
        }
      }
    }

    return si;
  }

}
#endif // !SPHERE_HXX
