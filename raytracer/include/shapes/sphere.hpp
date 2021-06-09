#ifndef SPHERE_HXX
#define SPHERE_HXX

#include "utility/qualifiers.hpp"

#include "shape.hpp"
#include "intersect/hit_information.hpp"
#include "intersect/ray.hpp"
namespace rt {
  class Sphere : public CShape {
  public:
    DH_CALLABLE Sphere();
    DH_CALLABLE Sphere(float radius);
    //__device__ __host__ Sphere(float* modelToWorld, float* worldToModel, float* worldPos, float radius);
    DH_CALLABLE Sphere(const glm::vec3& worldPos, float radius);

    //DH_CALLABLE virtual SurfaceInteraction intersect(const Ray& ray) const override;
    DH_CALLABLE SHitInformation intersect(const Ray& ray) const;

    //DH_CALLABLE SurfaceInteraction Sphere::intersect(const Ray* ray) const;

  private:
    float m_radius;
  };

  inline SHitInformation Sphere::intersect(const Ray& ray) const {
    // Check the intersection, there might be an error
    // Test with Sphere Origin (1.0, 2.0, 3.0), Radius 5
    // Ray Origin (0.0, 2.0, 3.0), Direction (1.0, 0.0, 0.0)
    Ray rayModelSpace = ray.transform(m_worldToModel);

    float a = glm::dot(rayModelSpace.m_direction, rayModelSpace.m_direction);
    float b = 2.0f * glm::dot(rayModelSpace.m_direction, rayModelSpace.m_origin);
    float c = glm::dot(rayModelSpace.m_origin, rayModelSpace.m_origin) - m_radius * m_radius;

    SHitInformation si;
    si.hit = false;

    float discriminant = b * b - 4 * a * c;
    if (discriminant == 0.0f) {
      float t = (-b + glm::sqrt(discriminant)) / (2 * a);
      if (t > 0) { // Intersection in front of ray origin
        si.hit = true;
        glm::vec3 intersectionObjectSpace = rayModelSpace.m_origin + t * rayModelSpace.m_direction;
        si.pos = glm::vec3(m_modelToWorld * glm::vec4(intersectionObjectSpace, 1.0f));
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
        if (minimum > 0.0f) {
          si.hit = true;
          glm::vec3 intersectionObjectSpace = rayModelSpace.m_origin + minimum * rayModelSpace.m_direction;
          si.pos = glm::vec3(m_modelToWorld * glm::vec4(intersectionObjectSpace, 1.0f));
          si.t = minimum;
        }
        else {
          si.hit = true;
          glm::vec3 intersectionObjectSpace = rayModelSpace.m_origin + maximum * rayModelSpace.m_direction;
          si.pos = glm::vec3(m_modelToWorld * glm::vec4(intersectionObjectSpace, 1.0f));
          si.t = maximum;
        }
      }
    }

    return si;
  }

}
#endif // !SPHERE_HXX
