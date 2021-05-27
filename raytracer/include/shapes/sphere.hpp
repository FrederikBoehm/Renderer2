#ifndef SPHERE_HXX
#define SPHERE_HXX

#include "cuda_runtime.h"

#include "shape.hpp"
#include "intersect/surface_interaction.hpp"
#include "intersect/ray.hpp"
namespace rt {
  class Sphere : public Shape {
  public:
    __device__ __host__ Sphere(float radius);
    //__device__ __host__ Sphere(float* modelToWorld, float* worldToModel, float* worldPos, float radius);
    __device__ __host__ Sphere(const glm::vec3& worldPos, float radius);

    __device__ __host__ virtual SurfaceInteraction intersect(const Ray& ray) const override;

  private:
    float m_radius;
  };

  inline Sphere::Sphere(float radius) : Shape(), m_radius(radius) {
  }

  //inline Sphere::Sphere(float* modelToWorld, float* worldToModel, float* worldPos, float radius) :
  //  Shape(modelToWorld, worldToModel, worldPos), m_radius(radius) {

  //}

  inline Sphere::Sphere(const glm::vec3& worldPos, float radius) :
    Shape(worldPos),
    m_radius(radius) {

  }

  inline SurfaceInteraction Sphere::intersect(const Ray& ray) const {
    // Check the intersection, there might be an error
    // Test with Sphere Origin (1.0, 2.0, 3.0), Radius 5
    // Ray Origin (0.0, 2.0, 3.0), Direction (1.0, 0.0, 0.0)
    Ray rayModelSpace = ray.toObjectSpace(m_worldToModel);
    
    float a = glm::dot(rayModelSpace.m_direction, rayModelSpace.m_direction);
    float b = 2.0f * glm::dot(rayModelSpace.m_direction, rayModelSpace.m_origin);
    float c = glm::dot(rayModelSpace.m_origin, rayModelSpace.m_origin) - m_radius * m_radius;

    SurfaceInteraction si;
    si.hit = false;

    float discriminant = b * b - 4 * a * c;
    if (discriminant == 0.0f) {
      float t = (-b + glm::sqrt(discriminant)) / (2 * a);
      if (t > 0) { // Intersection in front of ray origin
        si.hit = true;
        glm::vec3 intersectionObjectSpace = rayModelSpace.m_origin + t * rayModelSpace.m_direction;
        si.pos = glm::vec3(m_modelToWorld * glm::vec4(intersectionObjectSpace, 1.0f));
      }
    }
    else if (discriminant > 0.0f) {
      float sqrtDiscriminant = glm::sqrt(discriminant);
      float denominator =  1 / (2.0f * a);
      float t1 = (-b + sqrtDiscriminant) * denominator;
      float t2 = (-b - sqrtDiscriminant) * denominator;

      float minimum = glm::min(t1, t2);
      float maximum = glm::max(t1, t2);
      if (maximum > 0.0f) {
        if (minimum > 0.0f) {
          si.hit = true;
          glm::vec3 intersectionObjectSpace = rayModelSpace.m_origin + minimum * rayModelSpace.m_direction;
          si.pos = glm::vec3(m_modelToWorld * glm::vec4(intersectionObjectSpace, 1.0f));
        }
        else {
          si.hit = true;
          glm::vec3 intersectionObjectSpace = rayModelSpace.m_origin + maximum * rayModelSpace.m_direction;
          si.pos = glm::vec3(m_modelToWorld * glm::vec4(intersectionObjectSpace, 1.0f));
        }
      }
    }

    return si;
  }
}
#endif // !SPHERE_HXX
