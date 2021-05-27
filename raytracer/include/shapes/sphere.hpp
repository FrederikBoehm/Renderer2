#ifndef SPHERE_HXX
#define SPHERE_HXX

#include "cuda_runtime.h"

#include "shape.hpp"
#include "intersect/surface_interaction.hpp"
#include "intersect/ray.hpp"
namespace rt {
  class Ray;

  class Sphere : public Shape {
  public:
    __device__ __host__ Sphere(float radius);
    __device__ __host__ Sphere(float* modelToWorld, float* worldToModel, float* worldPos, float radius);

    __device__ __host__ SurfaceInteraction intersect(const Ray& ray);

  private:
    float m_radius;
  };

  inline Sphere::Sphere(float radius) : Shape(), m_radius(radius) {
  }

  inline Sphere::Sphere(float* modelToWorld, float* worldToModel, float* worldPos, float radius) :
    Shape(modelToWorld, worldToModel, worldPos), m_radius(radius) {

  }

  inline SurfaceInteraction Sphere::intersect(const Ray& ray) {
    Ray rayModelSpace = ray.toObjectSpace(m_worldToModel);
    float discriminant = glm::pow(glm::dot(rayModelSpace.m_direction, rayModelSpace.m_origin), 2.0f) - (glm::pow(glm::length(rayModelSpace.m_origin), 2.0f) - m_radius * m_radius);

    SurfaceInteraction si;
    si.hit = false;
    if (discriminant == 0.0f) {
      float d = -glm::dot(ray.m_direction, ray.m_origin) + std::sqrt(discriminant);
      if (d > 0) { // Intersection in front of ray origin
        si.hit = true;
        glm::vec3 intersectionObjectSpace = rayModelSpace.m_origin + d * rayModelSpace.m_direction;
        si.pos = glm::vec3(m_modelToWorld * glm::vec4(intersectionObjectSpace, 1.0f));
      }
    }
    else if (discriminant > 0.0f) {
      float sqrtDiscriminant = glm::sqrt(discriminant);
      float d1 = -glm::dot(ray.m_direction, ray.m_origin) + sqrtDiscriminant;
      float d2 = -glm::dot(ray.m_direction, ray.m_origin) - sqrtDiscriminant;

      float minimum = glm::min(d1, d2);
      float maximum = glm::max(d1, d2);
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
