#include <glm/glm.hpp>
#include <iostream>

#include "shapes/sphere.hpp"
#include "intersect/ray.hpp"

namespace rt {
  Sphere::Sphere() :CShape(EShape::SPHERE), m_radius(1.0f) {

  }

  Sphere::Sphere(float radius) : CShape(EShape::SPHERE), m_radius(radius) {
  }

  Sphere::Sphere(const glm::vec3& worldPos, float radius) :
    CShape(EShape::SPHERE, worldPos),
    m_radius(radius) {

  }

  SHitInformation Sphere::intersect(const Ray& ray) const {
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
        si.normal = glm::normalize(glm::vec3(m_modelToWorld * glm::vec4(intersectionObjectSpace, 0.0f)));
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
          si.normal = glm::normalize(glm::vec3(m_modelToWorld * glm::vec4(intersectionObjectSpace, 0.0f)));
          si.t = minimum;
        }
        else {
          si.hit = true;
          glm::vec3 intersectionObjectSpace = rayModelSpace.m_origin + maximum * rayModelSpace.m_direction;
          si.pos = glm::vec3(m_modelToWorld * glm::vec4(intersectionObjectSpace, 1.0f));
          si.normal = glm::normalize(glm::vec3(m_modelToWorld * glm::vec4(intersectionObjectSpace, 0.0f)));
          si.t = maximum;
        }
      }
    }

    return si;
  }
}