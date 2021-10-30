#include "shapes/rectangle.hpp"
#include "intersect/ray.hpp"
#include "utility/functions.hpp"

namespace rt {
  CRectangle::CRectangle() :
    CShape(EShape::RECTANGLE) {

  }

  CRectangle::CRectangle(const glm::vec3& worldPos, const glm::vec2& dimensions, const glm::vec3& normal):
    CShape(EShape::RECTANGLE, worldPos, normal),
    m_dimensions(dimensions) {

    }

  SHitInformation CRectangle::intersect(const CRay& ray) const {
    CRay rayObjectSpace = ray.transform(m_worldToModel);

    float cosine = rayObjectSpace.m_direction.y;
    if (cosine != 0.f) {
      float t = -rayObjectSpace.m_origin.y / cosine;
      if (t > 0.f) {
        glm::vec3 pos = rayObjectSpace.m_origin + t * rayObjectSpace.m_direction;
        if (inside(glm::vec3(pos.x, 0.f, pos.z), glm::vec3(-m_dimensions.x / 2.f, 0.f, -m_dimensions.y / 2.f), glm::vec3(m_dimensions.x / 2.f, 0.f, m_dimensions.y / 2.f))) {
          glm::vec3 posWorld = glm::vec3(m_modelToWorld * glm::vec4(pos, 1.f));
          float tWorld = glm::length(posWorld - ray.m_origin) / glm::length(ray.m_direction);
          return { true, posWorld, m_normal, tWorld };
        }
      }
    }
    return { false, glm::vec3(), glm::vec3(), 0.f };
  }
}