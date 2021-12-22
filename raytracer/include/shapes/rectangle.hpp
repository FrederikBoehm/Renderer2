#ifndef RECTANGLE_HPP
#define RECTANGLE_HPP

#include "shape.hpp"
#include "utility/qualifiers.hpp"
#include "intersect/hit_information.hpp"
#include "intersect/ray.hpp"

namespace rt {
  class CRay;

  class CRectangle : public CShape {
  public:
    DH_CALLABLE CRectangle();
    DH_CALLABLE CRectangle(const glm::vec3& worldPos, const glm::vec2& dimensions, const glm::vec3& normal);
  
    DH_CALLABLE SHitInformation intersect(const CRay& ray) const;
  private:
    glm::vec2 m_dimensions;
  };

  inline SHitInformation CRectangle::intersect(const CRay& ray) const {
    CRay rayObjectSpace = ray.transform(m_worldToModel);

    float cosine = rayObjectSpace.m_direction.y;
    if (cosine != 0.f) {
      float t = -rayObjectSpace.m_origin.y / cosine;
      if (t > 0.f) {
        glm::vec3 pos = rayObjectSpace.m_origin + t * rayObjectSpace.m_direction;
        if (inside(glm::vec3(pos.x, 0.f, pos.z), glm::vec3(-m_dimensions.x / 2.f, 0.f, -m_dimensions.y / 2.f), glm::vec3(m_dimensions.x / 2.f, 0.f, m_dimensions.y / 2.f))) {
          glm::vec3 posWorld = glm::vec3(m_modelToWorld * glm::vec4(pos, 1.f));
          float tWorld = glm::length(posWorld - ray.m_origin) / glm::length(ray.m_direction);
          return { true, posWorld, m_normal, glm::vec2(0.f), tWorld };
        }
      }
    }
    return { false, glm::vec3(), glm::vec3(), glm::vec2(0.f), 0.f };
  }
}
#endif
