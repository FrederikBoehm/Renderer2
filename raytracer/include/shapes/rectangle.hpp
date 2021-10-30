#ifndef RECTANGLE_HPP
#define RECTANGLE_HPP

#include "shape.hpp"
#include "utility/qualifiers.hpp"
#include "intersect/hit_information.hpp"

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
}
#endif
