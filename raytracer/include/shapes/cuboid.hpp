#ifndef CUBOID_HPP
#define CUBOID_HPP

#include "utility/qualifiers.hpp"
#include "shapes/shape.hpp"
#include "rectangle.hpp"

namespace rt {
  struct SHitInformation;
  class CRay;

  class CCuboid : public CShape {
  public:
    DH_CALLABLE CCuboid(const glm::vec3& worldPos, const glm::vec3& dimensions, const glm::vec3& normal);
    
    DH_CALLABLE SHitInformation intersect(const CRay& ray) const;
  private:
    glm::vec3 m_dimensions;
    CRectangle m_faces[6];
  };
}
#endif // !CUBOID_HPP
