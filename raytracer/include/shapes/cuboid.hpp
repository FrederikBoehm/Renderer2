#ifndef CUBOID_HPP
#define CUBOID_HPP

#include "utility/qualifiers.hpp"
#include "shapes/shape.hpp"
#include "rectangle.hpp"
#include "scene/types.hpp"

namespace rt {
  struct SHitInformation;
  class CRay;

  class CCuboid : public CShape {
  public:
    DH_CALLABLE CCuboid(const glm::vec3& worldPos, const glm::vec3& dimensions, const glm::vec3& normal);
    
    DH_CALLABLE SHitInformation intersect(const CRay& ray) const;
    DH_CALLABLE const glm::vec3& dimensions() const;

    H_CALLABLE SBuildInputWrapper getOptixBuildInput();
    H_CALLABLE OptixProgramGroup getOptixProgramGroup() const;
  private:
    glm::vec3 m_dimensions;
    CRectangle m_faces[6];
    
    H_CALLABLE OptixAabb getAABB() const;
  };

  inline const glm::vec3& CCuboid::dimensions() const {
    return m_dimensions;
  }

  inline SHitInformation CCuboid::intersect(const CRay& ray) const {
    SHitInformation hit = { false, glm::vec3(), glm::vec3(), FLT_MAX };
    for (uint8_t i = 0; i < 6; ++i) {
      SHitInformation currentHit = m_faces[i].intersect(ray);
      if (currentHit.hit && currentHit.t < hit.t) {
        hit = currentHit;
      }
    }
    return hit;
  }
}
#endif // !CUBOID_HPP
