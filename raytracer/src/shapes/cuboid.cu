#include "shapes/cuboid.hpp"
#include "intersect/hit_information.hpp"

namespace rt {
  CCuboid::CCuboid(const glm::vec3& worldPos, const glm::vec3& dimensions, const glm::vec3& normal) :
    CShape(EShape::CUBOID, worldPos, normal),
    m_dimensions(dimensions),
    m_faces{
      CRectangle(worldPos + glm::vec3(dimensions.x * 0.5f, 0.f, 0.f), glm::vec2(dimensions.y, dimensions.z), glm::vec3(1.f, 0.f, 0.f)),
      CRectangle(worldPos + glm::vec3(-dimensions.x * 0.5f, 0.f, 0.f), glm::vec2(dimensions.y, dimensions.z), glm::vec3(-1.f, 0.f, 0.f)),
      CRectangle(worldPos + glm::vec3(0.f, dimensions.y * 0.5f, 0.f), glm::vec2(dimensions.x, dimensions.z), glm::vec3(0.f, 1.f, 0.f)),
      CRectangle(worldPos + glm::vec3(0.f, -dimensions.y * 0.5f, 0.f), glm::vec2(dimensions.x, dimensions.z), glm::vec3(0.f, -1.f, 0.f)),
      CRectangle(worldPos + glm::vec3(0.f, 0.f, dimensions.z * 0.5f), glm::vec2(dimensions.x, dimensions.y), glm::vec3(0.f, 0.f, 1.f)),
      CRectangle(worldPos + glm::vec3(0.f, 0.f, -dimensions.z * 0.5f), glm::vec2(dimensions.x, dimensions.y), glm::vec3(0.f, 0.f, -1.f)),
    } {
    }

  SHitInformation CCuboid::intersect(const CRay& ray) const {
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