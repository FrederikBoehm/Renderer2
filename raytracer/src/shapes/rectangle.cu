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

  
}