#include <glm/gtc/type_ptr.hpp>

#include "shapes/shape.hpp"
#include "shapes/circle.hpp"
#include "shapes/cuboid.hpp"
#include "shapes/sphere.hpp"

namespace rt {


  SBuildInputWrapper CShape::getOptixBuildInput() {
    switch (m_shape) {
      case CIRCLE:
        return static_cast<CCircle*>(this)->getOptixBuildInput();
      case CUBOID:
        return static_cast<CCuboid*>(this)->getOptixBuildInput();
      case SPHERE:
        return static_cast<Sphere*>(this)->getOptixBuildInput();
    }
    fprintf(stderr, "[ERROR] No Build input found for given shape type\n");
  }

  OptixProgramGroup CShape::getOptixProgramGroup() const {
    switch (m_shape) {
    case CIRCLE:
      return static_cast<const CCircle*>(this)->getOptixProgramGroup();
    case CUBOID:
      return static_cast<const CCuboid*>(this)->getOptixProgramGroup();
    case SPHERE:
      return static_cast<const Sphere*>(this)->getOptixProgramGroup();
    }
    fprintf(stderr, "[ERROR] No OptixProgramGroup found for given shape type\n");
    return OptixProgramGroup();
  }

  
}
