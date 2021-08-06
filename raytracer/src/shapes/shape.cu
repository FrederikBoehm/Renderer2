#include <glm/gtc/type_ptr.hpp>

#include "shapes/shape.hpp"


namespace rt {

  CShape::CShape(EShape shape, const glm::vec3& worldPos) :
    m_shape(shape),
    m_worldPos(worldPos),
    m_modelToWorld(glm::translate(glm::mat4(1.0f), worldPos)),
    m_worldToModel(glm::inverse(m_modelToWorld)) {

  }

  CShape::CShape(EShape shape) :
    m_shape(shape), m_modelToWorld(glm::mat4(1.0f)), m_worldToModel(glm::mat4(1.0f)), m_worldPos(glm::vec3(0.0f)) {

  }

  CShape::~CShape() {
  }

  
}
