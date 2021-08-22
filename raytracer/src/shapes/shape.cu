#include <glm/gtc/type_ptr.hpp>

#include "shapes/shape.hpp"


namespace rt {

  CShape::CShape(EShape shape, const glm::vec3& worldPos, const glm::vec3& normal) :
    m_shape(shape),
    m_worldPos(worldPos),
    m_normal(normal),
    m_modelToWorld(glm::translate(glm::mat4(1.0f), worldPos) * getRotation(normal)),
    m_worldToModel(glm::inverse(m_modelToWorld)) {

  }

  CShape::CShape(EShape shape) :
    m_shape(shape), m_modelToWorld(glm::mat4(1.0f)), m_worldToModel(glm::mat4(1.0f)), m_worldPos(glm::vec3(0.0f)), m_normal(glm::vec3(0.0f, 1.0f, 0.0f)) {

  }

  CShape::~CShape() {
  }

  glm::mat4 CShape::getRotation(const glm::vec3& normal) {
    float cos = glm::dot(normal, glm::vec3(0.0f, 1.0f, 0.0f));
    if (cos == 1.0f) {
      return glm::mat4(1.0f);
    }
    else if (cos == -1.0f) {
      return glm::mat4(glm::mat3(-1.0f));
    }
    else {
      float angle = glm::acos(glm::dot(normal, glm::vec3(0.0f, 1.0f, 0.0f)));
      glm::vec3 rotationAxis = glm::normalize(glm::cross(normal, glm::vec3(0.0f, 1.0f, 0.0f)));
      return glm::rotate(glm::mat4(1.0f), angle, rotationAxis);
    }
    
  }

  
}
