#ifndef SHAPE_HPP
#define SHAPE_HPP

#include "utility/qualifiers.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

//#include "intersect/ray.hpp"

namespace rt {
  enum EShape {
    SPHERE,
    PLANE
  };

  class SurfaceInteraction;
  class Ray;

  class CShape {
  public:
    DH_CALLABLE EShape shape() const;

    DH_CALLABLE virtual ~CShape();
  protected:
    DH_CALLABLE CShape(EShape shape, const glm::vec3& worldPos, const glm::vec3& normal);
    DH_CALLABLE CShape(EShape shape);

    const EShape m_shape;
    glm::vec3 m_worldPos;
    glm::vec3 m_normal;
    glm::mat4 m_modelToWorld;
    glm::mat4 m_worldToModel;

  private:
    glm::mat4 getRotation(const glm::vec3& normal);
  };

  inline EShape CShape::shape() const {
    return m_shape;
  }
}
#endif // !SHAPE_HPP
