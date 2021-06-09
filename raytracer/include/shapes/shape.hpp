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
    //DH_CALLABLE virtual SurfaceInteraction intersect(const Ray& ray) const = 0;

    DH_CALLABLE EShape shape() const;

    DH_CALLABLE virtual ~CShape();
  protected:
    //__device__ __host__ Shape(float* modelToWorld, float* worldToModel, float* worldPos);
    DH_CALLABLE CShape(EShape shape, const glm::vec3& worldPos);
    DH_CALLABLE CShape(EShape shape);

    const EShape m_shape;
    glm::vec3 m_worldPos;
    glm::mat4 m_modelToWorld;
    glm::mat4 m_worldToModel;

  };

  //inline Shape::Shape(float* modelToWorld, float* worldToModel, float* worldPos) :
  //  m_modelToWorld(glm::make_mat4(modelToWorld)), m_worldToModel(glm::make_mat4(worldToModel)), m_worldPos(glm::make_vec3(worldPos)) {

  //}

  inline EShape CShape::shape() const {
    return m_shape;
  }
}
#endif // !SHAPE_HPP
