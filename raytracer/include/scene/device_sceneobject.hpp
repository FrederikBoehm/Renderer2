#ifndef DEVICE_SCENEOBJECT_HPP
#define DEVICE_SCENEOBJECT_HPP
#include "utility/qualifiers.hpp"
#include "shapes/rectangle.hpp"
#include "shapes/cuboid.hpp"
#include "shapes/circle.hpp"
#include "shapes/sphere.hpp"
#include "interaction.hpp"
#include <optix/optix_types.h>
#include "sceneobject_mask.hpp"
namespace rt {
  class CHostSceneobject;
  class CNVDBMedium;
  class CMesh;

  enum ESceneobjectFlag {
    GEOMETRY,
    VOLUME
  };

  class CDeviceSceneobject {
    friend class CSceneobjectConnection;
    friend struct SSharedMemoryInitializer;
    friend __global__ void getTransformPointers(CDeviceSceneobject* sceneobject, glm::mat4x3** modelToWorld, glm::mat4x3** worldToModel);
  public:
    D_CALLABLE SInteraction intersect(const CRay& ray) const;
    D_CALLABLE CShape* shape() const;
    D_CALLABLE CMesh* mesh() const;
    D_CALLABLE float power() const;
    D_CALLABLE const glm::vec3& dimensions() const;
    D_CALLABLE ESceneobjectFlag flag() const;
    D_CALLABLE CMaterial* material() const;
    D_CALLABLE CMediumInstance* medium() const;
    D_CALLABLE const glm::mat4x3& modelToWorld() const;
    D_CALLABLE const glm::mat4x3& worldToModel() const;

  private:
    CShape* m_shape;
    CMesh* m_mesh;
    CMaterial* m_material;
    CMediumInstance* m_medium;
    ESceneobjectFlag m_flag;

    glm::mat4x3 m_modelToWorld;
    glm::mat4x3 m_worldToModel;

    ESceneobjectMask m_mask;

    //CDeviceSceneobject() {}
  };

  inline SInteraction CDeviceSceneobject::intersect(const CRay& ray) const {
    SInteraction si;
    switch (m_shape->shape()) {
    case EShape::CIRCLE:
      //si.hitInformation = ((CCircle*)m_shape)->intersect(ray);
      si.hitInformation = static_cast<const CCircle*>(m_shape)->intersect(ray);
      break;
    case EShape::SPHERE:
      si.hitInformation = ((Sphere*)m_shape)->intersect(ray);
      break;
    //case EShape::RECTANGLE:
    //  si.hitInformation = ((CRectangle*)m_shape)->intersect(ray);
    //  break;
    case EShape::CUBOID:
      si.hitInformation = static_cast<const CCuboid*>(m_shape)->intersect(ray);
      break;
    }
    si.material = m_material;
    si.medium = m_medium;
    si.object = this;
    return si;
  }

  inline const glm::vec3& CDeviceSceneobject::dimensions() const {
    switch (m_shape->shape()) {
    case EShape::CUBOID:
      return ((CCuboid*)m_shape)->dimensions();
    }
    return glm::vec3(0.f);
  }

  inline CShape* CDeviceSceneobject::shape() const {
    return m_shape;
  }

  inline ESceneobjectFlag CDeviceSceneobject::flag() const {
    return m_flag;
  }

  inline CMaterial* CDeviceSceneobject::material() const {
    return m_material;
  }

  inline CMediumInstance* CDeviceSceneobject::medium() const {
    return m_medium;
  }

  inline CMesh* CDeviceSceneobject::mesh() const {
    return m_mesh;
  }

  inline const glm::mat4x3& CDeviceSceneobject::modelToWorld() const {
    return m_modelToWorld;
  }

  inline const glm::mat4x3& CDeviceSceneobject::worldToModel() const {
    return m_worldToModel;
  }
}
#endif