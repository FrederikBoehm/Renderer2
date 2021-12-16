#ifndef DEVICE_SCENEOBJECT_HPP
#define DEVICE_SCENEOBJECT_HPP
#include "utility/qualifiers.hpp"
#include "shapes/rectangle.hpp"
#include "shapes/cuboid.hpp"
#include "shapes/circle.hpp"
#include "shapes/sphere.hpp"
#include "interaction.hpp"
#include <optix/optix_types.h>
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
  public:
    D_CALLABLE SInteraction intersect(const CRay& ray) const;
    D_CALLABLE CShape* shape() const;
    D_CALLABLE CMesh* mesh() const;
    D_CALLABLE float power() const;
    D_CALLABLE const glm::vec3& dimensions() const;
    D_CALLABLE ESceneobjectFlag flag() const;
    D_CALLABLE CMaterial* material() const;
    D_CALLABLE CMedium* medium() const;

  private:
    CShape* m_shape;
    CMesh* m_mesh;
    CMaterial* m_material;
    CMedium* m_medium;
    ESceneobjectFlag m_flag;

    //CDeviceSceneobject() {}
  };

  inline SInteraction CDeviceSceneobject::intersect(const CRay& ray) const {
    SInteraction si;
    switch (m_shape->shape()) {
    case EShape::CIRCLE:
      //si.hitInformation = ((CCircle*)m_shape)->intersect(ray);
      si.hitInformation = static_cast<const CCircle*>(m_shape)->intersect(ray);
      break;
    //case EShape::SPHERE:
    //  si.hitInformation = ((Sphere*)m_shape)->intersect(ray);
    //  break;
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

  inline float CDeviceSceneobject::power() const {
    if (m_flag == ESceneobjectFlag::GEOMETRY) {
      glm::vec3 L = m_material->Le();
      switch (m_shape->shape()) {
      case EShape::CIRCLE:
        return (L.x + L.y + L.z) * ((CCircle*)m_shape)->area();
      }
    }
    return 0.0f;
  }

  inline ESceneobjectFlag CDeviceSceneobject::flag() const {
    return m_flag;
  }

  inline CMaterial* CDeviceSceneobject::material() const {
    return m_material;
  }

  inline CMedium* CDeviceSceneobject::medium() const {
    return m_medium;
  }

  inline CMesh* CDeviceSceneobject::mesh() const {
    return m_mesh;
  }
}
#endif