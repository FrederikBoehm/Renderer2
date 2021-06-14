#ifndef SCENEOBJECT_HPP
#define SCENEOBJECT_HPP
#include "shapes/shape.hpp"
#include "shapes/plane.hpp"
#include "shapes/sphere.hpp"
#include "material/material.hpp"
#include "surface_interaction.hpp"
namespace rt {
  class CHostSceneobject;

  class CDeviceSceneobject {
    friend class CSceneobjectConnection;
  public:
    D_CALLABLE SSurfaceInteraction intersect(const Ray& ray) const;

  private:
    CShape* m_shape;
    CMaterial m_material;

    //CDeviceSceneobject() {}
  };

  class CSceneobjectConnection {
  public:
    CSceneobjectConnection(CHostSceneobject* hostSceneobject);
    CSceneobjectConnection(const CSceneobjectConnection&& connection);
    void allocateDeviceMemory();
    void setDeviceSceneobject(CDeviceSceneobject* destination);
    void copyToDevice();
    void freeDeviceMemory();
  private:
    CHostSceneobject* m_hostSceneobject = nullptr;
    CDeviceSceneobject* m_deviceSceneobject = nullptr;

    CShape* m_deviceShape = nullptr;
  };

  class CHostSceneobject {
    friend class CSceneobjectConnection;
  public:
    CHostSceneobject(EShape shape, const glm::vec3& worldPos, float radius, const glm::vec3& normal, const glm::vec3& albedo);
    CHostSceneobject::CHostSceneobject(CHostSceneobject&& sceneobject);

    void allocateDeviceMemory();
    void setDeviceSceneobject(CDeviceSceneobject* destination);
    void copyToDevice();
    void freeDeviceMemory();
  private:
    CShape* m_shape;
    CMaterial m_material;
    CSceneobjectConnection m_hostDeviceConnection;

    static CShape* getShape(EShape shape, const glm::vec3& worldPos, float radius, const glm::vec3& normal);
  };

  inline void CHostSceneobject::allocateDeviceMemory() {
    m_hostDeviceConnection.allocateDeviceMemory();
  }

  inline void CHostSceneobject::setDeviceSceneobject(CDeviceSceneobject* destination) {
    m_hostDeviceConnection.setDeviceSceneobject(destination);
  }

  inline void CHostSceneobject::copyToDevice() {
    m_hostDeviceConnection.copyToDevice();
  }

  inline void CSceneobjectConnection::setDeviceSceneobject(CDeviceSceneobject* destination) {
    m_deviceSceneobject = destination;
  }

  inline void CHostSceneobject::freeDeviceMemory() {
    m_hostDeviceConnection.freeDeviceMemory();
  }
}
#endif // !SCENEOBJECT_HPP
