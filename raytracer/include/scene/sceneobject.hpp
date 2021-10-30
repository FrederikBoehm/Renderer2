#ifndef SCENEOBJECT_HPP
#define SCENEOBJECT_HPP
#include "shapes/shape.hpp"
#include "shapes/circle.hpp"
#include "shapes/sphere.hpp"
#include "material/material.hpp"
#include "interaction.hpp"
#include <memory>

namespace rt {
  class CHostSceneobject;

  enum ESceneobjectFlag {
    GEOMETRY,
    VOLUME
  };

  class CDeviceSceneobject {
    friend class CSceneobjectConnection;
    friend struct SSharedMemoryInitializer;
  public:
    D_CALLABLE SInteraction intersect(const CRay& ray);
    D_CALLABLE CShape* shape() const;
    D_CALLABLE float power() const;

  private:
    CShape* m_shape;
    CMaterial* m_material;
    CMedium* m_medium;
    ESceneobjectFlag m_flag;

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
    CMaterial* m_deviceMaterial = nullptr;
    CMedium* m_deviceMedium = nullptr;
  };

  class CHostSceneobject {
    friend class CSceneobjectConnection;
  public:
    CHostSceneobject(const CShape* shape, const glm::vec3& le);
    CHostSceneobject(const CShape* shape, const glm::vec3& diffuseReflection, float diffuseRougness, const glm::vec3& specularReflection, float alphaX, float alphaY, float etaI, float etaT);
    CHostSceneobject(const CShape* shape, CMedium* medium);
    CHostSceneobject(CHostSceneobject&& sceneobject);

    float power() const;

    void allocateDeviceMemory();
    void setDeviceSceneobject(CDeviceSceneobject* destination);
    void copyToDevice();
    void freeDeviceMemory();
  private:
    std::shared_ptr<const CShape> m_shape;
    std::shared_ptr<CMaterial> m_material;
    std::shared_ptr<CMedium> m_medium;
    ESceneobjectFlag m_flag;
    float m_absorption;

    CSceneobjectConnection m_hostDeviceConnection;

    static std::shared_ptr<CShape> getShape(EShape shape, const glm::vec3& worldPos, float radius, const glm::vec3& normal);
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
