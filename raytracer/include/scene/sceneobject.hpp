#ifndef SCENEOBJECT_HPP
#define SCENEOBJECT_HPP
#include "shapes/shape.hpp"
#include "shapes/circle.hpp"
#include "shapes/sphere.hpp"
#include "material/material.hpp"
#include "interaction.hpp"
#include <memory>
#include "device_sceneobject.hpp"
#include "backend/types.hpp"
#include "mesh/mesh.hpp"

namespace rt {
  

  class CSceneobjectConnection {
  public:
    CSceneobjectConnection(CHostSceneobject* hostSceneobject);
    CSceneobjectConnection(const CSceneobjectConnection&& connection);
    void allocateDeviceMemory();
    void setDeviceSceneobject(CDeviceSceneobject* destination);
    void copyToDevice();
    void freeDeviceMemory();
    const CDeviceSceneobject* deviceSceneobject() const;
  private:
    CHostSceneobject* m_hostSceneobject = nullptr;
    CDeviceSceneobject* m_deviceSceneobject = nullptr;

    CShape* m_deviceShape = nullptr;
    CMesh* m_deviceMesh = nullptr;
    CMaterial* m_deviceMaterial = nullptr;
    CMedium* m_deviceMedium = nullptr;
  };

  class CHostSceneobject {
    friend class CSceneobjectConnection;
  public:
    CHostSceneobject(CShape* shape, const glm::vec3& le);
    CHostSceneobject(CShape* shape, const glm::vec3& diffuseReflection, float diffuseRougness, const glm::vec3& specularReflection, float alphaX, float alphaY, float etaI, float etaT);
    CHostSceneobject(CShape* shape, CMedium* medium);
    CHostSceneobject(CNVDBMedium* medium);
    CHostSceneobject(CMesh* mesh, const glm::vec3& diffuseReflection, float diffuseRougness, const glm::vec3& specularReflection, float alphaX, float alphaY, float etaI, float etaT);
    CHostSceneobject(CHostSceneobject&& sceneobject);
    ~CHostSceneobject();

    float power() const;

    void allocateDeviceMemory();
    void setDeviceSceneobject(CDeviceSceneobject* destination);
    void copyToDevice();
    void freeDeviceMemory();
    void buildOptixAccel();
    OptixInstance getOptixInstance(uint32_t instanceId, uint32_t sbtOffset) const;
    SRecord<const CDeviceSceneobject*> getSBTHitRecord() const;
  private:
    std::shared_ptr<CShape> m_shape;
    std::shared_ptr<CMesh> m_mesh;
    std::shared_ptr<CMaterial> m_material;
    std::shared_ptr<CMedium> m_medium;
    ESceneobjectFlag m_flag;
    float m_absorption;

    OptixAabb m_aabb;
    OptixTraversableHandle m_traversableHandle;
    CUdeviceptr m_deviceGasBuffer;

    CSceneobjectConnection m_hostDeviceConnection;

    static std::shared_ptr<CShape> getShape(EShape shape, const glm::vec3& worldPos, float radius, const glm::vec3& normal);
    OptixProgramGroup getOptixProgramGroup() const;
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

  inline const CDeviceSceneobject* CSceneobjectConnection::deviceSceneobject() const {
    return m_deviceSceneobject;
  }

}
#endif // !SCENEOBJECT_HPP
