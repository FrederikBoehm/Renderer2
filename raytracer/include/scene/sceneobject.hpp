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
#include "medium/medium_instance.hpp"
#include "scene/sceneobject_mask.hpp"

namespace rt {
  class CMedium;

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
    CMediumInstance* m_deviceMedium = nullptr;
  };

  class CHostSceneobject {
    friend class CSceneobjectConnection;
  public:
    CHostSceneobject(CShape* shape, const glm::vec3& diffuseReflection, float diffuseRougness, const glm::vec3& specularReflection, float alphaX, float alphaY, float etaI, float etaT, ESceneobjectMask mask = ESceneobjectMask::NONE);
    CHostSceneobject(CNVDBMedium* medium, const glm::vec3& worldPos, const glm::vec3& orientation, const glm::vec3& scaling, ESceneobjectMask mask = ESceneobjectMask::NONE);
    CHostSceneobject(CMesh* mesh, CMaterial* material, const glm::vec3& worldPos, const glm::vec3& orientation, const glm::vec3& scaling, ESceneobjectMask mask = ESceneobjectMask::NONE);
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
    const SAABB& worldAABB() const;
    SAABB modelAABB() const;
    ESceneobjectMask mask() const;
    const glm::mat4x3& worldToModel() const;
    const CMesh* mesh() const;
    const glm::vec3& orientation() const;
  private:
    std::shared_ptr<CShape> m_shape;
    CMesh* m_mesh;
    CMaterial* m_material;
    std::shared_ptr<CMediumInstance> m_medium;
    ESceneobjectFlag m_flag;
    float m_absorption;
    glm::mat4x3 m_modelToWorld;
    glm::mat4x3 m_worldToModel;
    SAABB m_aabb;
    ESceneobjectMask m_mask;
    glm::vec3 m_orientation;

    OptixTraversableHandle m_traversableHandle;
    CUdeviceptr m_deviceGasBuffer;

    CSceneobjectConnection m_hostDeviceConnection;

    static std::shared_ptr<CShape> getShape(EShape shape, const glm::vec3& worldPos, float radius, const glm::vec3& normal);
    OptixProgramGroup getOptixProgramGroup() const;
    void getTransform(float* outMatrix) const;
    H_CALLABLE static glm::mat4 getModelToWorldTransform(const glm::vec3& worldPos, const glm::vec3& orientation, const glm::vec3& scaling);
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

  inline const SAABB& CHostSceneobject::worldAABB() const {
    return m_aabb;
  }

  inline ESceneobjectMask CHostSceneobject::mask() const {
    return m_mask;
  }

  inline const glm::mat4x3& CHostSceneobject::worldToModel() const {
    return m_worldToModel;
  }

  inline const CMesh* CHostSceneobject::mesh() const {
    return m_mesh;
  }

  inline const glm::vec3& CHostSceneobject::orientation() const {
    return m_orientation;
  }

}
#endif // !SCENEOBJECT_HPP
