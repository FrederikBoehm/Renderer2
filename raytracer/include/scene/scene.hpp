#ifndef SCENE_HPP
#define SCENE_HPP

#include <vector>
#include "utility/qualifiers.hpp"
#include "scene/sceneobject.hpp"
#include "device_scene.hpp"
#include "filtering/openvdb_backend.hpp"
#include "scene/sceneobject_mask.hpp"
#include <tuple>

namespace rt {
  


  class CSceneConnection {
  public:
    H_CALLABLE CSceneConnection(CHostScene* hostScene);
    H_CALLABLE CSceneConnection(CSceneConnection&& connection, CHostScene* hostScene);
    H_CALLABLE CSceneConnection& operator=(CSceneConnection&& connection);
    H_CALLABLE void allocateDeviceMemory();
    H_CALLABLE void copyToDevice();
    H_CALLABLE void freeDeviceMemory();
    H_CALLABLE CDeviceScene* deviceScene();
    H_CALLABLE void setHostScene(CHostScene* hostScene);
  private:
    CHostScene* m_hostScene;
    CDeviceScene* m_deviceScene;
    
    CDeviceSceneobject* m_deviceSceneobjects;
    CDeviceSceneobject* m_deviceLights;
    CDistribution1D* m_deviceLightDist;
    CEnvironmentMap* m_deviceEnvMap;
  };

  class CHostScene { 
    friend class CSceneConnection;
  public:
    H_CALLABLE CHostScene();
    H_CALLABLE ~CHostScene();
    H_CALLABLE CHostScene(CHostScene&& scene);
    H_CALLABLE CHostScene& operator=(CHostScene&& scene);
    H_CALLABLE const std::vector<CHostSceneobject>& sceneobjects() const;
    H_CALLABLE void addSceneobject(CHostSceneobject&& sceneobject);
    H_CALLABLE void addSceneobjectsFromAssimp(const std::string& assetsBasePath, const std::string& meshFileName, const glm::vec3& worldPos, const glm::vec3& normal, const glm::vec3& scaling, ESceneobjectMask mask = ESceneobjectMask::NONE);
    H_CALLABLE void addLightsource(CHostSceneobject&& lightsource);
    H_CALLABLE void setEnvironmentMap(CEnvironmentMap&& envMap);

    H_CALLABLE void allocateDeviceMemory();
    H_CALLABLE void copyToDevice();
    H_CALLABLE void freeDeviceMemory();
    H_CALLABLE CDeviceScene* deviceScene();

    H_CALLABLE void buildOptixAccel();
    H_CALLABLE std::vector<SRecord<const CDeviceSceneobject*>> getSBTHitRecords() const;
    H_CALLABLE std::tuple<std::vector<SAABB>, std::vector<SAABB>, glm::mat4x3, std::string> getObjectBBs(ESceneobjectMask mask = ESceneobjectMask::ALL) const; // returns modelBB and worldBB of first object with mask
  private:
    std::vector<CHostSceneobject> m_sceneobjects;
    std::vector<CHostSceneobject> m_lights;
    CDistribution1D* m_lightDist;
    CEnvironmentMap* m_envMap;
    OptixTraversableHandle m_traversableHandle;
    CUdeviceptr m_deviceIasBuffer;
    CUdeviceptr m_deviceInstances;

    CSceneConnection m_hostDeviceConnection;
	};


  inline const std::vector<CHostSceneobject>& CHostScene::sceneobjects() const {
    return m_sceneobjects;
  }

  inline void CHostScene::allocateDeviceMemory() {
    m_hostDeviceConnection.allocateDeviceMemory();
  }

  inline void CHostScene::copyToDevice() {
    m_hostDeviceConnection.copyToDevice();
  }

  inline void CHostScene::freeDeviceMemory() {
    m_hostDeviceConnection.freeDeviceMemory();
  }

  inline CDeviceScene* CHostScene::deviceScene() {
    return m_hostDeviceConnection.deviceScene();
  }

  inline CDeviceScene* CSceneConnection::deviceScene() {
    return m_deviceScene;
  }

  inline void CSceneConnection::setHostScene(CHostScene* hostScene) {
    m_hostScene = hostScene;
  }

}

#endif