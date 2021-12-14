#ifndef SCENE_HPP
#define SCENE_HPP

#include <vector>
#include "utility/qualifiers.hpp"
#include "scene/sceneobject.hpp"
#include "device_scene.hpp"

namespace rt {
  


  class CSceneConnection {
  public:
    H_CALLABLE CSceneConnection(CHostScene* hostScene);
    H_CALLABLE void allocateDeviceMemory();
    H_CALLABLE void copyToDevice();
    H_CALLABLE void freeDeviceMemory();
    H_CALLABLE CDeviceScene* deviceScene();
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
    H_CALLABLE const std::vector<CHostSceneobject>& sceneobjects() const;
    H_CALLABLE void addSceneobject(CHostSceneobject&& sceneobject);
    H_CALLABLE void addLightsource(CHostSceneobject&& lightsource);
    H_CALLABLE void setEnvironmentMap(CEnvironmentMap&& envMap);

    H_CALLABLE void allocateDeviceMemory();
    H_CALLABLE void copyToDevice();
    H_CALLABLE void freeDeviceMemory();
    H_CALLABLE CDeviceScene* deviceScene();

    H_CALLABLE void buildOptixAccel();
    H_CALLABLE OptixTraversableHandle getOptixTraversableHandle();
    H_CALLABLE std::vector<SRecord<const CDeviceSceneobject*>> getSBTHitRecords() const;
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

}

#endif