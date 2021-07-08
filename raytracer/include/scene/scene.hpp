#ifndef SCENE_HPP
#define SCENE_HPP

#include <vector>
#include "utility/qualifiers.hpp"
#include "scene/sceneobject.hpp"

namespace rt {
  class CHostScene;

  class CDeviceScene {
    friend class CSceneConnection;
  public:
    //DH_CALLABLE CDeviceScene();
    D_CALLABLE SSurfaceInteraction intersect(const Ray& ray) const;

  private:
    uint16_t m_numSceneobjects;
    CDeviceSceneobject* m_sceneobjects;
  };


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

  };

  class CHostScene { 
    friend class CSceneConnection;
  public:
    H_CALLABLE CHostScene();
    H_CALLABLE const std::vector<CHostSceneobject>& sceneobjects() const;
    H_CALLABLE void addSceneobject(CHostSceneobject&& sceneobject);

    H_CALLABLE void allocateDeviceMemory();
    H_CALLABLE void copyToDevice();
    H_CALLABLE void freeDeviceMemory();
    H_CALLABLE CDeviceScene* deviceScene();
  private:
    std::vector<CHostSceneobject> m_sceneobjects;
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

  //inline SSurfaceInteraction CDeviceScene::intersect(const Ray& ray) const {
  //  SSurfaceInteraction closestInteraction;
  //  closestInteraction.hitInformation.t = 1e+10;
  //  CDeviceSceneobject* sceneobjects = m_sceneobjects;
  //  for (uint8_t i = 0; i < m_numSceneobjects; ++i) {
  //    SSurfaceInteraction currentInteraction = m_sceneobjects[i].intersect(ray);
  //    if (currentInteraction.hitInformation.hit && currentInteraction.hitInformation.t < closestInteraction.hitInformation.t) {
  //      closestInteraction = currentInteraction;
  //    }
  //  }
  //  return closestInteraction;
  //}
}

#endif