#ifndef SCENE_HPP
#define SCENE_HPP

#include <vector>
#include "utility/qualifiers.hpp"
#include "scene/sceneobject.hpp"

namespace rt {
  class CHostScene;
  class CDistribution1D;

  class CDeviceScene {
    friend class CSceneConnection;
  public:
    //DH_CALLABLE CDeviceScene();
    D_CALLABLE SSurfaceInteraction intersect(const Ray& ray) const;
    D_CALLABLE glm::vec3 sampleLightSources(CSampler& sampler, float* pdf) const;
    D_CALLABLE float lightSourcePdf(const SSurfaceInteraction& lightHit, const Ray& shadowRay) const;

  private:
    uint16_t m_numSceneobjects;
    CDeviceSceneobject* m_sceneobjects;
    uint16_t m_numLights;
    CDeviceSceneobject* m_lights; // For now lights are also sceneobjects
    CDistribution1D* m_lightDist;
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
    CDeviceSceneobject* m_deviceLights;
    CDistribution1D* m_deviceLightDist;
  };

  class CHostScene { 
    friend class CSceneConnection;
  public:
    H_CALLABLE CHostScene();
    H_CALLABLE const std::vector<CHostSceneobject>& sceneobjects() const;
    H_CALLABLE void addSceneobject(CHostSceneobject&& sceneobject);
    H_CALLABLE void addLightsource(CHostSceneobject&& lightsource);

    H_CALLABLE void allocateDeviceMemory();
    H_CALLABLE void copyToDevice();
    H_CALLABLE void freeDeviceMemory();
    H_CALLABLE CDeviceScene* deviceScene();
  private:
    std::vector<CHostSceneobject> m_sceneobjects;
    std::vector<CHostSceneobject> m_lights;
    CDistribution1D* m_lightDist;
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