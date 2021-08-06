#include "scene/scene.hpp"
#include "scene/surface_interaction.hpp"
#include "cuda_runtime.h"

namespace rt {

  CHostScene::CHostScene() :
    m_hostDeviceConnection(this) {

  }

  void CHostScene::addSceneobject(CHostSceneobject&& sceneobject) {
    m_sceneobjects.push_back(std::move(sceneobject));
    
  }

  CSceneConnection::CSceneConnection(CHostScene* hostScene) :
    m_hostScene(hostScene) {

  }

  void CSceneConnection::allocateDeviceMemory() {
    cudaMalloc(&m_deviceScene, sizeof(CDeviceScene));
    cudaMalloc(&m_deviceSceneobjects, m_hostScene->m_sceneobjects.size() * sizeof(CDeviceSceneobject));

    for (auto& sceneObject : m_hostScene->m_sceneobjects) {
      sceneObject.allocateDeviceMemory();
    }
  }

  void CSceneConnection::copyToDevice() {
    CDeviceScene deviceScene;
    deviceScene.m_numSceneobjects = m_hostScene->m_sceneobjects.size();
    deviceScene.m_sceneobjects = m_deviceSceneobjects;
    cudaMemcpy(m_deviceScene, &deviceScene, sizeof(CDeviceScene), cudaMemcpyHostToDevice);
    for (size_t i = 0; i < m_hostScene->m_sceneobjects.size(); ++i) {
      m_hostScene->m_sceneobjects[i].setDeviceSceneobject(&m_deviceSceneobjects[i]);
      m_hostScene->m_sceneobjects[i].copyToDevice();
    }
  }

  void CSceneConnection::freeDeviceMemory() {
    for (auto& sceneObject : m_hostScene->m_sceneobjects) {
      sceneObject.freeDeviceMemory();
    }

    cudaFree(m_deviceSceneobjects);
    cudaFree(m_deviceScene);
  }

  SSurfaceInteraction CDeviceScene::intersect(const Ray& ray) const {
    SSurfaceInteraction closestInteraction;
    closestInteraction.hitInformation.t = 1e+10;
    CDeviceSceneobject* sceneobjects = m_sceneobjects;
    for (uint8_t i = 0; i < m_numSceneobjects; ++i) {
      SSurfaceInteraction currentInteraction = m_sceneobjects[i].intersect(ray);
      if (currentInteraction.hitInformation.hit && currentInteraction.hitInformation.t < closestInteraction.hitInformation.t) {
        closestInteraction = currentInteraction;
      }
    }
    return closestInteraction;
  }
}