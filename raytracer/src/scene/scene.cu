#include "scene/scene.hpp"
#include "intersect/surface_interaction.hpp"

namespace rt {
  //CDeviceScene::CDeviceScene() :
  //  m_numSceneobjects(0),
  //  m_sceneobjects(nullptr) {

  //}

  //SSurfaceInteraction CDeviceScene::intersect(const Ray& ray) const {
  //  SSurfaceInteraction closestInteraction;
  //  closestInteraction.surfaceAlbedo = glm::vec3(0.0f);
  //  closestInteraction.hitInformation.t = 1e+10;
  //  for (uint8_t i = 0; i < m_numSceneobjects; ++i) {
  //    SSurfaceInteraction currentInteraction = m_sceneobjects[i].intersect(ray);
  //    if (currentInteraction.hitInformation.hit && currentInteraction.hitInformation.t < closestInteraction.hitInformation.t) {
  //      closestInteraction = currentInteraction;
  //    }
  //  }
  //  return closestInteraction;
  //}

  CHostScene::CHostScene() :
    m_hostDeviceConnection(this) {

  }

  void CHostScene::addSceneobject(CHostSceneobject&& sceneobject) {
    m_sceneobjects.push_back(std::move(sceneobject));
  }
  //const CDeviceScene& CHostScene::allocateDeviceMemory() {
  //  // TODO: insert return statement here
  //  //cudaMalloc(&m_deviceScene, sizeof(CDeviceScene));
  //  //cudaMalloc(&m_deviceSceneobjects, m_sceneobjects.size() * sizeof(CSceneobject));
  //}

  CSceneConnection::CSceneConnection(CHostScene* hostScene) :
    m_hostScene(hostScene) {

  }

  void CSceneConnection::allocateDeviceMemory() {
    cudaError_t error1 = cudaMalloc(&m_deviceScene, sizeof(CDeviceScene));
    cudaError_t error2 = cudaMalloc(&m_deviceSceneobjects, m_hostScene->m_sceneobjects.size() * sizeof(CDeviceSceneobject));

    for (auto& sceneObject : m_hostScene->m_sceneobjects) {
      sceneObject.allocateDeviceMemory();
    }
  }

  void CSceneConnection::copyToDevice() {
    CDeviceScene deviceScene;
    deviceScene.m_numSceneobjects = m_hostScene->m_sceneobjects.size();
    deviceScene.m_sceneobjects = m_deviceSceneobjects;
    cudaError_t error1 = cudaMemcpy(m_deviceScene, &deviceScene, sizeof(CDeviceScene), cudaMemcpyHostToDevice);
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
}