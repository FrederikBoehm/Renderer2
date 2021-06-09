#include "scene/sceneobject.hpp"

#include "shapes/plane.hpp"
#include "shapes/sphere.hpp"

namespace rt {
  CShape* CHostSceneobject::getShape(EShape shape, const glm::vec3& worldPos, float radius, const glm::vec3& normal) {
    switch (shape) {
    case EShape::PLANE:
      return new Plane(worldPos, radius, normal); // TODO: Allocate from host
      break;
    case EShape::SPHERE:
      return new Sphere(worldPos, radius);
    }
  }

  CHostSceneobject::CHostSceneobject(EShape shape, const glm::vec3& worldPos, float radius, const glm::vec3& normal, const glm::vec3& albedo):
    m_shape(getShape(shape, worldPos, radius, normal)),
    m_material(CMaterial(albedo)),
    m_hostDeviceConnection(this) {
    
  }
  CSceneobjectConnection::CSceneobjectConnection(CHostSceneobject* hostSceneobject):
    m_hostSceneobject(hostSceneobject) {
  }
  void CSceneobjectConnection::allocateDeviceMemory() {
    switch (m_hostSceneobject->m_shape->shape()) {
    case EShape::PLANE:
      cudaMalloc(&m_deviceShape, sizeof(Plane));
      break;
    case EShape::SPHERE:
      cudaMalloc(&m_deviceShape, sizeof(Sphere));
      break;
    }
  }
  void CSceneobjectConnection::copyToDevice() {
    switch (m_hostSceneobject->m_shape->shape()) {
    case EShape::PLANE:
      cudaMemcpy(m_deviceShape, m_hostSceneobject->m_shape, sizeof(Plane), cudaMemcpyHostToDevice);
      break;
    case EShape::SPHERE:
      cudaMemcpy(m_deviceShape, m_hostSceneobject->m_shape, sizeof(Sphere), cudaMemcpyHostToDevice);
      break;
    }
    if (m_deviceSceneobject) {
      CDeviceSceneobject deviceSceneobject;
      deviceSceneobject.m_shape = m_deviceShape;
      deviceSceneobject.m_material = m_hostSceneobject->m_material;
      cudaMemcpy(m_deviceSceneobject, &deviceSceneobject, sizeof(CDeviceSceneobject), cudaMemcpyHostToDevice);
    }
  }

  void CSceneobjectConnection::freeDeviceMemory() {
    cudaFree(m_deviceShape);
  }
}