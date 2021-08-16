#include "scene/sceneobject.hpp"
#include <iostream>

#include "shapes/plane.hpp"
#include "shapes/sphere.hpp"

namespace rt {
  std::shared_ptr<CShape> CHostSceneobject::getShape(EShape shape, const glm::vec3& worldPos, float radius, const glm::vec3& normal) {
    switch (shape) {
    case EShape::PLANE:
      return std::make_shared<Plane>(worldPos, radius, normal);
      break;
    case EShape::SPHERE:
      return std::make_shared<Sphere>(worldPos, radius);
    }
  }

  CHostSceneobject::CHostSceneobject(EShape shape, const glm::vec3& worldPos, float radius, const glm::vec3& normal, const glm::vec3& le):
    m_shape(getShape(shape, worldPos, radius, normal)),
    m_material(CMaterial(le)),
    m_hostDeviceConnection(this) {
  }

  CHostSceneobject::CHostSceneobject(EShape shape, const glm::vec3& worldPos, float radius, const glm::vec3& normal, const glm::vec3& diffuseReflection, const glm::vec3& specularReflection, float shininess):
    m_shape(getShape(shape, worldPos, radius, normal)),
    m_material(CMaterial(COrenNayarBRDF(diffuseReflection, 0.0f), CBlinnPhongBRDF(specularReflection, shininess))),
    m_hostDeviceConnection(this) {

  }

  CHostSceneobject::CHostSceneobject(EShape shape, const glm::vec3& worldPos, float radius, const glm::vec3& normal, const glm::vec3& diffuseReflection, float diffuseRougness, const glm::vec3& specularReflection, float alphaX, float alphaY, float etaI, float etaT) :
    m_shape(getShape(shape, worldPos, radius, normal)),
    m_material(CMaterial(COrenNayarBRDF(diffuseReflection, diffuseRougness), CMicrofacetBRDF(specularReflection, alphaX, alphaY, etaI, etaT))),
    m_hostDeviceConnection(this) {

  }

  CHostSceneobject::CHostSceneobject(CHostSceneobject&& sceneobject) :
    m_shape(std::move(sceneobject.m_shape)),
    m_material(std::move(sceneobject.m_material)),
    m_hostDeviceConnection(this) {
  }

  CSceneobjectConnection::CSceneobjectConnection(CHostSceneobject* hostSceneobject):
    m_hostSceneobject(hostSceneobject) {
  }

  CSceneobjectConnection::CSceneobjectConnection(const CSceneobjectConnection&& connection) :
    m_hostSceneobject(std::move(connection.m_hostSceneobject)) {
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
      cudaMemcpy(m_deviceShape, m_hostSceneobject->m_shape.get(), sizeof(Plane), cudaMemcpyHostToDevice);
      break;
    case EShape::SPHERE:
      cudaMemcpy(m_deviceShape, m_hostSceneobject->m_shape.get(), sizeof(Sphere), cudaMemcpyHostToDevice);
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

  SSurfaceInteraction CDeviceSceneobject::intersect(const Ray& ray) {
    SSurfaceInteraction si;
    switch (m_shape->shape()) {
    case EShape::PLANE:
      si.hitInformation = ((Plane*)m_shape)->intersect(ray);
      break;
    case EShape::SPHERE:
      si.hitInformation = ((Sphere*)m_shape)->intersect(ray);
      break;
    }
    si.material = m_material;
    return si;
  }
}