#include "scene/sceneobject.hpp"
#include <iostream>

#include "shapes/plane.hpp"
#include "shapes/sphere.hpp"
#include "medium/medium.hpp"

namespace rt {
  std::shared_ptr<CShape> CHostSceneobject::getShape(EShape shape, const glm::vec3& worldPos, float radius, const glm::vec3& normal) {
    switch (shape) {
    case EShape::PLANE:
      return std::make_shared<Plane>(worldPos, radius, normal);
      break;
    case EShape::SPHERE:
      return std::make_shared<Sphere>(worldPos, radius, normal);
    }
  }

  CHostSceneobject::CHostSceneobject(EShape shape, const glm::vec3& worldPos, float radius, const glm::vec3& normal, const glm::vec3& le):
    m_shape(getShape(shape, worldPos, radius, normal)),
    m_material(nullptr),
    m_medium(nullptr),
    m_flag(ESceneobjectFlag::GEOMETRY),
    m_hostDeviceConnection(this) {
    m_material = std::make_shared<CMaterial>(CMaterial(le));
  }

  CHostSceneobject::CHostSceneobject(EShape shape, const glm::vec3& worldPos, float radius, const glm::vec3& normal, const glm::vec3& diffuseReflection, float diffuseRougness, const glm::vec3& specularReflection, float alphaX, float alphaY, float etaI, float etaT) :
    m_shape(getShape(shape, worldPos, radius, normal)),
    m_material(nullptr),
    m_medium(nullptr),
    m_flag(ESceneobjectFlag::GEOMETRY),
    m_hostDeviceConnection(this) {
    m_material = std::make_shared<CMaterial>(CMaterial(COrenNayarBRDF(diffuseReflection, diffuseRougness), CMicrofacetBRDF(specularReflection, alphaX, alphaY, etaI, etaT)));
  }

  CHostSceneobject::CHostSceneobject(EShape shape, const glm::vec3& worldPos, float radius, const glm::vec3& normal, const glm::vec3& absorption, const glm::vec3&outScattering, float asymmetry):
    m_shape(getShape(shape, worldPos, radius, normal)),
    m_material(nullptr),
    m_medium(nullptr),
    m_flag(ESceneobjectFlag::VOLUME),
    m_hostDeviceConnection(this) {
    m_medium = std::make_shared<CMedium>(CMedium(absorption, outScattering, asymmetry));
  }

  CHostSceneobject::CHostSceneobject(CHostSceneobject&& sceneobject) :
    m_shape(std::move(sceneobject.m_shape)),
    m_material(std::move(sceneobject.m_material)),
    m_medium(std::move(sceneobject.m_medium)),
    m_flag(std::move(sceneobject.m_flag)),
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
    if (m_hostSceneobject->m_material) {
      cudaMalloc(&m_deviceMaterial, sizeof(CMaterial));
    }
    if (m_hostSceneobject->m_medium) {
      cudaMalloc(&m_deviceMedium, sizeof(CMedium));
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
    if (m_deviceMaterial) {
      cudaMemcpy(m_deviceMaterial, m_hostSceneobject->m_material.get(), sizeof(CMaterial), cudaMemcpyHostToDevice);
    }
    if (m_deviceMedium) {
      cudaMemcpy(m_deviceMedium, m_hostSceneobject->m_medium.get(), sizeof(CMedium), cudaMemcpyHostToDevice);
    }
    if (m_deviceSceneobject) {

      CDeviceSceneobject deviceSceneobject;
      deviceSceneobject.m_shape = m_deviceShape;
      deviceSceneobject.m_material = m_deviceMaterial;
      deviceSceneobject.m_medium = m_deviceMedium;
      deviceSceneobject.m_flag = m_hostSceneobject->m_flag;
      cudaMemcpy(m_deviceSceneobject, &deviceSceneobject, sizeof(CDeviceSceneobject), cudaMemcpyHostToDevice);
    }
  }

  void CSceneobjectConnection::freeDeviceMemory() {
    cudaFree(m_deviceShape);
    cudaFree(m_deviceMaterial);
    cudaFree(m_deviceMedium);
  }

  SInteraction CDeviceSceneobject::intersect(const CRay& ray) {
    SInteraction si;
    switch (m_shape->shape()) {
    case EShape::PLANE:
      si.hitInformation = ((Plane*)m_shape)->intersect(ray);
      break;
    case EShape::SPHERE:
      si.hitInformation = ((Sphere*)m_shape)->intersect(ray);
      break;
    }
    si.material = m_material;
    si.medium = m_medium;
    si.object = this;
    return si;
  }

  float CHostSceneobject::power() const {
    if (m_flag == ESceneobjectFlag::GEOMETRY) {
      glm::vec3 L = m_material->Le();
      switch (m_shape->shape()) {
      case EShape::PLANE:
        return (L.x + L.y + L.z) * ((Plane*)m_shape.get())->area();
      }
    }
    return 0.0f;
  }

  CShape* CDeviceSceneobject::shape() const {
    return m_shape;
  }

  float CDeviceSceneobject::power() const {
    if (m_flag == ESceneobjectFlag::GEOMETRY) {
      glm::vec3 L = m_material->Le();
      switch (m_shape->shape()) {
      case EShape::PLANE:
        return (L.x + L.y + L.z) * ((Plane*)m_shape)->area();
      }
    }
    return 0.0f;
  }
}