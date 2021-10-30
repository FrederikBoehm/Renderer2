#include "scene/sceneobject.hpp"
#include <iostream>

#include "shapes/circle.hpp"
#include "shapes/sphere.hpp"
#include "medium/homogeneous_medium.hpp"
#include "shapes/rectangle.hpp"
#include "shapes/cuboid.hpp"
#include "medium/heterogenous_medium.hpp"

namespace rt {
  std::shared_ptr<CShape> CHostSceneobject::getShape(EShape shape, const glm::vec3& worldPos, float radius, const glm::vec3& normal) {
    switch (shape) {
    case EShape::CIRCLE:
      return std::make_shared<CCircle>(worldPos, radius, normal);
      break;
    case EShape::SPHERE:
      return std::make_shared<Sphere>(worldPos, radius, normal);
    }
  }

  CHostSceneobject::CHostSceneobject(const CShape* shape, const glm::vec3& le):
    m_shape(shape),
    m_material(nullptr),
    m_medium(nullptr),
    m_flag(ESceneobjectFlag::GEOMETRY),
    m_hostDeviceConnection(this) {
    m_material = std::make_shared<CMaterial>(CMaterial(le));
  }

  CHostSceneobject::CHostSceneobject(const CShape* shape, const glm::vec3& diffuseReflection, float diffuseRougness, const glm::vec3& specularReflection, float alphaX, float alphaY, float etaI, float etaT) :
    m_shape(shape),
    m_material(nullptr),
    m_medium(nullptr),
    m_flag(ESceneobjectFlag::GEOMETRY),
    m_hostDeviceConnection(this) {
    m_material = std::make_shared<CMaterial>(CMaterial(COrenNayarBRDF(diffuseReflection, diffuseRougness), CMicrofacetBRDF(specularReflection, alphaX, alphaY, etaI, etaT)));
  }

  CHostSceneobject::CHostSceneobject(const CShape* shape, CMedium* medium):
    m_shape(shape),
    m_material(nullptr),
    m_medium(medium),
    m_flag(ESceneobjectFlag::VOLUME),
    m_hostDeviceConnection(this) {
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
    case EShape::CIRCLE:
      cudaMalloc(&m_deviceShape, sizeof(CCircle));
      break;
    case EShape::SPHERE:
      cudaMalloc(&m_deviceShape, sizeof(Sphere));
      break;
    case EShape::RECTANGLE:
      cudaMalloc(&m_deviceShape, sizeof(CRectangle));
      break;
    case EShape::CUBOID:
      cudaMalloc(&m_deviceShape, sizeof(CCuboid));
      break;
    }
    if (m_hostSceneobject->m_material) {
      cudaMalloc(&m_deviceMaterial, sizeof(CMaterial));
    }
    if (m_hostSceneobject->m_medium) {
      switch (m_hostSceneobject->m_medium->type()) {
      case EMediumType::HOMOGENEOUS_MEDIUM:
        cudaMalloc(&m_deviceMedium, sizeof(CHomogeneousMedium));
        break;
      case EMediumType::HETEROGENOUS_MEDIUM:
        cudaMalloc(&m_deviceMedium, sizeof(CHeterogenousMedium));
        std::static_pointer_cast<CHeterogenousMedium>(m_hostSceneobject->m_medium)->allocateDeviceMemory();
        break;
      }
    }

  }
  void CSceneobjectConnection::copyToDevice() {
    switch (m_hostSceneobject->m_shape->shape()) {
    case EShape::CIRCLE:
      cudaMemcpy(m_deviceShape, m_hostSceneobject->m_shape.get(), sizeof(CCircle), cudaMemcpyHostToDevice);
      break;
    case EShape::SPHERE:
      cudaMemcpy(m_deviceShape, m_hostSceneobject->m_shape.get(), sizeof(Sphere), cudaMemcpyHostToDevice);
      break;
    case EShape::RECTANGLE:
      cudaMemcpy(m_deviceShape, m_hostSceneobject->m_shape.get(), sizeof(CRectangle), cudaMemcpyHostToDevice);
      break;
    case EShape::CUBOID:
      cudaMemcpy(m_deviceShape, m_hostSceneobject->m_shape.get(), sizeof(CCuboid), cudaMemcpyHostToDevice);
    }
    if (m_deviceMaterial) {
      cudaMemcpy(m_deviceMaterial, m_hostSceneobject->m_material.get(), sizeof(CMaterial), cudaMemcpyHostToDevice);
    }
    if (m_deviceMedium) {
      switch (m_hostSceneobject->m_medium->type()) {
      case EMediumType::HOMOGENEOUS_MEDIUM:
        cudaMemcpy(m_deviceMedium, m_hostSceneobject->m_medium.get(), sizeof(CHomogeneousMedium), cudaMemcpyHostToDevice);
        break;
      case EMediumType::HETEROGENOUS_MEDIUM:
        std::shared_ptr<CHeterogenousMedium> hetMedium = std::static_pointer_cast<CHeterogenousMedium>(m_hostSceneobject->m_medium);
        cudaMemcpy(m_deviceMedium, &hetMedium->copyToDevice(), sizeof(CHeterogenousMedium), cudaMemcpyHostToDevice);
        break;
      }
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
    case EShape::CIRCLE:
      si.hitInformation = ((CCircle*)m_shape)->intersect(ray);
      break;
    case EShape::SPHERE:
      si.hitInformation = ((Sphere*)m_shape)->intersect(ray);
      break;
    case EShape::RECTANGLE:
      si.hitInformation = ((CRectangle*)m_shape)->intersect(ray);
      break;
    case EShape::CUBOID:
      si.hitInformation = ((CCuboid*)m_shape)->intersect(ray);
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
      case EShape::CIRCLE:
        return (L.x + L.y + L.z) * ((CCircle*)m_shape.get())->area();
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
      case EShape::CIRCLE:
        return (L.x + L.y + L.z) * ((CCircle*)m_shape)->area();
      }
    }
    return 0.0f;
  }
}