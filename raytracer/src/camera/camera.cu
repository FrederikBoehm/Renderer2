#include "camera/camera.hpp"

#include <glm/gtc/matrix_transform.hpp>

namespace rt {
  //const float CCamera::s_pixelSize = 1e-5;

  CCamera::CCamera(uint16_t sensorWidth, uint16_t sensorHeight, float fov, const glm::vec3& pos, const glm::vec3& lookAt, const glm::vec3& up) :
    m_pixelSize(1e-5),
    m_sensorWidth(sensorWidth),
    m_sensorHeight(sensorHeight),
    m_fov(glm::radians(fov)),
    m_position(pos),
    m_nearPlaneDistance(CCamera::getNearPlaneDistance(sensorWidth, m_fov, m_pixelSize)),
    m_worldToView(glm::lookAt(m_position, lookAt, up)),
    m_viewToWorld(glm::inverse(m_worldToView)) {

  }

  float CCamera::getNearPlaneDistance(uint16_t sensorWidth, float fov, float pixelSize) {
    float sensorWidthInMeters = (float)sensorWidth * pixelSize;
    return sensorWidthInMeters / (2.0f * glm::tan(fov / 2.0f));
  }

  void CCamera::initCurandState() {
    curand_init(0, 0, 0, &m_curandState);
  }

  //Ray CCamera::samplePixel(uint16_t x, uint16_t y) {
  //  //float randomHorizontal = s_dis(s_gen);
  //  float randomHorizontal = curand_uniform(&m_curandState);
  //  float horizontal = (x - m_sensorWidth / 2 + randomHorizontal) * m_pixelSize;
  //  //float randomVertical = s_dis(s_gen);
  //  float randomVertical = curand_uniform(&m_curandState);
  //  float vertical = (y - m_sensorHeight / 2 + randomHorizontal) * m_pixelSize;
  //  float depth = -m_nearPlaneDistance;

  //  glm::vec4 rayDir = glm::vec4(glm::normalize(glm::vec3(horizontal, vertical, depth)), 0.0f);
  //  glm::vec4 rayOrigin = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
  //  Ray viewSpaceRay(rayOrigin, rayDir);
  //  return viewSpaceRay.transform(m_viewToWorld);
  //}
}