#include "camera/camera.hpp"

#include <glm/gtc/matrix_transform.hpp>

namespace rt {
  CCamera::CCamera(uint16_t sensorWidth, uint16_t sensorHeight, float fov, const glm::vec3& pos, const glm::vec3& lookAt, const glm::vec3& up) :
    m_pixelSize(1e-5),
    m_sensorWidth(sensorWidth),
    m_sensorHeight(sensorHeight),
    m_fov(glm::radians(fov)),
    m_position(pos),
    m_lookAt(normalizeLookAt(pos, lookAt)),
    m_up(up),
    m_nearPlaneDistance(CCamera::getNearPlaneDistance(sensorWidth, m_fov, m_pixelSize)),
    m_worldToView(glm::lookAt(m_position, lookAt, up)),
    m_viewToWorld(glm::inverse(m_worldToView)){

  }

  float CCamera::getNearPlaneDistance(uint16_t sensorWidth, float fov, float pixelSize) {
    float sensorWidthInMeters = (float)sensorWidth * pixelSize;
    return sensorWidthInMeters / (2.0f * glm::tan(fov / 2.0f));
  }

  void CCamera::updatePosition(const glm::vec3& pos) {
    glm::vec3 diff = pos - m_position;
    m_position = pos;
    m_lookAt += diff;
    m_worldToView = glm::lookAt(m_position, m_lookAt, m_up);
    m_viewToWorld = glm::inverse(m_worldToView);
  }

  void CCamera::updateLookAt(const glm::vec3& lookAt) {
    m_lookAt = lookAt;
    m_worldToView = glm::lookAt(m_position, m_lookAt, m_up);
    m_viewToWorld = glm::inverse(m_worldToView);
  }

  glm::vec3 CCamera::normalizeLookAt(const glm::vec3& camPos, const glm::vec3& lookAt) {
    glm::vec3 lookDir = glm::normalize(lookAt - camPos);
    return camPos + lookDir;
  }

}