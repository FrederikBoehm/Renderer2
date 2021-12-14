#include "camera/camera.hpp"

#include <glm/gtc/matrix_transform.hpp>

namespace rt {
  CCamera::CCamera(uint16_t sensorWidth, uint16_t sensorHeight, float fov, const glm::vec3& pos, const glm::vec3& lookAt, const glm::vec3& up) :
    m_pixelSize(1e-5),
    m_sensorWidth(sensorWidth),
    m_sensorHeight(sensorHeight),
    m_fov(glm::radians(fov)),
    m_position(pos),
    m_lookAt(lookAt),
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
    m_position = pos;
    m_worldToView = glm::lookAt(m_position, m_lookAt, m_up);
    m_viewToWorld = glm::inverse(m_worldToView);
  }

}