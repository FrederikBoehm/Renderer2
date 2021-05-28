#include <glm/gtc/matrix_transform.hpp>

#include "camera/camera.hpp"

namespace rt {
  const float Camera::s_pixelSize = 1e-5;
  std::random_device Camera::s_rd;
  std::mt19937 Camera::s_gen(s_rd());
  std::uniform_real_distribution<> Camera::s_dis;

  Camera::Camera(uint16_t sensorWidth, uint16_t sensorHeight, float fov, const glm::vec3& pos, const glm::vec3& lookAt, const glm::vec3& up) :
    m_sensorWidth(sensorWidth),
    m_sensorHeight(sensorHeight),
    m_fov(glm::radians(fov)),
    m_position(pos),
    m_nearPlaneDistance(Camera::getNearPlaneDistance(sensorWidth, m_fov)),
    m_worldToView(glm::lookAt(m_position, lookAt, up)),
    m_viewToWorld(glm::inverse(m_worldToView)){

  }

  float Camera::getNearPlaneDistance(uint16_t sensorWidth, float fov) {
    float sensorWidthInMeters = (float)sensorWidth * s_pixelSize;
    return sensorWidthInMeters / (2.0f * glm::tan(fov / 2.0f));
  }

  Ray Camera::samplePixel(uint16_t x, uint16_t y) const {
    float randomHorizontal = s_dis(s_gen);
    float horizontal = (x - m_sensorWidth / 2 + randomHorizontal) * s_pixelSize;
    float randomVertical = s_dis(s_gen);
    float vertical = (y - m_sensorHeight / 2 + randomHorizontal) * s_pixelSize;
    float depth = -m_nearPlaneDistance;

    glm::vec4 rayDir = glm::vec4(glm::normalize(glm::vec3(horizontal, vertical, depth)), 0.0f);
    glm::vec4 rayOrigin = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    Ray viewSpaceRay(rayOrigin, rayDir);
    return viewSpaceRay.transform(m_viewToWorld);
  }
}