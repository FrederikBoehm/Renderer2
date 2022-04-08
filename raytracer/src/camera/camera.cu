#include "camera/camera.hpp"

#include <glm/gtc/matrix_transform.hpp>

namespace rt {
  CCamera::CCamera():
    m_pixelSize(0.f),
    m_sensorWidth(0),
    m_sensorHeight(0),
    m_fov(0.f),
    m_position(0.f),
    m_lookAt(0.f),
    m_up(0.f),
    m_nearPlaneDistance(0.f),
    m_worldToView(1.f),
    m_viewToWorld(1.f) {

  }

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
    m_viewToWorld(glm::inverse(glm::mat4(m_worldToView))){
    glm::vec3 test1 = m_worldToView * glm::vec4(0.f, 0.f, m_nearPlaneDistance, 1.f);
    glm::vec3 test2 = m_worldToView * glm::vec4(0.f, 0.f, m_nearPlaneDistance, 0.f);
    glm::vec3 test3 = m_viewToWorld * glm::vec4(0.f, 0.f, m_nearPlaneDistance, 1.f);

    glm::vec3 nearPlaneOrigin = m_position + m_nearPlaneDistance * glm::normalize(m_lookAt - m_position);
    glm::vec3 nearPlaneOriginView = m_worldToView * glm::vec4(nearPlaneOrigin, 1.f);
    glm::vec3 test4 = m_viewToWorld * glm::vec4(0.f, 0.f, m_nearPlaneDistance, 0.f);

    glm::vec3 g = glm::normalize(m_lookAt - m_position);
    glm::vec3 x = glm::normalize(glm::cross(g, m_up));
    glm::vec3 pixelX = m_position + m_pixelSize * x;
    glm::vec3 pixelXView = m_worldToView * glm::vec4(pixelX, 1.f);

    glm::vec4 p0(-374.48406982421875f, 3.033600091934204f, -431.85845947265625f, 1.0f);
    glm::vec4 p1(354.92767333984375f, 3.033600091934204f, -431.85845947265625f, 1.0f);
    glm::vec4 p2(354.92767333984375f, 3.033600091934204f, 274.0238342285156f, 1.f);
    glm::vec4 p3(-374.48406982421875f, 3.033600091934204f, 274.0238342285156f, 1.f);
    glm::vec4 p4(-374.48406982421875f, 944.2099609375f, -431.85845947265625f, 1.f);
    glm::vec4 p5(354.92767333984375f, 944.2099609375f, -431.85845947265625f, 1.f);
    glm::vec4 p6(354.92767333984375f, 944.2099609375f, 274.0238342285156, 1.f);
    glm::vec4 p7(-374.48406982421875, 944.2099609375, 274.0238342285156, 1.f);

    glm::vec3 tp0 = m_worldToView * p0;
    glm::vec3 tp1 = m_worldToView * p1;
    glm::vec3 tp2 = m_worldToView * p2;
    glm::vec3 tp3 = m_worldToView * p3;
    glm::vec3 tp4 = m_worldToView * p4;
    glm::vec3 tp5 = m_worldToView * p5;
    glm::vec3 tp6 = m_worldToView * p6;
    glm::vec3 tp7 = m_worldToView * p7;

    glm::vec3 maxWorld = p0;
    glm::vec3 minWorld = p0;
    maxWorld = glm::max(maxWorld, glm::vec3(p0));
    minWorld = glm::min(minWorld, glm::vec3(p0));
    maxWorld = glm::max(maxWorld, glm::vec3(p1));
    minWorld = glm::min(minWorld, glm::vec3(p1));
    maxWorld = glm::max(maxWorld, glm::vec3(p2));
    minWorld = glm::min(minWorld, glm::vec3(p2));
    maxWorld = glm::max(maxWorld, glm::vec3(p3));
    minWorld = glm::min(minWorld, glm::vec3(p3));
    maxWorld = glm::max(maxWorld, glm::vec3(p4));
    minWorld = glm::min(minWorld, glm::vec3(p4));
    maxWorld = glm::max(maxWorld, glm::vec3(p5));
    minWorld = glm::min(minWorld, glm::vec3(p5));
    maxWorld = glm::max(maxWorld, glm::vec3(p6));
    minWorld = glm::min(minWorld, glm::vec3(p6));
    maxWorld = glm::max(maxWorld, glm::vec3(p7));
    minWorld = glm::min(minWorld, glm::vec3(p7));
    float l1 = glm::length(p0 - p1);

    glm::vec3 maxView = tp0;
    glm::vec3 minView = tp0;
    maxView = glm::max(maxView, glm::vec3(tp0));
    minView = glm::min(minView, glm::vec3(tp0));
    maxView = glm::max(maxView, glm::vec3(tp1));
    minView = glm::min(minView, glm::vec3(tp1));
    maxView = glm::max(maxView, glm::vec3(tp2));
    minView = glm::min(minView, glm::vec3(tp2));
    maxView = glm::max(maxView, glm::vec3(tp3));
    minView = glm::min(minView, glm::vec3(tp3));
    maxView = glm::max(maxView, glm::vec3(tp4));
    minView = glm::min(minView, glm::vec3(tp4));
    maxView = glm::max(maxView, glm::vec3(tp5));
    minView = glm::min(minView, glm::vec3(tp5));
    maxView = glm::max(maxView, glm::vec3(tp6));
    minView = glm::min(minView, glm::vec3(tp6));
    maxView = glm::max(maxView, glm::vec3(tp7));
    minView = glm::min(minView, glm::vec3(tp7));
    float l2 = glm::length(tp0 - tp1);


    glm::vec3 u = glm::normalize(m_viewToWorld * glm::vec4(1.f, 0.f, 0.f, 0.f));
    glm::vec3 v = glm::normalize(m_viewToWorld * glm::vec4(0.f, 1.f, 0.f, 0.f));
    glm::vec3 w = glm::normalize(m_viewToWorld * glm::vec4(0.f, 0.f, 1.f, 0.f));

    glm::vec3 adjustedNearPlaneDistance = m_worldToView * glm::vec4(m_position - w * m_nearPlaneDistance, 1.f);
    //m_nearPlaneDistance = adjustedNearPlaneDistance.z;
    glm::vec3 adjustedPixelSizeX = m_worldToView * glm::vec4(m_position + u * m_pixelSize, 1.f);
    glm::vec3 adjustedPixelSizeY = m_worldToView * glm::vec4(m_position + v * m_pixelSize, 1.f);

  }

  float CCamera::getNearPlaneDistance(uint16_t sensorWidth, float fov, float pixelSize) {
    float sensorWidthInMeters = (float)sensorWidth * pixelSize;
    return sensorWidthInMeters / (2.0f * glm::tan(fov / 2.0f));
  }

  void CCamera::updatePosition(const glm::vec3& pos) {
    glm::vec3 diff = pos - m_position;
    m_position = pos;
    m_lookAt += diff;
    glm::mat4 worldToView = glm::lookAt(m_position, m_lookAt, m_up);
    m_worldToView = worldToView;
    m_viewToWorld = glm::inverse(worldToView);
  }

  void CCamera::updateLookAt(const glm::vec3& lookAt) {
    m_lookAt = lookAt;
    glm::mat4 worldToView = glm::lookAt(m_position, m_lookAt, m_up);
    m_worldToView = worldToView;
    m_viewToWorld = glm::inverse(worldToView);
  }

  glm::vec3 CCamera::normalizeLookAt(const glm::vec3& camPos, const glm::vec3& lookAt) {
    glm::vec3 lookDir = glm::normalize(lookAt - camPos);
    return camPos + lookDir;
  }

}