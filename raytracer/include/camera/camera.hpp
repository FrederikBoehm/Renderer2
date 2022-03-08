#ifndef CAMERA_HPP
#define CAMERA_HPP

#include <glm/glm.hpp>

#include "utility/qualifiers.hpp"

extern "C" __global__ void __raygen__pinhole();

namespace rt {
  class CCamera {
    friend class CPixelSampler;
    friend __global__ void ::__raygen__pinhole();
    
  public:
    H_CALLABLE CCamera();

    /*
      Creates a Camera object
      sensorWidth: width of the sensor in pixels
      sensorHeight: height of the sensor in pixels
      fov: horizontal fov in degrees
      pos: world position of the camera
      lookAt: world position of the target that the cam looks to
      up: world direction of the up vector
    */
    H_CALLABLE CCamera(uint16_t sensorWidth, uint16_t sensorHeight, float fov, const glm::vec3& pos, const glm::vec3& lookAt, const glm::vec3& up);

    DH_CALLABLE uint16_t sensorWidth() const;
    DH_CALLABLE uint16_t sensorHeight() const;
    DH_CALLABLE void updatePosition(const glm::vec3& pos);
    DH_CALLABLE void updateLookAt(const glm::vec3& lookAt);
    DH_CALLABLE const glm::vec3& position() const;
    DH_CALLABLE const glm::vec3& lookAt() const;
    DH_CALLABLE const glm::mat4x3& worldToView() const;
    DH_CALLABLE const glm::mat4x3& viewToWorld() const;
    DH_CALLABLE const glm::vec3& up() const;
  private:

    float m_pixelSize;

    // Width and height of sensor in pixels, in our scale one pixel has width and height of 1e-5 meters = 10 micrometers
    uint16_t m_sensorWidth;
    uint16_t m_sensorHeight;
    float m_fov; // Horizontal FOV in rad
    float m_nearPlaneDistance;

    glm::vec3 m_position; // World position of camera
    glm::vec3 m_lookAt;
    glm::vec3 m_up;
    glm::mat4x3 m_worldToView;
    glm::mat4x3 m_viewToWorld;

    H_CALLABLE static float getNearPlaneDistance(uint16_t sensorWidth, float fov, float pixelSize);
    H_CALLABLE static glm::vec3 normalizeLookAt(const glm::vec3& camPos, const glm::vec3& lookAt);

  };

  inline uint16_t CCamera::sensorWidth() const {
    return m_sensorWidth;
  }

  inline uint16_t CCamera::sensorHeight() const {
    return m_sensorHeight;
  }

  inline const glm::vec3& CCamera::position() const {
    return m_position;
  }

  inline const glm::vec3& CCamera::lookAt() const {
    return m_lookAt;
  }

  inline const glm::mat4x3& CCamera::worldToView() const {
    return m_worldToView;
  }

  inline const glm::mat4x3& CCamera::viewToWorld() const {
    return m_viewToWorld;
  }

  inline const glm::vec3& CCamera::up() const {
    return m_up;
  }

}

#endif // !CAMERA_HPP
