#ifndef CAMERA_HPP
#define CAMERA_HPP

#include <random>

#include <glm/glm.hpp>

#include "intersect/ray.hpp"

namespace rt {
  class Camera {
    
  public:
    /*
      Creates a Camera object
      sensorWidth: width of the sensor in pixels
      sensorHeight: height of the sensor in pixels
      fov: horizontal fov in degrees
      pos: world position of the camera
      lookAt: world position of the target that the cam looks to
      up: world direction of the up vector
    */
    Camera(uint16_t sensorWidth, uint16_t sensorHeight, float fov, const glm::vec3& pos, const glm::vec3& lookAt, const glm::vec3& up);

    /*
      Samples the specified pixel randomly. Returns a ray in world space
    */
    Ray samplePixel(uint16_t x, uint16_t y) const;

  private:

    static const float s_pixelSize;
    static std::random_device s_rd;
    static std::mt19937 s_gen;
    static std::uniform_real_distribution<> s_dis;

    // Width and height of sensor in pixels, in our scale one pixel has width and height of 1e-5 meters = 10 micrometers
    uint16_t m_sensorWidth;
    uint16_t m_sensorHeight;
    float m_fov; // Horizontal FOV in rad
    float m_nearPlaneDistance;

    glm::vec3 m_position; // World position of camera
    glm::mat4 m_worldToView;
    glm::mat4 m_viewToWorld;

    static float getNearPlaneDistance(uint16_t sensorWidth, float fov);

  };
}

#endif // !CAMERA_HPP
