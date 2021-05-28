#include "raytracer.hpp"

namespace rt {

  Raytracer::Raytracer():
  m_camera(1280, 720, 90, glm::vec3(0.0f, 1.0f, 3.0f), glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f)) {

  }

  Raytracer::Frame Raytracer::renderFrame() {
    Frame i;
    return i;
  }

}
