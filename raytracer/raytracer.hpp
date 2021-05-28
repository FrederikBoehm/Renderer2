#ifndef RAYTRACER_H
#define RAYTRACER_H

#include <cstdint>
#include <vector>

#include "scene/scene.hpp"
#include "camera/camera.hpp"

namespace rt {
	class Raytracer {
  public:
    struct Frame {
      uint16_t width;
      uint16_t height;
      std::vector<float> data;
    };

    Raytracer();

    Frame renderFrame();
	private:
    Scene m_scene;
    Camera m_camera;
	};
}

#endif // !RAYTRACER_H
