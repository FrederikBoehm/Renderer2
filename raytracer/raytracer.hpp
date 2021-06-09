#ifndef RAYTRACER_H
#define RAYTRACER_H

#include "cuda_runtime.h"

#include <cstdint>
#include <vector>

#include "scene/scene.hpp"
#include "camera/camera.hpp"

namespace rt {
  struct SFrame {
    uint16_t width;
    uint16_t height;
    uint8_t bpp;
    float* data;
  };

  struct SHostFrame {
    uint16_t width;
    uint16_t height;
    uint8_t bpp;
    std::vector<float> data;
  };

	class Raytracer {
  public:

    Raytracer();
    ~Raytracer();

    SHostFrame renderFrame();
	private:
    CHostScene m_scene;
    CCamera m_hostCamera;
    CCamera* m_deviceCamera;
    SFrame* m_deviceFrame;
    float* m_deviceFrameData;

    static glm::vec3 getSpherePosition(float sphereRadius, uint8_t index, uint8_t maxSpheres);

    void allocateDeviceMemory();
    void copyToDevice();
    void initDeviceData();
    void freeDeviceMemory();

    SHostFrame retrieveFrame() const;
	};

}

#endif // !RAYTRACER_H
