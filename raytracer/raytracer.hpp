#ifndef RAYTRACER_H
#define RAYTRACER_H

#include "cuda_runtime.h"

#include <cstdint>
#include <vector>

#include "scene/scene.hpp"
#include "camera/camera.hpp"
#include "../common/frame.hpp"
#include "../common/pressed_key.hpp"
#include <functional>

namespace rt {
  struct SDeviceFrame {
    uint16_t width;
    uint16_t height;
    uint8_t bpp;
    float* data;
    float* filtered;
    uint8_t* dataBytes;
  };

  class CSampler;
  struct SLaunchParams;

	class Raytracer {
  public:

    Raytracer(uint16_t frameWidth, uint16_t frameHeight);
    ~Raytracer();

    SFrame renderFrame(const std::function<bool()>& keyCallback);
    SFrame renderPreview();
    void updateCamera(EPressedKey pressedKeys, const glm::vec2& mouseMoveDir);
	private:
    uint16_t m_frameWidth;
    uint16_t m_frameHeight;
    uint8_t m_bpp;

    CHostScene m_scene;
    CCamera m_hostCamera;
    uint16_t m_numSamples;
    float m_tonemappingFactor;
    float m_gamma;
    CCamera* m_deviceCamera;
    SDeviceFrame* m_deviceFrame;
    float* m_deviceFrameData;
    float* m_deviceFilteredFrame;
    uint8_t* m_deviceFrameDataBytes;
    CSampler* m_deviceSampler;
    float* m_deviceAverage;
    float* m_deviceTonemappingValue;
    const uint16_t m_blockSize;
    SLaunchParams* m_deviceLaunchParams;

    static glm::vec3 getSpherePosition(float sphereRadius, uint8_t index, uint8_t maxSpheres);

    void allocateDeviceMemory();
    void copyToDevice();
    void initDeviceData();
    void freeDeviceMemory();
    void initOptix();

    SFrame retrieveFrame() const;
	};

}

#endif // !RAYTRACER_H
