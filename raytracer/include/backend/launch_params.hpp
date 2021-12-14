#include <cstdint>

namespace rt {
  class CDeviceScene;
  struct CCamera;
  class CSampler;
  struct SLaunchParams {
    uint16_t width;
    uint16_t height;
    uint8_t bpp;
    float* data;
    float* filtered;
    uint8_t* dataBytes;

    CDeviceScene* scene;
    CCamera* camera;
    CSampler* sampler;
    uint16_t numSamples;
  };
}