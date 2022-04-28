#ifndef BACKEND_TYPES_HPP
#define BACKEND_TYPES_HPP
#include <optix/optix_types.h>
#include <cstdint>

namespace rt {
  template <typename T>
  struct SRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
  };  

  struct SEmptyData {}; 

  enum class EDebugMode {
    NONE,
    VISUALIZE_LODS,
    VISUALIZE_VOLUME_LOOKUPS
  };

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
    bool useBrickGrid;
    EDebugMode debugMode;
  };
}

#endif