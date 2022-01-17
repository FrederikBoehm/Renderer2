#ifndef CONFIG_LOADER_HPP
#define CONFIG_LOADER_HPP
#include "utility/qualifiers.hpp"
#include "scene/scene.hpp"
#include "camera/camera.hpp"

namespace rt {
  struct SConfig {
    uint16_t frameWidth;
    uint16_t frameHeight;
    uint8_t channelsPerPixel;
    uint16_t samples;
    float gamma;
    CCamera camera;
    CHostScene scene;
  };

  class CConfigLoader {
  public:
    H_CALLABLE static SConfig loadConfig(const char* configPath);

  };
}
#endif // !CONFIG_LOADER_HPP
