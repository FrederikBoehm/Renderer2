#ifndef CONFIG_LOADER_HPP
#define CONFIG_LOADER_HPP
#include "utility/qualifiers.hpp"
#include "scene/scene.hpp"
#include "camera/camera.hpp"
#include <memory>

struct SFilteringConfig {
  bool filter;
  glm::vec3 voxelSize;
  uint32_t samplesPerVoxel;
  bool debug;
  uint32_t debugSamples;
  float sigmaT;
  uint32_t estimationIterations;
  float alpha;
  bool clipRays;
};

struct SConfig {
  uint16_t frameWidth;
  uint16_t frameHeight;
  uint8_t channelsPerPixel;
  uint16_t samples;
  float gamma;
  std::shared_ptr<rt::CCamera> camera;
  std::shared_ptr<rt::CHostScene> scene;
  SFilteringConfig filteringConfig;
};

class CConfigLoader {
public:
  H_CALLABLE static SConfig loadConfig(const char* configPath);

};
#endif // !CONFIG_LOADER_HPP
