#ifndef FILTER_HPP
#define FILTER_HPP

#include "sampling/sampler.hpp"
#include "scene/device_scene.hpp"
#include <string>

struct SConfig;

namespace filter {
  class COpenvdbBackend;
  struct SFilterLaunchParams;
  struct SFilteredDataCompact;
  class CFilter {
  public:
    CFilter(const SConfig& config);
    ~CFilter();
    void runFiltering() const;

  private:
    COpenvdbBackend* m_backend;
    SFilterLaunchParams* m_deviceLaunchParams;
    rt::CSampler* m_deviceSampler;
    rt::CDeviceScene* m_deviceScene;
    SFilteredDataCompact* m_deviceFilterData;
    uint32_t m_samplesPerVoxel;
    bool m_debug;
    uint32_t m_debugSamples;
    float m_sigma_t;
    uint32_t m_estimationIterations;
    float m_alpha;
    bool m_clipRays;
    float m_voxelSize;
    uint8_t m_lods;
    std::string m_outDir;
    std::string m_filename;
    glm::vec3 m_orientation;
    
    void initOptix(const SConfig& config);
    void allocateDeviceMemory();
    void copyToDevice(SFilterLaunchParams& launchParams) const;
    void freeDeviceMemory();
    void initDeviceData() const;
  };
}
#endif