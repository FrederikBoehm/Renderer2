#ifndef FILTER_HPP
#define FILTER_HPP

#include "sampling/sampler.hpp"
#include "scene/device_scene.hpp"

struct SConfig;

namespace filter {
  class COpenvdbBackend;
  struct SFilterLaunchParams;
  struct SFilteredData;
  class CFilter {
  public:
    CFilter(const SConfig& config);
    ~CFilter();
    void runFiltering() const;
    void initDeviceData() const;

  private:
    COpenvdbBackend* m_backend;
    SFilterLaunchParams* m_deviceLaunchParams;
    rt::CSampler* m_deviceSampler;
    rt::CDeviceScene* m_deviceScene;
    SFilteredData* m_deviceFilterData;
    uint32_t m_samplesPerVoxel;
    bool m_debug;
    uint32_t m_debugSamples;
    float m_sigma_t;
    uint32_t m_estimationIterations;
    
    void initOptix(const SConfig& config);
    void allocateDeviceMemory();
    void copyToDevice();
    void freeDeviceMemory();
  };
}
#endif