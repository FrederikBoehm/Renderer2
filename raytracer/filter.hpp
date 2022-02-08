#ifndef FILTER_HPP
#define FILTER_HPP

#include "sampling/sampler.hpp"
#include "scene/device_scene.hpp"

struct SConfig;

namespace filter {
  class COpenvdbBackend;
  struct SFilterLaunchParams;
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
    float* m_deviceFilterData;
    uint32_t m_samplesPerVoxel;
    
    void initOptix(const SConfig& config);
    void allocateDeviceMemory();
    void copyToDevice();
    void freeDeviceMemory();
  };
}
#endif