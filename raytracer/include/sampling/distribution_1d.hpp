#ifndef DISTRIBUTION_1D_HPP
#define DISTRIBUTION_1D_HPP
#include <vector>
#include <utility/qualifiers.hpp>

namespace rt {
  class CSampler;

  struct SDistribution1D_DeviceResource {
    float* d_func;
    float* d_cdf;
  };

  class CDistribution1D {
  public:
    DH_CALLABLE CDistribution1D();
    H_CALLABLE CDistribution1D(std::vector<float>& f);
    H_CALLABLE ~CDistribution1D();

    DH_CALLABLE float sampleContinuous(CSampler& sampler, float* pdf, size_t* off = nullptr) const;
    DH_CALLABLE size_t sampleDiscrete(CSampler& sampler, float* pdf = nullptr, float* uRemapped = nullptr) const;
    DH_CALLABLE size_t count() const;
    DH_CALLABLE float discretePdf(size_t index) const;
    DH_CALLABLE float integral() const;

    // CUDA stuff
    H_CALLABLE void copyToDevice(CDistribution1D* dst);
    H_CALLABLE void freeDeviceMemory();

  private:
    size_t m_nFunc;
    float* m_func;
    size_t m_nCdf;
    float* m_cdf;
    float m_funcInt;
    float m_integral;

    SDistribution1D_DeviceResource* m_deviceResource;
  };

  DH_CALLABLE inline size_t CDistribution1D::count() const {
    return m_nFunc;
  }

  DH_CALLABLE inline float CDistribution1D::integral() const {
    return m_integral;
  }
}

#endif // !DISTRIBUTION_1D_HPP

