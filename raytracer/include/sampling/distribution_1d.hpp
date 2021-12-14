#ifndef DISTRIBUTION_1D_HPP
#define DISTRIBUTION_1D_HPP
#include <vector>
#include <utility/qualifiers.hpp>
#include "sampling/sampler.hpp"
#include "utility/functions.hpp"

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
    DH_CALLABLE float pdf(float pos) const;
    DH_CALLABLE float integral() const;
    DH_CALLABLE float func(size_t pos) const;
    DH_CALLABLE float funcInt() const;

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

  inline CDistribution1D::CDistribution1D() :
    m_func(nullptr),
    m_cdf(nullptr),
    m_deviceResource(nullptr) {

  }

  inline float CDistribution1D::sampleContinuous(CSampler& sampler, float* pdf, size_t* off) const {
    float u = sampler.uniformSample01();

    auto& predicate = [&](int index) { return m_cdf[index] <= u; };
    //auto& predicate = [&](int index) { return !(u <= m_cdf[index]); };
    int offset = findInterval(m_nCdf, predicate);

    if (off) {
      *off = offset;
    }

    float du = u - m_cdf[offset];
    if ((m_cdf[offset + 1] - m_cdf[offset]) > 0) {
      du /= (m_cdf[offset + 1] - m_cdf[offset]);
    }

    if (pdf) {
      *pdf = m_func[offset] / m_funcInt;
    }

    return (offset + du) / count();
  }

  inline size_t CDistribution1D::sampleDiscrete(CSampler& sampler, float* pdf, float* uRemapped) const {
    float u = sampler.uniformSample01();

    auto& predicate = [&](int index) { return m_cdf[index] <= u; };
    int offset = findInterval(m_nCdf, predicate);

    if (pdf) {
      *pdf = m_func[offset] / (m_funcInt * count());
    }

    if (uRemapped) {
      *uRemapped = (u - m_cdf[offset]) / (m_cdf[offset + 1] - m_cdf[offset]);
    }

    return offset;
  }

  inline float CDistribution1D::discretePdf(size_t index) const {
    float* func = m_func;
    float* cdf = m_cdf;
    return m_func[index] / (m_funcInt * count());
  }

  inline float CDistribution1D::pdf(float pos) const {
    size_t lowerIndex = pos * m_nFunc;
    size_t upperIndex = glm::ceil(pos * m_nFunc);
    float interpolation = pos * m_nFunc - lowerIndex;
    return (m_func[lowerIndex] * (1 - interpolation) + m_func[upperIndex] * interpolation) / m_funcInt;
  }

  DH_CALLABLE inline size_t CDistribution1D::count() const {
    return m_nFunc;
  }

  DH_CALLABLE inline float CDistribution1D::integral() const {
    return m_integral;
  }

  DH_CALLABLE inline float CDistribution1D::func(size_t pos) const {
    return m_func[pos];
  }

  DH_CALLABLE inline float CDistribution1D::funcInt() const {
    return m_funcInt;
  }
}

#endif // !DISTRIBUTION_1D_HPP

