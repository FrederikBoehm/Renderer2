#include  "sampling/distribution_1d.hpp"
#include "utility/functions.hpp"
#include "utility/debugging.hpp"

namespace rt {
  CDistribution1D::CDistribution1D(std::vector<float>& f) :
    m_func(nullptr),
    m_cdf(nullptr),
    m_deviceResource(nullptr),
    m_integral(0) {
    m_nFunc = f.size();
    m_func = new float[f.size()];
    memcpy(m_func, f.data(), f.size() * sizeof(float));

    m_nCdf = f.size() + 1;
    m_cdf = new float[m_nCdf];


    for (size_t i = 0; i < f.size(); ++i) {
      m_integral += f[i];
    }

    m_cdf[0] = 0;
    for (size_t i = 1; i < m_nCdf; ++i) {
      m_cdf[i] = m_cdf[i - 1] + m_func[i - 1] / m_nFunc;
    }

    m_funcInt = m_cdf[m_nFunc];
    if (m_funcInt == 0) {
      for (size_t i = 1; i < m_nCdf; ++i) {
        m_cdf[i] = (float)i / m_nFunc;
      }
    }
    else {
      for (size_t i = 1; i < m_nCdf; ++i) {
        m_cdf[i] /= m_funcInt;
      }
    }
  }

  CDistribution1D::CDistribution1D(CDistribution1D&& dist):
    m_func(std::exchange(dist.m_func, nullptr)),
    m_cdf(std::exchange(dist.m_cdf, nullptr)),
    m_deviceResource(std::exchange(dist.m_deviceResource, nullptr)),
    m_integral(std::move(dist.m_integral)) {
  }

  CDistribution1D::~CDistribution1D() {
#ifndef __CUDA_ARCH__
    if (m_func) {
      delete m_func;
    }
    if (m_cdf) {
      delete m_cdf;
    }
#endif
  }

  void CDistribution1D::copyToDevice(CDistribution1D* dst) {
    if (!m_deviceResource) {
      m_deviceResource = new SDistribution1D_DeviceResource();
      CUDA_ASSERT(cudaMalloc(&(m_deviceResource->d_func), sizeof(float) * m_nFunc));
      CUDA_ASSERT(cudaMalloc(&(m_deviceResource->d_cdf), sizeof(float) * m_nCdf));
    }

    CUDA_ASSERT(cudaMemcpy(m_deviceResource->d_func, m_func, sizeof(float) * m_nFunc, cudaMemcpyHostToDevice));
    CUDA_ASSERT(cudaMemcpy(m_deviceResource->d_cdf, m_cdf, sizeof(float) * m_nCdf, cudaMemcpyHostToDevice));

    CDistribution1D temp;
    temp.m_nFunc = m_nFunc;
    temp.m_func = m_deviceResource->d_func;
    temp.m_nCdf = m_nCdf;
    temp.m_cdf = m_deviceResource->d_cdf;
    temp.m_funcInt = m_funcInt;
    temp.m_deviceResource = nullptr;
    temp.m_integral = m_integral;

    CUDA_ASSERT(cudaMemcpy(dst, &temp, sizeof(CDistribution1D), cudaMemcpyHostToDevice));

    temp.m_func = nullptr;
    temp.m_cdf = nullptr;
  }

  void CDistribution1D::freeDeviceMemory() {
    if (m_deviceResource) {
      CUDA_ASSERT(cudaFree(m_deviceResource->d_func));
      CUDA_ASSERT(cudaFree(m_deviceResource->d_cdf));
      delete m_deviceResource;
      m_deviceResource = nullptr;
    }
  }


}