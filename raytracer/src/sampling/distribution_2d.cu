#include "sampling/distribution_2d.hpp"
#include "utility/debugging.hpp"


namespace rt {

  CDistribution2D::CDistribution2D(uint16_t width, uint16_t height, const std::vector<float>& data):
    m_isHostObject(true), m_nElements(width * height), m_width(width), m_height(height), m_deviceResource(nullptr) {
    m_marginal = new CDistribution1D(sumRows(data));
    m_rows = static_cast<CDistribution1D*>(operator new[](m_height * sizeof(CDistribution1D)));
    for (uint32_t i = 0; i < m_height; ++i) {
      new(&m_rows[i]) CDistribution1D(std::vector<float>(data.data() + i * m_width, data.data() + (i + 1) * m_width));
    }
  }

  CDistribution2D::CDistribution2D(CDistribution2D&& dist) :
    m_isHostObject(std::move(dist.m_isHostObject)),
    m_nElements(std::move(dist.m_nElements)),
    m_width(std::move(dist.m_width)),
    m_height(std::move(dist.m_height)),
    m_marginal(std::exchange(dist.m_marginal, nullptr)),
    m_rows(std::exchange(dist.m_rows, nullptr)),
    m_deviceResource(std::exchange(dist.m_deviceResource, nullptr)) {

  }

  CDistribution2D::~CDistribution2D() {
#ifndef __CUDA_ARCH__
    //for (int i = m_height - 1; i >= 0; --i) { // TODO: clean up objects
    //  m_rows[i].~CDistribution1D();
    //}
    if (m_isHostObject) {
      operator delete[](m_rows);
    }
#endif
  }

  CDistribution2D& CDistribution2D::operator=(CDistribution2D&& dist) {
    m_isHostObject = std::move(dist.m_isHostObject);
    m_nElements = std::move(dist.m_nElements);
    m_width = std::move(dist.m_width);
    m_height = std::move(dist.m_height);
    m_marginal = std::exchange(dist.m_marginal, nullptr);
    m_rows = std::exchange(dist.m_rows, nullptr);
    m_deviceResource = std::exchange(dist.m_deviceResource, nullptr);
    return *this;
  }

  void CDistribution2D::allocateDeviceMemory() {
    if (m_deviceResource) {
      freeDeviceMemory();
      delete m_deviceResource;
    }

    m_deviceResource = new SDistribution2D_DeviceResource;
    CUDA_ASSERT(cudaMalloc(&m_deviceResource->d_marginal, sizeof(CDistribution1D)));
    CUDA_ASSERT(cudaMalloc(&m_deviceResource->d_rows, sizeof(CDistribution1D) * m_height));
  }

  CDistribution2D CDistribution2D::copyToDevice() {
    if (m_deviceResource) {
      m_marginal->copyToDevice(m_deviceResource->d_marginal);
      for (size_t i = 0; i < m_height; ++i) {
        m_rows[i].copyToDevice(m_deviceResource->d_rows + i);
      }
      //cudaMemcpy(m_deviceResource->d_marginal, m_marginal, sizeof(CDistribution1D), cudaMemcpyHostToDevice);
      //cudaMemcpy(m_deviceResource->d_rows, m_rows, sizeof(CDistribution1D) * m_height, cudaMemcpyHostToDevice);
    }

    CDistribution2D deviceDist;
    deviceDist.m_isHostObject = false;
    deviceDist.m_nElements = m_nElements;
    deviceDist.m_width = m_width;
    deviceDist.m_height = m_height;
    deviceDist.m_marginal = m_deviceResource->d_marginal;
    deviceDist.m_rows = m_deviceResource->d_rows;
    return deviceDist;
  }

  void CDistribution2D::freeDeviceMemory() {
    CUDA_ASSERT(cudaFree(m_deviceResource->d_marginal));
    CUDA_ASSERT(cudaFree(m_deviceResource->d_rows));
  }
}