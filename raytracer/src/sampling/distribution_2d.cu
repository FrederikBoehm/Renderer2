#include "sampling/distribution_2d.hpp"
#include "sampling/distribution_1d.hpp"

namespace rt {
  CDistribution2D::CDistribution2D() : m_width(0), m_height(0), m_marginal(nullptr), m_rows(0), m_deviceResource(nullptr) {

  }

  CDistribution2D::CDistribution2D(uint16_t width, uint16_t height, const std::vector<float>& data):
    m_nElements(width * height), m_width(width), m_height(height), m_deviceResource(nullptr) {
    m_marginal = new CDistribution1D(sumRows(data));
    m_rows = static_cast<CDistribution1D*>(operator new[](m_height * sizeof(CDistribution1D)));
    for (uint32_t i = 0; i < m_height; ++i) {
      new(&m_rows[i]) CDistribution1D(std::vector<float>(data.data() + i * m_width, data.data() + (i + 1) * m_width));
    }
  }

  CDistribution2D::~CDistribution2D() {
#ifndef __CUDA_ARCH__
    //for (int i = m_height - 1; i >= 0; --i) { // TODO: clean up objects
    //  m_rows[i].~CDistribution1D();
    //}
    //operator delete[](m_rows);
#endif
  }

  glm::vec2 CDistribution2D::sampleContinuous(CSampler& sampler, float* pdf, size_t* off) const {
    float pdfs[2];
    size_t v;
    float y = m_marginal->sampleContinuous(sampler, &pdfs[1], &v);
    float x = m_rows[v].sampleContinuous(sampler, &pdfs[0]);
    *pdf = pdfs[0] * pdfs[1];
    return glm::vec2(x, y);
  }

  std::vector<float> CDistribution2D::sumRows(const std::vector<float>& data) const {
    std::vector<float> sums;
    sums.reserve(m_height);
    float sum = 0.0f;
    for (size_t i = 0; i < m_nElements; ++i) {
      sum += data[i];
      if (i % m_width == 0) {
        sums.push_back(sum);
        sum = 0.f;
      }
    }
    return sums;
  }

  float CDistribution2D::pdf(const glm::vec2& pos) const {
    uint16_t lowerRowIndex = pos.y * (m_height - 1);
    uint16_t upperRowIndex = lowerRowIndex + 1;

    uint16_t lowerColumnIndex = pos.x * (m_width - 1);
    uint16_t upperColumnIndex = lowerColumnIndex + 1;

    float lowerRowInterpolation = m_rows[lowerRowIndex].func(lowerColumnIndex) * (upperColumnIndex - pos.x * (m_width - 1.f)) + m_rows[lowerRowIndex].func(upperColumnIndex) * (pos.x * (m_width - 1.f) - lowerColumnIndex);
    float upperRowInterpolation = m_rows[upperRowIndex].func(lowerColumnIndex) * (upperColumnIndex - pos.x * (m_width - 1.f)) + m_rows[upperRowIndex].func(upperColumnIndex) * (pos.x * (m_width - 1.f) - lowerColumnIndex);

    return (lowerRowInterpolation * (upperRowIndex - pos.y * (m_height - 1.f)) + upperRowInterpolation * (pos.y * (m_height - 1.f) - lowerRowIndex));
  }

  void CDistribution2D::allocateDeviceMemory() {
    if (m_deviceResource) {
      freeDeviceMemory();
      delete m_deviceResource;
    }

    m_deviceResource = new SDistribution2D_DeviceResource;
    cudaMalloc(&m_deviceResource->d_marginal, sizeof(CDistribution1D));
    cudaMalloc(&m_deviceResource->d_rows, sizeof(CDistribution1D) * m_height);
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
    deviceDist.m_nElements = m_nElements;
    deviceDist.m_width = m_width;
    deviceDist.m_height = m_height;
    deviceDist.m_marginal = m_deviceResource->d_marginal;
    deviceDist.m_rows = m_deviceResource->d_rows;
    return deviceDist;
  }

  void CDistribution2D::freeDeviceMemory() {
    cudaFree(m_deviceResource->d_marginal);
    cudaFree(m_deviceResource->d_rows);
  }
}