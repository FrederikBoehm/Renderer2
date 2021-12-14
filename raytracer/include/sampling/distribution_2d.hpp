#ifndef DISTRIBUTION_2D_HPP
#define DISTRIBUTION_2D_HPP
#include <vector>
#include "utility/qualifiers.hpp"
#include <glm/glm.hpp>
#include "sampling/distribution_1d.hpp"
namespace rt {
  class CDistribution1D;
  class CSampler;
  struct SDistribution2D_DeviceResource {
    CDistribution1D* d_marginal;
    CDistribution1D* d_rows;
  };
  class CDistribution2D {
  public:
    DH_CALLABLE CDistribution2D();
    H_CALLABLE CDistribution2D(uint16_t width, uint16_t height, const std::vector<float>& data);
    H_CALLABLE ~CDistribution2D();

    DH_CALLABLE glm::vec2 sampleContinuous(CSampler& sampler, float* pdf, size_t* off = nullptr) const;
    //DH_CALLABLE size_t sampleDiscrete(CSampler& sampler, float* pdf = nullptr, float* uRemapped = nullptr) const;
    //DH_CALLABLE size_t count() const;
    //DH_CALLABLE float discretePdf(size_t index) const;
    DH_CALLABLE float pdf(const glm::vec2& pos) const;
    //DH_CALLABLE float integral() const;
    H_CALLABLE std::vector<float> sumRows(const std::vector<float>& data) const;

    // CUDA stuff
    H_CALLABLE void allocateDeviceMemory();
    H_CALLABLE CDistribution2D copyToDevice();
    H_CALLABLE void freeDeviceMemory();

  private:
    uint32_t m_nElements;
    uint16_t m_width;
    uint16_t m_height;
    CDistribution1D* m_marginal;
    CDistribution1D* m_rows;

    SDistribution2D_DeviceResource* m_deviceResource;
  };

  inline CDistribution2D::CDistribution2D() : m_width(0), m_height(0), m_marginal(nullptr), m_rows(0), m_deviceResource(nullptr) {

  }

  inline glm::vec2 CDistribution2D::sampleContinuous(CSampler& sampler, float* pdf, size_t* off) const {
    float pdfs[2];
    size_t v;
    float y = m_marginal->sampleContinuous(sampler, &pdfs[1], &v);
    float x = m_rows[v].sampleContinuous(sampler, &pdfs[0]);
    *pdf = pdfs[0] * pdfs[1];
    return glm::vec2(x, y);
  }

  inline std::vector<float> CDistribution2D::sumRows(const std::vector<float>& data) const {
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

  inline float CDistribution2D::pdf(const glm::vec2& pos) const {
    uint16_t lowerRowIndex = glm::clamp(int(pos.y * (m_height - 2)), 0, m_height - 2);
    uint16_t upperRowIndex = lowerRowIndex + 1;

    uint16_t lowerColumnIndex = glm::clamp(int(pos.x * (m_width - 2)), 0, m_width - 2);
    uint16_t upperColumnIndex = lowerColumnIndex + 1;

    float lowerRowInterpolation = m_rows[lowerRowIndex].func(lowerColumnIndex) * (upperColumnIndex - pos.x * (m_width - 2.f)) + m_rows[lowerRowIndex].func(upperColumnIndex) * (pos.x * (m_width - 2.f) - lowerColumnIndex);
    float upperRowInterpolation = m_rows[upperRowIndex].func(lowerColumnIndex) * (upperColumnIndex - pos.x * (m_width - 2.f)) + m_rows[upperRowIndex].func(upperColumnIndex) * (pos.x * (m_width - 2.f) - lowerColumnIndex);

    return (lowerRowInterpolation * (upperRowIndex - pos.y * (m_height - 2.f)) + upperRowInterpolation * (pos.y * (m_height - 2.f) - lowerRowIndex));
  }
}
#endif // !DISTRIBUTION_2D_HPP
