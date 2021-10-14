#ifndef DISTRIBUTION_2D_HPP
#define DISTRIBUTION_2D_HPP
#include <vector>
#include "utility/qualifiers.hpp"
#include <glm/glm.hpp>
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
}
#endif // !DISTRIBUTION_2D_HPP
