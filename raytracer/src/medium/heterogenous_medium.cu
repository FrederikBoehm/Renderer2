#include "medium/heterogenous_medium.hpp"
#include <glm/gtx/transform.hpp>
#include "utility/functions.hpp"
#include "intersect/ray.hpp"
#include "sampling/sampler.hpp"
#include "scene/interaction.hpp"
#include <stdio.h>
#include "medium/medium.hpp"

namespace rt {
  CHeterogenousMedium::CHeterogenousMedium(const glm::vec3& sigma_a, const glm::vec3& sigma_s, float g, uint32_t nx, uint32_t ny, uint32_t nz, const glm::vec3& pos, const glm::vec3& size, const float* d) :
    CMedium(EMediumType::HETEROGENOUS_MEDIUM),
    m_sigma_a(sigma_a),
    m_sigma_s(sigma_s),
    m_phase(g),
    m_nx(nx),
    m_ny(ny),
    m_nz(nz),
    m_mediumToWorld(CHeterogenousMedium::getMediumToWorldTransformation(pos, size)),
    m_worldToMedium(glm::inverse(m_mediumToWorld)),
    m_density(new float[nx * ny * nz]),
    m_deviceResource(nullptr)  {
    memcpy(m_density, d, nx * ny * nz * sizeof(float));
    m_sigma_t = (m_sigma_a + m_sigma_s).z; // Why first channel?
    float maxDensity = 0.f;
    for (size_t i = 0; i < nx * ny * nz; ++i) {
      maxDensity = glm::max(maxDensity, m_density[i]);
    }
    m_invMaxDensity = 1.f / maxDensity;
  }

  CHeterogenousMedium::CHeterogenousMedium(const glm::vec3& sigma_a,
    const glm::vec3& sigma_s,
    const CHenyeyGreensteinPhaseFunction& phase,
    const uint32_t nx,
    const uint32_t ny,
    const uint32_t nz,
    const glm::mat4& mediumToWorld,
    const glm::mat4& worldToMedium,
    float* density,
    float sigma_t,
    float invMaxDensity,
    SHeterogenousMedium_DeviceResource* deviceResource):
    CMedium(EMediumType::HETEROGENOUS_MEDIUM),
    m_sigma_a(sigma_a),
    m_sigma_s(sigma_s),
    m_phase(phase),
    m_nx(nx),
    m_ny(ny),
    m_nz(nz),
    m_mediumToWorld(mediumToWorld),
    m_worldToMedium(worldToMedium),
    m_density(density),
    m_sigma_t(sigma_t),
    m_invMaxDensity(invMaxDensity),
    m_deviceResource(deviceResource) {

  }

  glm::mat4 CHeterogenousMedium::getMediumToWorldTransformation(const glm::vec3& pos, const glm::vec3& size) {
    glm::mat4 scaling = glm::scale(size);
    glm::mat4 translation1 = glm::translate(glm::vec3(-0.5f)); // Move corner to origin
    glm::mat4 translation2 = glm::translate(pos);
    return translation2 * scaling * translation1;
  }

  float CHeterogenousMedium::density(const glm::vec3& p) const {
    glm::vec3 pSamples(p.x * m_nx - 0.5f, p.y * m_ny - 0.5f, p.z * m_nz - 0.5f);
    glm::ivec3 pi = glm::floor(pSamples);
    glm::vec3 d = pSamples - (glm::vec3)pi;
    
    float d00 = interpolate(d.x, D(pi), D(pi + glm::ivec3(1, 0, 0)));
    float d10 = interpolate(d.x, D(pi + glm::ivec3(0, 1, 0)), D(pi + glm::ivec3(1, 1, 0)));
    float d01 = interpolate(d.x, D(pi + glm::ivec3(0, 0, 1)), D(pi + glm::ivec3(1, 0, 1)));
    float d11 = interpolate(d.x, D(pi + glm::ivec3(0, 1, 1)), D(pi + glm::ivec3(1, 1, 1)));

    float d0 = interpolate(d.y, d00, d10);
    float d1 = interpolate(d.y, d01, d11);
    
    return interpolate(d.z, d0, d1);
  }

  float CHeterogenousMedium::D(const glm::ivec3& p) const {
    if (!insideExclusive(p, glm::ivec3(0), glm::ivec3(m_nx, m_ny, m_nz))) {
      return 0.f;
    }
    return m_density[(p.z * m_ny + p.y) * m_nx + p.x];
  }

  glm::vec3 CHeterogenousMedium::sample(const CRay& rayWorld, CSampler& sampler, SInteraction* mi) const {
    const CRay ray = rayWorld.transform(m_worldToMedium);

    float t = 0.f;
    while (true) {
      t -= glm::log(1.f - sampler.uniformSample01()) * m_invMaxDensity / m_sigma_t;
      if (t >= ray.m_t_max) {
        break;
      }
      if (density(ray.m_origin + t * ray.m_direction) * m_invMaxDensity > sampler.uniformSample01()) {
        ray.m_t_max = t;
        CRay rayWorldNew = ray.transform(m_mediumToWorld);
        SHitInformation hitInfo = { true, rayWorldNew.m_origin + rayWorldNew.m_t_max * rayWorldNew.m_direction, glm::vec3(0.f), glm::vec3(0.f), glm::vec2(0.f), rayWorldNew.m_t_max };
        *mi = { hitInfo, nullptr, nullptr, this };
        return m_sigma_s / m_sigma_t;
      }
    }
    return glm::vec3(1.f);
  }

  glm::vec3 CHeterogenousMedium::tr(const CRay& rayWorld, CSampler& sampler) const {
    const CRay ray = rayWorld.transform(m_worldToMedium);
    float tr = 1.f;
    float t = 0.f;
    while (true) {
      t -= glm::log(1.f - sampler.uniformSample01()) * m_invMaxDensity / m_sigma_t;
      if (t >= ray.m_t_max) {
        break;
      }

      float d = density(ray.m_origin + t * ray.m_direction);
      tr *= 1.f - glm::max(0.f, d * m_invMaxDensity);
    }
    return glm::vec3(tr);
  }

  void CHeterogenousMedium::allocateDeviceMemory() {
    if (m_deviceResource) {
      freeDeviceMemory();
      delete m_deviceResource;
    }
    m_deviceResource = new SHeterogenousMedium_DeviceResource;
    cudaMalloc(&m_deviceResource->d_density, sizeof(float) * m_nx * m_ny * m_nz);
  }

  CHeterogenousMedium CHeterogenousMedium::copyToDevice() const {
    float* density = nullptr;
    if (m_deviceResource) {
      cudaMemcpy(m_deviceResource->d_density, m_density, sizeof(float) * m_nx * m_ny * m_nz, cudaMemcpyHostToDevice);
      density = m_deviceResource->d_density;
    }
    CHeterogenousMedium m(m_sigma_a,
                          m_sigma_s,
                          m_phase,
                          m_nx,
                          m_ny,
                          m_nz,
                          m_mediumToWorld,
                          m_worldToMedium,
                          density,
                          m_sigma_t,
                          m_invMaxDensity,
                          nullptr);
    return m;
  }

  void CHeterogenousMedium::freeDeviceMemory() const {
    if (m_deviceResource) {
      cudaFree(m_deviceResource->d_density);
    }
  }
}