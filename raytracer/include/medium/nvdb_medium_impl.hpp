#ifndef NVDB_MEDIUM_IMPL_HPP
#define NVDB_MEDIUM_IMPL_HPP
#include "nvdb_medium.hpp"
#include <optix/optix_device.h>
#include "utility/functions.hpp"
#include "intersect/ray.hpp"
#include "scene/interaction.hpp"

namespace rt {
  inline CNVDBMedium::~CNVDBMedium() {
#ifndef __CUDA_ARCH__
    if (m_isHostObject) {
      freeDeviceMemory();
      delete m_readAccessor;
      delete m_handle;
      delete m_phase;
      cudaFree((void*)m_deviceAabb);
    }
#endif
  }

  inline const nanovdb::NanoGrid<float>* CNVDBMedium::grid() const {
    return m_grid;
  }

  inline const CPhaseFunction& CNVDBMedium::phase() const {
    return *m_phase;
  }

  inline float CNVDBMedium::density(const glm::vec3& p, const nanovdb::DefaultReadAccessor<float>& accessor) const {
    glm::vec3 pSamples(p.x * m_size.x - 0.5f, p.y * m_size.y - 0.5f, p.z * m_size.z - 0.5f);
    glm::ivec3 pi = glm::floor(pSamples);
    glm::vec3 d = pSamples - (glm::vec3)pi;
    int x = pi.x;
    int y = pi.y;
    int z = pi.z;

    float d00 = interpolate(d.x, D(pi, accessor), D(pi + glm::ivec3(1, 0, 0), accessor));
    float d10 = interpolate(d.x, D(pi + glm::ivec3(0, 1, 0), accessor), D(pi + glm::ivec3(1, 1, 0), accessor));
    float d01 = interpolate(d.x, D(pi + glm::ivec3(0, 0, 1), accessor), D(pi + glm::ivec3(1, 0, 1), accessor));
    float d11 = interpolate(d.x, D(pi + glm::ivec3(0, 1, 1), accessor), D(pi + glm::ivec3(1, 1, 1), accessor));

    float d0 = interpolate(d.y, d00, d10);
    float d1 = interpolate(d.y, d01, d11);

    return interpolate(d.z, d0, d1);
  }

  inline float CNVDBMedium::D(const glm::ivec3& p, const nanovdb::DefaultReadAccessor<float>& accessor) const {
    glm::vec3 pCopy = p;
    if (!insideExclusive(p, glm::ivec3(-m_size.x / 2.f, -m_size.y / 2.f, -m_size.z / 2.f), glm::ivec3(m_size.x / 2.f, m_size.y / 2.f, m_size.z / 2.f))) {
      return 0.f;
    }
    nanovdb::Coord coord(p.x, p.y, p.z);
    return accessor.getValue(coord);
  }

  inline glm::vec3 CNVDBMedium::normal(const glm::vec3& p, const nanovdb::DefaultReadAccessor<float>& accessor) const {
    glm::vec3 pMedium = glm::vec3(m_worldToMedium * glm::vec4(p.x, p.y, p.z, 1.f));


    glm::vec3 pSamples(pMedium.x * m_size.x, pMedium.y * m_size.y, pMedium.z * m_size.z);
    glm::ivec3 pi = glm::floor(pSamples);

    float x = D(pi - glm::ivec3(1, 0, 0), accessor) - D(pi + glm::ivec3(1, 0, 0), accessor);
    float y = D(pi - glm::ivec3(0, 1, 0), accessor) - D(pi + glm::ivec3(0, 1, 0), accessor);
    float z = D(pi - glm::ivec3(0, 0, 1), accessor) - D(pi + glm::ivec3(0, 0, 1), accessor);

    glm::vec3 n = glm::normalize(glm::vec3(x, y, z));
    if (glm::any(glm::isnan(n)) || glm::any(glm::isinf(n))) { // this can happen if x, y, z is zero or really close to zero
      return glm::vec3(0.f);
    }
    else {
      return glm::normalize(glm::vec3(m_mediumToWorld * glm::vec4(n.x, n.y, n.z, 0.f)));
    }
  }

  inline glm::vec3 CNVDBMedium::normal(const glm::vec3& p, CSampler& sampler) const {
    glm::vec3 pMedium = glm::vec3(m_worldToMedium * glm::vec4(p.x, p.y, p.z, 1.f));
    nanovdb::DefaultReadAccessor<float> accessor(m_grid->getAccessor());


    glm::vec3 pSamples(pMedium.x * m_size.x, pMedium.y * m_size.y, pMedium.z * m_size.z);
    glm::ivec3 pi = glm::floor(pSamples);

    float x = D(pi - glm::ivec3(1, 0, 0), accessor) - D(pi + glm::ivec3(1, 0, 0), accessor);
    float y = D(pi - glm::ivec3(0, 1, 0), accessor) - D(pi + glm::ivec3(0, 1, 0), accessor);
    float z = D(pi - glm::ivec3(0, 0, 1), accessor) - D(pi + glm::ivec3(0, 0, 1), accessor);

    glm::vec3 n = glm::normalize(glm::vec3(x, y, z));
    if (glm::any(glm::isnan(n)) || glm::any(glm::isinf(n))) { // this can happen if x, y, z is zero or really close to zero
      return sampler.uniformSampleSphere(); // As a fallback sample sphere uniformly
      //return glm::vec3(1.f, 0.f, 0.f);
    }
    else {
      return glm::normalize(glm::vec3(m_mediumToWorld * glm::vec4(n.x, n.y, n.z, 0.f)));
    }
  }

  inline glm::vec3 CNVDBMedium::sample(const CRay& rayWorld, CSampler& sampler, SInteraction* mi) const {
    const CRay ray = rayWorld.transform(m_worldToMedium);
    const CRay rayWorldCopy = rayWorld;
    nanovdb::DefaultReadAccessor<float> accessor(m_grid->getAccessor());
    uint3 launchIdx = optixGetLaunchIndex();
    float t = 0.f;
    while (true) {
      t -= glm::log(1.f - sampler.uniformSample01()) * m_invMaxDensity / m_sigma_t;
      if (t >= ray.m_t_max) {
        break;
      }
      float d = density(ray.m_origin + t * ray.m_direction, accessor);
      if (d * m_invMaxDensity > sampler.uniformSample01()) {
        ray.m_t_max = t;
        CRay rayWorldNew = ray.transform(m_mediumToWorld);
        glm::vec3 worldPos = rayWorldNew.m_origin + rayWorldNew.m_t_max * rayWorldNew.m_direction;
        SHitInformation hitInfo = { true, worldPos, normal(worldPos, accessor), rayWorldNew.m_t_max };
        *mi = { hitInfo, nullptr, nullptr, this };
        return m_sigma_s / m_sigma_t;
      }
    }
    return glm::vec3(1.f);
  }

  inline glm::vec3 CNVDBMedium::tr(const CRay& rayWorld, CSampler& sampler) const {
    const CRay rayWorldCopy = rayWorld;
    const CRay ray = rayWorld.transform(m_worldToMedium);
    nanovdb::DefaultReadAccessor<float> accessor(m_grid->getAccessor());
    float tr = 1.f;
    float t = 0.f;
    while (true) {
      t -= glm::log(1.f - sampler.uniformSample01()) * m_invMaxDensity / m_sigma_t;
      if (t >= ray.m_t_max) {
        break;
      }

      float d = density(ray.m_origin + t * ray.m_direction, accessor);
      tr *= 1.f - glm::max(0.f, d * m_invMaxDensity);

      if (tr < 1.f) {
        float p = 1.f - tr;
        if (sampler.uniformSample01() < p) {
          return glm::vec3(0.f);
        }
        else {
          tr /= 1.f - p;
        }
      }
    }
    return glm::vec3(tr);
  }
}
#endif