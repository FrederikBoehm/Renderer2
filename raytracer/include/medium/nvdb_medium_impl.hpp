#ifndef NVDB_MEDIUM_IMPL_HPP
#define NVDB_MEDIUM_IMPL_HPP
#include "nvdb_medium.hpp"
#include <optix/optix_device.h>
#include "utility/functions.hpp"
#include "intersect/ray.hpp"
#include "scene/interaction.hpp"
#include "filtering/filtered_data.hpp"
#include "material/fresnel.hpp"

namespace rt {


  inline const CPhaseFunction& CNVDBMedium::phase() const {
    return *m_phase;
  }

  inline const SAABB& CNVDBMedium::worldBB() const {
    return m_worldBB;
  }

  inline float CNVDBMedium::density(const glm::vec3& p, const nanovdb::DefaultReadAccessor<float>& accessor) const {
    glm::vec3 pSamples(p.x * m_size.x + m_ibbMin.x - 0.5f, p.y * m_size.y + m_ibbMin.y - 0.5f, p.z * m_size.z + m_ibbMin.z - 0.5f);
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

  inline filter::SFilteredData CNVDBMedium::filteredData(const glm::vec3& p, const nanovdb::DefaultReadAccessor<nanovdb::Vec4d>& accessor) const {
    glm::vec3 pSamples(p.x * m_size.x + m_ibbMin.x - 0.5f, p.y * m_size.y + m_ibbMin.y - 0.5f, p.z * m_size.z + m_ibbMin.z - 0.5f);
    glm::ivec3 pi = glm::floor(pSamples);
    glm::vec3 d = pSamples - (glm::vec3)pi;
    int x = pi.x;
    int y = pi.y;
    int z = pi.z;

    filter::SFilteredData d00 = interpolate(d.x, getValue(pi, accessor), getValue(pi + glm::ivec3(1, 0, 0), accessor));
    filter::SFilteredData d10 = interpolate(d.x, getValue(pi + glm::ivec3(0, 1, 0), accessor), getValue(pi + glm::ivec3(1, 1, 0), accessor));
    filter::SFilteredData d01 = interpolate(d.x, getValue(pi + glm::ivec3(0, 0, 1), accessor), getValue(pi + glm::ivec3(1, 0, 1), accessor));
    filter::SFilteredData d11 = interpolate(d.x, getValue(pi + glm::ivec3(0, 1, 1), accessor), getValue(pi + glm::ivec3(1, 1, 1), accessor));

    filter::SFilteredData d0 = interpolate(d.y, d00, d10);
    filter::SFilteredData d1 = interpolate(d.y, d01, d11);

    return interpolate(d.z, d0, d1);
  }

  inline float CNVDBMedium::D(const glm::ivec3& p, const nanovdb::DefaultReadAccessor<float>& accessor) const {
    glm::vec3 pCopy = p;
    nanovdb::Coord coord(p.x, p.y, p.z);
    return accessor.getValue(coord);
  }

  inline filter::SFilteredData CNVDBMedium::getValue(const glm::ivec3& p, const nanovdb::DefaultReadAccessor<nanovdb::Vec4d>& accessor) const {
    glm::vec3 pCopy = p;
    nanovdb::Coord coord(p.x, p.y, p.z);
    nanovdb::Vec4d value = accessor.getValue(coord);
    return filter::SFilteredData(reinterpret_cast<filter::SFilteredDataCompact&>(value));
  }

  inline glm::vec3 CNVDBMedium::sample(const CRay& rayWorld, CSampler& sampler, float filterRenderRatio, SInteraction* mi) const {
    switch (m_gridType) {
    case nanovdb::GridType::Float:
      return sampleInternal(rayWorld, sampler, filterRenderRatio, mi, m_grid->getAccessor());
    case nanovdb::GridType::Vec4d:
      return sampleInternal(rayWorld, sampler, filterRenderRatio, mi, m_vec4grid->getAccessor());
    }
  }

  template <typename TReadAccessor>
  inline glm::vec3 CNVDBMedium::sampleInternal(const CRay& rayWorld, CSampler& sampler, float filterRenderRatio, SInteraction* mi, const TReadAccessor& accessor) const {
    static_assert(std::is_same<TReadAccessor, nanovdb::DefaultReadAccessor<float>>::value || std::is_same<TReadAccessor, nanovdb::DefaultReadAccessor<nanovdb::Vec4d>>::value,
      "Argument accessor has to be of type const nanovdb::DefaultReadAccessor<float>& or const nanovdb::DefaultReadAccessor<nanovdb::Vec4d>>&");

    const CRay ray = rayWorld.transform2(m_modelToIndex);
    float t = 0.f;
    float invMaxDensity = 1.f / (filterRenderRatio / m_invMaxDensity);
    while (true) {
      t -= glm::log(1.f - sampler.uniformSample01()) * invMaxDensity / m_sigma_t;
      if (t >= ray.m_t_max) {
        break;
      }
      if constexpr (std::is_same<TReadAccessor, nanovdb::DefaultReadAccessor<float>>::value) {
        float d = density(ray.m_origin + t * ray.m_direction, accessor);
        if (filterRenderRatio * d * invMaxDensity > sampler.uniformSample01()) {
          ray.m_t_max = t;
          CRay rayWorldNew = ray.transform2(m_indexToModel);
          rayWorld.m_t_max = rayWorldNew.m_t_max;
          glm::vec3 worldPos = rayWorldNew.m_origin + rayWorldNew.m_t_max * rayWorldNew.m_direction;
          SHitInformation hitInfo = { true, worldPos, glm::vec3(0.f), glm::vec3(0.f), glm::mat3(0.f), glm::vec2(0.f), 1.f, rayWorldNew.m_t_max };
          *mi = { hitInfo, nullptr, nullptr, nullptr };
          return m_sigma_s / m_sigma_t;
        }
      }
      else {
        filter::SFilteredData fD = filteredData(ray.m_origin + t * ray.m_direction, accessor);
        if (filterRenderRatio * fD.density * invMaxDensity > sampler.uniformSample01()) {
          ray.m_t_max = t;
          CRay rayWorldNew = ray.transform2(m_indexToModel);
          rayWorld.m_t_max = rayWorldNew.m_t_max;
          glm::vec3 worldPos = rayWorldNew.m_origin + rayWorldNew.m_t_max * rayWorldNew.m_direction;
          const glm::vec3 normal = fD.n();
          SHitInformation hitInfo = { true, worldPos, normal, normal, fD.S, glm::vec2(0.f), fD.ior, rayWorldNew.m_t_max };
          *mi = { hitInfo, nullptr, nullptr, nullptr };
          CFresnel fresnel(1.f, fD.ior);
          float weight = fresnel.evaluate(glm::abs(glm::dot(-rayWorldNew.m_direction, normal)));
          return (fD.diffuseColor * (1.f - weight) + fD.specularColor * weight) * m_sigma_s / m_sigma_t;
        }
      }

    }
    return glm::vec3(1.f);
  }

  inline glm::vec3 CNVDBMedium::tr(const CRay& rayWorld, CSampler& sampler, float filterRenderRatio) const {
    switch (m_gridType) {
    case nanovdb::GridType::Float:
      return trInternal(rayWorld, sampler, filterRenderRatio, m_grid->getAccessor());
    case nanovdb::GridType::Vec4d:
      return trInternal(rayWorld, sampler, filterRenderRatio, m_vec4grid->getAccessor());
    }
  }

  template <typename TReadAccessor>
  inline glm::vec3 CNVDBMedium::trInternal(const CRay& rayWorld, CSampler& sampler, float filterRenderRatio, const TReadAccessor& accessor) const {
    static_assert(std::is_same<TReadAccessor, nanovdb::DefaultReadAccessor<float>>::value || std::is_same<TReadAccessor, nanovdb::DefaultReadAccessor<nanovdb::Vec4d>>::value,
      "Argument accessor has to be of type const nanovdb::DefaultReadAccessor<float>& or const nanovdb::DefaultReadAccessor<nanovdb::Vec4d>>&");
    
    const CRay rayWorldCopy = rayWorld;
    const CRay ray = rayWorld.transform2(m_modelToIndex);
    float tr = 1.f;
    float t = 0.f;
    float invMaxDensity = 1.f / (filterRenderRatio / m_invMaxDensity);
    while (true) {
      t -= glm::log(1.f - sampler.uniformSample01()) * invMaxDensity / m_sigma_t;
      if (t >= ray.m_t_max) {
        break;
      }

      float d;
      if constexpr (std::is_same<TReadAccessor, nanovdb::DefaultReadAccessor<float>>::value) {
        d = density(ray.m_origin + t * ray.m_direction, accessor);
      }
      else {
        d = filteredData(ray.m_origin + t * ray.m_direction, accessor).density;
      }
      tr *= 1.f - glm::max(0.f, filterRenderRatio * d * invMaxDensity);

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