#ifndef NVDB_MEDIUM_IMPL_HPP
#define NVDB_MEDIUM_IMPL_HPP
#include "nvdb_medium.hpp"
#include <optix/optix_device.h>
#include "utility/functions.hpp"
#include "intersect/ray.hpp"
#include "scene/interaction.hpp"
#include "filtering/filtered_data.hpp"

namespace rt {


  inline const CPhaseFunction& CNVDBMedium::phase() const {
    return *m_phase;
  }

  inline const SAABB& CNVDBMedium::worldBB() const {
    return m_worldBB;
  }

  inline float CNVDBMedium::density(const glm::vec3& p, const nanovdb::DefaultReadAccessor<float>& accessor) const {
    //static_assert(std::is_same<TReadAccessor, nanovdb::DefaultReadAccessor<float>>::value || std::is_same<TReadAccessor, nanovdb::DefaultReadAccessor<nanovdb::Vec4d>>::value,
    //  "Argument accessor has to be of type const nanovdb::DefaultReadAccessor<float>& or const nanovdb::DefaultReadAccessor<nanovdb::Vec4d>>&");
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
    //static_assert(std::is_same<TReadAccessor, nanovdb::DefaultReadAccessor<float>>::value || std::is_same<TReadAccessor, nanovdb::DefaultReadAccessor<nanovdb::Vec4d>>::value,
    //  "Argument accessor has to be of type const nanovdb::DefaultReadAccessor<float>& or const nanovdb::DefaultReadAccessor<nanovdb::Vec4d>>&");
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

  template <typename TReadAccessor>
  inline glm::vec3 CNVDBMedium::normal(const glm::vec3& p, const TReadAccessor& accessor) const {
    static_assert(std::is_same<TReadAccessor, nanovdb::DefaultReadAccessor<float>>::value || std::is_same<TReadAccessor, nanovdb::DefaultReadAccessor<nanovdb::Vec4d>>::value,
      "Argument accessor has to be of type const nanovdb::DefaultReadAccessor<float>& or const nanovdb::DefaultReadAccessor<nanovdb::Vec4d>>&");
    glm::vec3 pMedium = m_modelToIndex * glm::vec4(p.x, p.y, p.z, 1.f);


    glm::vec3 pSamples(pMedium.x * m_size.x + m_ibbMin.x, pMedium.y * m_size.y + m_ibbMin.y, pMedium.z * m_size.z + m_ibbMin.z);
    glm::ivec3 pi = glm::floor(pSamples);

    float x;
    float y;
    float z;
    if constexpr (std::is_same<TReadAccessor, nanovdb::DefaultReadAccessor<float>>::value) {
      x = D(pi - glm::ivec3(1, 0, 0), accessor) - D(pi + glm::ivec3(1, 0, 0), accessor);
      y = D(pi - glm::ivec3(0, 1, 0), accessor) - D(pi + glm::ivec3(0, 1, 0), accessor);
      z = D(pi - glm::ivec3(0, 0, 1), accessor) - D(pi + glm::ivec3(0, 0, 1), accessor);
    }
    else {
      x = getValue(pi - glm::ivec3(1, 0, 0), accessor).density - getValue(pi + glm::ivec3(1, 0, 0), accessor).density;
      y = getValue(pi - glm::ivec3(0, 1, 0), accessor).density - getValue(pi + glm::ivec3(0, 1, 0), accessor).density;
      z = getValue(pi - glm::ivec3(0, 0, 1), accessor).density - getValue(pi + glm::ivec3(0, 0, 1), accessor).density;
    }

    glm::vec3 n = glm::normalize(glm::vec3(x, y, z));
    if (glm::any(glm::isnan(n)) || glm::any(glm::isinf(n))) { // this can happen if x, y, z is zero or really close to zero
      return glm::vec3(0.f);
    }
    else {
      return glm::normalize(m_indexToModel * glm::vec4(n.x, n.y, n.z, 0.f));
    }
  }

  inline glm::vec3 CNVDBMedium::normal(const glm::vec3& p, CSampler& sampler) const {
    glm::vec3 n;
    switch (m_gridType) {
    case nanovdb::GridType::Float:
      n = normal(p, m_grid->getAccessor());
      break;
    case nanovdb::GridType::Vec4d:
      n = normal(p, m_vec4grid->getAccessor());
      break;
    }
    if (n == glm::vec3(0.f)) {
      return sampler.uniformSampleSphere(); // As a fallback sample sphere uniformly
    }
    else {
      return n;
    }
  }

  inline glm::vec3 CNVDBMedium::sample(const CRay& rayWorld, CSampler& sampler, SInteraction* mi) const {
    switch (m_gridType) {
    case nanovdb::GridType::Float:
      return sampleInternal(rayWorld, sampler, mi, m_grid->getAccessor());
    case nanovdb::GridType::Vec4d:
      return sampleInternal(rayWorld, sampler, mi, m_vec4grid->getAccessor());
    }
  }

  template <typename TReadAccessor>
  inline glm::vec3 CNVDBMedium::sampleInternal(const CRay& rayWorld, CSampler& sampler, SInteraction* mi, const TReadAccessor& accessor) const {
    static_assert(std::is_same<TReadAccessor, nanovdb::DefaultReadAccessor<float>>::value || std::is_same<TReadAccessor, nanovdb::DefaultReadAccessor<nanovdb::Vec4d>>::value,
      "Argument accessor has to be of type const nanovdb::DefaultReadAccessor<float>& or const nanovdb::DefaultReadAccessor<nanovdb::Vec4d>>&");

    const CRay ray = rayWorld.transform(m_modelToIndex);
    float t = 0.f;
    while (true) {
      t -= glm::log(1.f - sampler.uniformSample01()) * m_invMaxDensity / m_sigma_t;
      if (t >= ray.m_t_max) {
        break;
      }
      if constexpr (std::is_same<TReadAccessor, nanovdb::DefaultReadAccessor<float>>::value) {
        float d = density(ray.m_origin + t * ray.m_direction, accessor);
        //filter::SFilteredData fD = filteredData(ray.m_origin + t * ray.m_direction, accessor);
        if (d * m_invMaxDensity > sampler.uniformSample01()) {
          ray.m_t_max = t;
          CRay rayWorldNew = ray.transform(m_indexToModel);
          rayWorld.m_t_max = rayWorldNew.m_t_max;
          glm::vec3 worldPos = rayWorldNew.m_origin + rayWorldNew.m_t_max * rayWorldNew.m_direction;
          glm::vec3 n = normal(worldPos, accessor);
          SHitInformation hitInfo = { true, worldPos, n, n, glm::vec2(0.f), rayWorldNew.m_t_max };
          *mi = { hitInfo, nullptr, nullptr, nullptr };
          return m_sigma_s / m_sigma_t;
          //const uint16_t MAX_U16 = -1;
          //const float fMAX_U16 = MAX_U16;
          //return glm::vec3(fD.diffuseColor) / fMAX_U16;
        }
      }
      else {
        filter::SFilteredData fD = filteredData(ray.m_origin + t * ray.m_direction, accessor);
        if (fD.density * m_invMaxDensity > sampler.uniformSample01()) {
          ray.m_t_max = t;
          CRay rayWorldNew = ray.transform(m_indexToModel);
          rayWorld.m_t_max = rayWorldNew.m_t_max;
          glm::vec3 worldPos = rayWorldNew.m_origin + rayWorldNew.m_t_max * rayWorldNew.m_direction;
          glm::vec3 n = normal(worldPos, accessor);
          SHitInformation hitInfo = { true, worldPos, n, n, glm::vec2(0.f), rayWorldNew.m_t_max };
          *mi = { hitInfo, nullptr, nullptr, nullptr };
          return fD.diffuseColor * m_sigma_s / m_sigma_t;
          //const uint16_t MAX_U16 = -1;
          //const float fMAX_U16 = MAX_U16;
          //return glm::vec3(fD.diffuseColor) / fMAX_U16;
        }
      }

    }
    return glm::vec3(1.f);
  }

  inline glm::vec3 CNVDBMedium::tr(const CRay& rayWorld, CSampler& sampler) const {
    switch (m_gridType) {
    case nanovdb::GridType::Float:
      return trInternal(rayWorld, sampler, m_grid->getAccessor());
    case nanovdb::GridType::Vec4d:
      return trInternal(rayWorld, sampler, m_vec4grid->getAccessor());
    }
  }

  template <typename TReadAccessor>
  inline glm::vec3 CNVDBMedium::trInternal(const CRay& rayWorld, CSampler& sampler, const TReadAccessor& accessor) const {
    static_assert(std::is_same<TReadAccessor, nanovdb::DefaultReadAccessor<float>>::value || std::is_same<TReadAccessor, nanovdb::DefaultReadAccessor<nanovdb::Vec4d>>::value,
      "Argument accessor has to be of type const nanovdb::DefaultReadAccessor<float>& or const nanovdb::DefaultReadAccessor<nanovdb::Vec4d>>&");
    
    const CRay rayWorldCopy = rayWorld;
    const CRay ray = rayWorld.transform(m_modelToIndex);
    float tr = 1.f;
    float t = 0.f;
    while (true) {
      t -= glm::log(1.f - sampler.uniformSample01()) * m_invMaxDensity / m_sigma_t;
      if (t >= ray.m_t_max) {
        break;
      }

      float d;
      if constexpr (std::is_same<TReadAccessor, nanovdb::DefaultReadAccessor<float>>::value) {
        d = density(ray.m_origin + t * ray.m_direction, accessor);
      }
      else {
        d = getValue(ray.m_origin + t * ray.m_direction, accessor).density;
      }
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