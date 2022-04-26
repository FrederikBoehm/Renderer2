#ifndef NVDB_MEDIUM_IMPL_HPP
#define NVDB_MEDIUM_IMPL_HPP
#include "nvdb_medium.hpp"
#include <optix/optix_device.h>
#include "utility/functions.hpp"
#include "intersect/ray.hpp"
#include "scene/interaction.hpp"
#include "filtering/filtered_data.hpp"
#include "material/fresnel.hpp"
#include "utility/functions.hpp"
#include "grid_brick/device_grid_brick_impl.hpp"

#define MIP_START 3
#define MIP_SPEED_UP 0.25
#define MIP_SPEED_DOWN 2

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
    //switch (m_gridType) {
    //case nanovdb::GridType::Float:
    //  return sampleInternal(rayWorld, sampler, filterRenderRatio, mi, m_grid->getAccessor());
    //case nanovdb::GridType::Vec4d:
      //return sampleInternal(rayWorld, sampler, filterRenderRatio, mi, m_vec4grid->getAccessor());
    //}
    return sampleDDA(rayWorld, sampler, filterRenderRatio, mi, m_vec4grid->getAccessor());
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

  inline glm::vec3 CNVDBMedium::sampleDDA(const CRay& rayWorld, CSampler& sampler, float filterRenderRatio, SInteraction* mi, const nanovdb::DefaultReadAccessor<nanovdb::Vec4d>& accessor) const {

    const CRay ray = rayWorld.transform2(m_modelToIndex);
    glm::mat4x3 indexToScaledIndex(glm::vec3(m_size.x, 0.f, 0.f), glm::vec3(0.f, m_size.y, 0.f), glm::vec3(0.f, 0.f, m_size.z), m_ibbMin);
    const CRay iray = ray.transform2(indexToScaledIndex);
    const glm::vec3 ri = 1.f / iray.m_direction;
    float tr = 1.f;
    float t = 0.f;
    float invMaxDensity = 1.f / (filterRenderRatio / m_invMaxDensity);
    float tau = -glm::log(1.f - sampler.uniformSample01());
    float mip = MIP_START;
    float volMajorant = 1.f / invMaxDensity;
    while (t < iray.m_t_max) {
      const glm::vec3 curr = iray.m_origin + t * iray.m_direction;

      const float majorant = filterRenderRatio * m_deviceBrickGrid->lookupMajorant(curr, int(glm::round(mip)));
      const float dt = stepDDA(curr, ri, int(glm::round(mip)));
      t += dt / m_sigma_t;
      tau -= majorant * dt;
      mip = glm::min(float(mip + MIP_SPEED_UP), 3.f);
      if (tau > 0.f) {
        continue; // no collision, step ahead
      }
      t += tau / (majorant * m_sigma_t); // step back to point of collision
      if (t >= iray.m_t_max) {
        break;
      }

      const float d = m_deviceBrickGrid->lookupDensity(iray.m_origin + t * iray.m_direction, glm::vec3(sampler.uniformSample01(), sampler.uniformSample01(), sampler.uniformSample01()));
      if (sampler.uniformSample01() * majorant < d * filterRenderRatio) {
        iray.m_t_max = t;
        CRay rayNew = iray.transform2(glm::inverse(glm::mat4(indexToScaledIndex)));
        glm::vec3 pos = ((iray.m_origin + t * iray.m_direction) - glm::vec3(m_ibbMin)) / glm::vec3(m_size);
        filter::SFilteredData fD = filteredData(pos, accessor);
        CRay rayWorldNew = rayNew.transform2(m_indexToModel);
        rayWorld.m_t_max = rayWorldNew.m_t_max;
        glm::vec3 worldPos = rayWorldNew.m_origin + rayWorldNew.m_t_max * rayWorldNew.m_direction;
        const glm::vec3 normal = fD.n();
        SHitInformation hitInfo = { true, worldPos, normal, normal, fD.S, glm::vec2(0.f), fD.ior, rayWorldNew.m_t_max };
        *mi = { hitInfo, nullptr, nullptr, nullptr };
        CFresnel fresnel(1.f, fD.ior);
        float weight = fresnel.evaluate(glm::abs(glm::dot(-rayWorldNew.m_direction, normal))); // TODO: normalize ray dir
        return (fD.diffuseColor * (1.f - weight) + fD.specularColor * weight) * m_sigma_s / m_sigma_t;
      }

      tau = -glm::log(1.f - sampler.uniformSample01());
      mip = glm::max(0.f, mip - MIP_SPEED_DOWN);
    }
    return glm::vec3(1.f);
  }

  inline glm::vec3 CNVDBMedium::tr(const CRay& rayWorld, CSampler& sampler, float filterRenderRatio) const {
    //switch (m_gridType) {
    //case nanovdb::GridType::Float:
    //  return trInternal(rayWorld, sampler, filterRenderRatio, m_grid->getAccessor());
    //case nanovdb::GridType::Vec4d:
      //return trInternal(rayWorld, sampler, filterRenderRatio, m_vec4grid->getAccessor());
    //}
    return trDDA(rayWorld, sampler, filterRenderRatio);
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

  inline float CNVDBMedium::stepDDA(const glm::vec3& pos, const glm::vec3& inv_dir, const int mip) const {
    const float dim = 8 << mip;
    const glm::vec3 offs = glm::mix(glm::vec3(-0.5f), glm::vec3(dim + 0.5f), glm::greaterThanEqual(inv_dir, glm::vec3(0.f)));
    const glm::vec3 tmax = (glm::floor(pos * (1.f / dim)) * dim + offs - pos) * inv_dir;
    return glm::min(tmax.x, glm::min(tmax.y, tmax.z));
  }

  inline glm::vec3 CNVDBMedium::trDDA(const CRay& rayWorld, CSampler& sampler, float filterRenderRatio) const {
    const CRay rayWorldCopy = rayWorld;
    const CRay ray = rayWorld.transform2(m_modelToIndex);
    glm::mat4x3 indexToScaledIndex(glm::vec3(m_size.x, 0.f, 0.f), glm::vec3(0.f, m_size.y, 0.f), glm::vec3(0.f, 0.f, m_size.z), m_ibbMin);
    const CRay iray = ray.transform2(indexToScaledIndex);
    const glm::vec3 ri = 1.f / iray.m_direction;
    float tr = 1.f;
    float t = 0.f;
    float invMaxDensity = 1.f / (filterRenderRatio / m_invMaxDensity);
    float tau = -glm::log(1.f - sampler.uniformSample01());
    float mip = MIP_START;
    float volMajorant = 1.f / invMaxDensity;
    while (t < iray.m_t_max) {
      const glm::vec3 curr = iray.m_origin + t * iray.m_direction;

      const float majorant = filterRenderRatio * m_deviceBrickGrid->lookupMajorant(curr, int(glm::round(mip)));
      const float dt = stepDDA(curr, ri, int(glm::round(mip)));
      t += dt / m_sigma_t;
      tau -= majorant * dt;
      mip = glm::min(float(mip + MIP_SPEED_UP), 3.f);
      if (tau > 0.f) {
        continue; // no collision, step ahead
      }
      t += tau / (majorant * m_sigma_t); // step back to point of collision
      if (t >= iray.m_t_max) {
        break;
      }

      const float d = m_deviceBrickGrid->lookupDensity(iray.m_origin + t * iray.m_direction, glm::vec3(sampler.uniformSample01(), sampler.uniformSample01(), sampler.uniformSample01()));

      if (sampler.uniformSample01() * majorant < d * filterRenderRatio) { // check if real or null collision
        tr *= glm::max(0.f, 1.f - volMajorant / majorant); // adjust by ratio of global to local majorant
        // russian roulette
        if (tr < .1f) {
          const float prob = 1.f - tr;
          if (sampler.uniformSample01() < prob) {
            return glm::vec3(0.f);
          }
          tr /= 1.f - prob;
        }
      }
      tau = -glm::log(1.f - sampler.uniformSample01());
      mip = glm::max(0.f, float(mip - MIP_SPEED_DOWN));

    }
    return glm::vec3(tr);
  }

  inline CRay CNVDBMedium::moveToVoxelBorder(const CRay& ray) const {
    CRay rayIndex = ray.transform(m_modelToIndex);
    glm::vec3 ibbMin = glm::vec3(m_ibbMin);
    glm::vec3 size = glm::vec3(m_size);
    glm::vec3 integerPos = rayIndex.m_origin * size + ibbMin;
    glm::vec3 voxelMin = (glm::floor(integerPos) - ibbMin) / size;
    glm::vec3 voxelMax = (glm::ceil(integerPos) - ibbMin) / size;

    constexpr float gamma = (3.f * FLT_EPSILON) / (1.f - 3.f * FLT_EPSILON);
    float t1 = rayIndex.m_t_max;
    for (uint8_t i = 0; i < 3; ++i) {
      float invRayDir = 1.f / rayIndex.m_direction[i];
      float tNear = (voxelMin[i] - rayIndex.m_origin[i]) * invRayDir;
      float tFar = (voxelMax[i] - rayIndex.m_origin[i]) * invRayDir;

      if (tNear > tFar) {
        swap(tNear, tFar);
      }
      tFar *= 1.f + 2.f * gamma;
      t1 = tFar < t1 ? tFar : t1;
    }

    t1 *= (1.f - CRay::OFFSET); // Just slightly before leaving voxel
    return CRay(rayIndex.m_origin + t1 * rayIndex.m_direction, rayIndex.m_direction, rayIndex.m_t_max - t1, rayIndex.m_medium).transform(m_indexToModel);
  }
}
#endif