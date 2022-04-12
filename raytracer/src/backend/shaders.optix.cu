#include <cuda_runtime.h>
#include <crt/device_functions.h>
#include "intersect/hit_information.hpp"
#include <optix/optix_device.h>
#include <glm/glm.hpp>
#include "backend/types.hpp"
#include "camera/camera.hpp"
#include "intersect/ray.hpp"
#include "scene/device_scene.hpp"
#include "scene/device_sceneobject.hpp"
#include <stdio.h>
#include "medium/nvdb_medium.hpp"
#include <nanovdb/util/Ray.h>
#include "integrators/path_integrator_impl.hpp"
#include "camera/pixel_sampler.hpp"
#include "mesh/mesh.hpp"
#include "filtering/mesh_filter.hpp"
#include "filtering/launch_params.hpp"


using namespace rt;
using namespace filter;

__constant__ rt::SLaunchParams params;
__constant__ filter::SFilterLaunchParams paramsFiltering;


extern "C" __global__ void __closesthit__ch() {
  const rt::CDeviceSceneobject** sceneobjectPtr = reinterpret_cast<const rt::CDeviceSceneobject**>(optixGetSbtDataPointer());
  const rt::CDeviceSceneobject* sceneobject = *sceneobjectPtr;

  unsigned int siAdress[2];
  siAdress[0] = optixGetPayload_0();
  siAdress[1] = optixGetPayload_1();

  SInteraction* si;
  memcpy(&si, siAdress, sizeof(SInteraction*));

  //float3 hitPos = { uint_as_float(optixGetAttribute_0()), uint_as_float(optixGetAttribute_1()), uint_as_float(optixGetAttribute_2()) };
  float3 rayOrigin = optixGetWorldRayOrigin();
  float3 rayDirection = optixGetWorldRayDirection();
  float tMax = optixGetRayTmax();
  float3 hitPos = { rayOrigin.x + tMax * rayDirection.x, rayOrigin.y + tMax * rayDirection.y, rayOrigin.z + tMax * rayDirection.z };
  glm::vec3 normal;
  glm::vec3 geometryNormal;
  uint3 launchIdx = optixGetLaunchIndex();
  glm::vec2 tc(0.f);
  if (sceneobject->mesh()) {
    const unsigned int primIdx = optixGetPrimitiveIndex();
    const float2       barycentrics = optixGetTriangleBarycentrics();
    const glm::uvec3& triangle = sceneobject->mesh()->ibo()[primIdx];
    const glm::vec3& N0 = sceneobject->mesh()->normals()[triangle.x];
    const glm::vec3& N1 = sceneobject->mesh()->normals()[triangle.y];
    const glm::vec3& N2 = sceneobject->mesh()->normals()[triangle.z];

     geometryNormal = glm::normalize((1.f - barycentrics.x - barycentrics.y) * N0 + barycentrics.x * N1 + barycentrics.y * N2);
     normal = geometryNormal;
     glm::vec3 interpolation = (1.f - barycentrics.x - barycentrics.y) * N0 + barycentrics.x * N1 + barycentrics.y * N2;


    const glm::vec2* tcs = sceneobject->mesh()->tcs();
    if (tcs) {
      const glm::vec2& TC0 = sceneobject->mesh()->tcs()[triangle.x];
      const glm::vec2& TC1 = sceneobject->mesh()->tcs()[triangle.y];
      const glm::vec2& TC2 = sceneobject->mesh()->tcs()[triangle.z];

      tc = (1.f - barycentrics.x - barycentrics.y) * TC0 + barycentrics.x * TC1 + barycentrics.y * TC2;
     
      if (sceneobject->material()) {
        const glm::vec3& P0 = sceneobject->mesh()->vbo()[triangle.x];
        const glm::vec3& P1 = sceneobject->mesh()->vbo()[triangle.y];
        const glm::vec3& P2 = sceneobject->mesh()->vbo()[triangle.z];

        const glm::vec3 P = (1.f - barycentrics.x - barycentrics.y) * P0 + barycentrics.x * P1 + barycentrics.y * P2;

        const glm::vec3 edge1 = P1 - P;
        const glm::vec3 edge2 = P2 - P;

        const glm::vec2 deltaUV1 = TC1 - tc;
        const glm::vec2 deltaUV2 = TC2 - tc;

        float denominator = (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y);
        if (denominator != 0.f) {
          float f = 1.f / denominator;
          glm::vec3 T;
          T.x = f * (deltaUV2.y * edge1.x - deltaUV1.y * edge2.x);
          T.y = f * (deltaUV2.y * edge1.y - deltaUV1.y * edge2.y);
          T.z = f * (deltaUV2.y * edge1.z - deltaUV1.y * edge2.z);

          glm::vec3 B;
          B.x = f * (-deltaUV2.x * edge1.x + deltaUV1.x * edge2.x);
          B.y = f * (-deltaUV2.x * edge1.y + deltaUV1.x * edge2.y);
          B.z = f * (-deltaUV2.x * edge1.z + deltaUV1.x * edge2.z);

          if (T != glm::vec3(0.f) && B != glm::vec3(0.f)) {
            normal = sceneobject->material()->normalmap(CCoordinateFrame::fromTBN(glm::normalize(T), glm::normalize(-B), geometryNormal), tc);
          }
        }


        //normal = sceneobject->material()->normalmap(geometryNormal, tc);
        //normal = sceneobject->material()->normalmap(CCoordinateFrame::fromNormal(geometryNormal), tc);
      }
    }

    normal = sceneobject->modelToWorld() * glm::vec4(normal, 0.f);
    geometryNormal = glm::normalize(sceneobject->modelToWorld() * glm::vec4(geometryNormal, 0.f));
  }
  else {
    geometryNormal = glm::normalize(glm::vec3(uint_as_float(optixGetAttribute_3()), uint_as_float(optixGetAttribute_4()), uint_as_float(optixGetAttribute_5())));
    normal = geometryNormal;
  }
  //geometryNormal = glm::normalize(geometryNormal);
  normal = glm::normalize(normal);

  si->hitInformation.hit = true;
  si->hitInformation.pos = glm::vec3(hitPos.x, hitPos.y, hitPos.z);
  si->hitInformation.normal = normal;
  si->hitInformation.normalG = geometryNormal;
  si->hitInformation.tc = tc;
  si->hitInformation.t = optixGetRayTmax();
  si->object = sceneobject;
  si->material = sceneobject->material();
  si->medium = sceneobject->medium();

}

extern "C" __global__ void __anyhit__mesh() {
  const rt::CDeviceSceneobject** sceneobjectPtr = reinterpret_cast<const rt::CDeviceSceneobject**>(optixGetSbtDataPointer());
  const rt::CDeviceSceneobject* sceneobject = *sceneobjectPtr;

  const unsigned int primIdx = optixGetPrimitiveIndex();
  const float2       barycentrics = optixGetTriangleBarycentrics();
  const glm::uvec3& triangle = sceneobject->mesh()->ibo()[primIdx];

  const glm::vec2* tcs = sceneobject->mesh()->tcs();
  if (tcs) {
    const glm::vec2& TC0 = sceneobject->mesh()->tcs()[triangle.x];
    const glm::vec2& TC1 = sceneobject->mesh()->tcs()[triangle.y];
    const glm::vec2& TC2 = sceneobject->mesh()->tcs()[triangle.z];

    glm::vec2 tc = (1.f - barycentrics.x - barycentrics.y) * TC0 + barycentrics.x * TC1 + barycentrics.y * TC2;

    if (!sceneobject->material()->opaque(tc)) {
      optixIgnoreIntersection();
    }
  }
}



extern "C" __global__ void __miss__ms() {
  unsigned int siAdress[2];
  siAdress[0] = optixGetPayload_0();
  siAdress[1] = optixGetPayload_1();

  SInteraction* si;
  memcpy(&si, siAdress, sizeof(SInteraction*));
  si->hitInformation.hit = false;
  si->object = nullptr;
  si->material = nullptr;
  si->medium = nullptr;
}


extern "C" __global__ void __intersection__surface() {
  const rt::CDeviceSceneobject** sceneobjectPtr = reinterpret_cast<const rt::CDeviceSceneobject**>(optixGetSbtDataPointer());
  const rt::CDeviceSceneobject* sceneobject = *sceneobjectPtr;

  float3 tempOrigin = optixGetWorldRayOrigin();
  float3 tempDirection = optixGetWorldRayDirection();
  rt::CRay ray(glm::vec3(tempOrigin.x, tempOrigin.y, tempOrigin.z), glm::vec3(tempDirection.x, tempDirection.y, tempDirection.z), optixGetRayTmax());
  rt::SInteraction si = sceneobject->intersect(ray);

  if (si.hitInformation.hit) {
    optixReportIntersection(si.hitInformation.t,
      0,
      float_as_uint(si.hitInformation.pos.x),
      float_as_uint(si.hitInformation.pos.y),
      float_as_uint(si.hitInformation.pos.z),
      float_as_uint(si.hitInformation.normal.x),
      float_as_uint(si.hitInformation.normal.y),
      float_as_uint(si.hitInformation.normal.z));
  }
}


extern "C" __global__ void __intersection__volume() {
  uint3 launchIdx = optixGetLaunchIndex();
  uint3 launchDim = optixGetLaunchDimensions();
  uint32_t samplerId = launchIdx.y * launchDim.x + launchIdx.x;
  CSampler& sampler = params.sampler[samplerId];
  const rt::CDeviceSceneobject** sceneobjectPtr = reinterpret_cast<const rt::CDeviceSceneobject**>(optixGetSbtDataPointer());
  const rt::CDeviceSceneobject* sceneobject = *sceneobjectPtr;

  float3 tempOrigin = optixGetWorldRayOrigin();
  float3 tempDirection = optixGetWorldRayDirection();
  glm::vec3 rayOrigin(tempOrigin.x, tempOrigin.y, tempOrigin.z);
  glm::vec3 rayDirection(tempDirection.x, tempDirection.y, tempDirection.z);

  const CMediumInstance* medium = sceneobject->medium();
  const nanovdb::BBox<nanovdb::Vec3f>& aabb = reinterpret_cast<const nanovdb::BBox<nanovdb::Vec3f>&>(medium->worldBB());
  float t0 = 0.f;
  float t1 = CRay::DEFAULT_TMAX;
  float initialT1 = t1;

  nanovdb::Ray<float> ray = nanovdb::Ray<float>(reinterpret_cast<const nanovdb::Vec3f&>(tempOrigin), reinterpret_cast<const nanovdb::Vec3f&>(tempDirection), t0, t1);
  bool intersects = ray.intersects(aabb, t0, t1);
  if (intersects && t1 < initialT1) {
    float t;
    nanovdb::Vec3f nanovdbOrigin(tempOrigin.x, tempOrigin.y, tempOrigin.z);
    if (aabb.isInside(nanovdbOrigin)) {// When ray starts in medium it seems that t1 gives first intersection while t0 remaining unchanged
      t = t1;
    }
    else {
      t = t0;
    }

    glm::vec3 intersectionPos = rayOrigin + t * rayDirection;
    optixReportIntersection(
      t,
      0,
      float_as_uint(intersectionPos.x),
      float_as_uint(intersectionPos.y),
      float_as_uint(intersectionPos.z),
      float_as_uint(0.f),
      float_as_uint(0.f),
      float_as_uint(0.f)
    );

  }
}



extern "C" __global__ void __raygen__rg() {
  uint3 launchIdx = optixGetLaunchIndex();
  uint3 launchDim = optixGetLaunchDimensions();

  uint32_t samplerId = launchIdx.y * launchDim.x + launchIdx.x;

  CPixelSampler pixelSampler(params.camera, launchIdx.x, launchIdx.y, &(params.sampler[samplerId]));
  //CPathIntegrator integrator((CDeviceScene*)sharedScene, &pixelSampler, &(sampler[samplerId]), numSamples);
  CPathIntegrator integrator(params.scene, &pixelSampler, &(params.sampler[samplerId]), params.numSamples);
  glm::vec3 L = integrator.Li();

  uint32_t currentPixel = params.bpp * (launchIdx.y * launchDim.x + launchIdx.x);

  params.data[currentPixel + 0] += L.x;
  params.data[currentPixel + 1] += L.y;
  params.data[currentPixel + 2] += L.z;
}

extern "C" __global__ void __raygen__filtering() {
  uint3 launchIdx = optixGetLaunchIndex();
  uint3 launchDim = optixGetLaunchDimensions();

  size_t id = launchIdx.x + launchIdx.y * launchDim.x + launchIdx.z * launchDim.x * launchDim.y;
  CSampler& sampler = paramsFiltering.samplers[id];
  CMeshFilter filter(reinterpret_cast<glm::ivec3&>(optixGetLaunchIndex()),
                     paramsFiltering.indexToModel,
                     paramsFiltering.modelToIndex,
                     paramsFiltering.modelToWorld,
                     paramsFiltering.worldToModel,
                     paramsFiltering.numVoxels,
                     paramsFiltering.worldBB,
                     sampler,
                     paramsFiltering.sigma_t,
                     paramsFiltering.estimationIterations,
                     paramsFiltering.alpha,
                     paramsFiltering.clipRays,
                     paramsFiltering.scaling);
  if (paramsFiltering.debug) {
    filter.debug(*paramsFiltering.scene, paramsFiltering.debugSamples);
  }
  else {
      SFilteredDataCompact filteredData = filter.run(*paramsFiltering.scene, paramsFiltering.samplesPerVoxel);
      paramsFiltering.filteredData[id] = filteredData;
  }
}
