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


using namespace rt;

__constant__ rt::SLaunchParams params;


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
  uint3 launchIdx = optixGetLaunchIndex();
  glm::vec2 tc(0.f);
  if (sceneobject->mesh()) {
    const unsigned int primIdx = optixGetPrimitiveIndex();
    const float2       barycentrics = optixGetTriangleBarycentrics();
    const glm::uvec3& triangle = sceneobject->mesh()->ibo()[primIdx];
    const glm::vec3& N0 = sceneobject->mesh()->normals()[triangle.x];
    const glm::vec3& N1 = sceneobject->mesh()->normals()[triangle.y];
    const glm::vec3& N2 = sceneobject->mesh()->normals()[triangle.z];
    //if (barycentrics.x > 1.f || barycentrics.x < 0.f || barycentrics.y > 1.f || barycentrics.y < 0.f) {
    //  printf("Barycentrics: (%f, %f)\n", barycentrics.x, barycentrics.y);
    //}

    normal = (1.f - barycentrics.x - barycentrics.y) * N0 + barycentrics.x * N1 + barycentrics.y * N2;

    const glm::vec2* tcs = sceneobject->mesh()->tcs();
    if (tcs) {
      const glm::vec2& TC0 = sceneobject->mesh()->tcs()[triangle.x];
      const glm::vec2& TC1 = sceneobject->mesh()->tcs()[triangle.y];
      const glm::vec2& TC2 = sceneobject->mesh()->tcs()[triangle.z];

      tc = (1.f - barycentrics.x - barycentrics.y) * TC0 + barycentrics.x * TC1 + barycentrics.y * TC2;
      //if (tc.x < 0.f || tc.x > 1.f || tc.y < 0.f || tc.y > 1.f) {
      //  printf("ch tc: (%f, %f)\n", tc.x, tc.y);
      //}
    }
  }
  else {
    normal = glm::vec3( uint_as_float(optixGetAttribute_3()), uint_as_float(optixGetAttribute_4()), uint_as_float(optixGetAttribute_5()) );
  }

  si->hitInformation.hit = true;
  si->hitInformation.pos = glm::vec3(hitPos.x, hitPos.y, hitPos.z);
  si->hitInformation.normal = glm::normalize(normal);
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

  const CNVDBMedium* medium = static_cast<const CNVDBMedium*>(sceneobject->medium());
  const nanovdb::BBox<nanovdb::Vec3R>& aabb = medium->grid()->worldBBox();
  float t0 = 0.f;
  float t1 = CRay::DEFAULT_TMAX;
  float initialT1 = t1;

  nanovdb::Ray<float> ray = nanovdb::Ray<float>(reinterpret_cast<const nanovdb::Vec3f&>(tempOrigin), reinterpret_cast<const nanovdb::Vec3f&>(tempDirection), t0, t1);
  bool intersects = ray.intersects(aabb, t0, t1);
  if (intersects && t1 < initialT1) {
    float t;
    nanovdb::Vec3R nanovdbOrigin(tempOrigin.x, tempOrigin.y, tempOrigin.z);
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
