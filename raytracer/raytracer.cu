
#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

#ifdef __INTELLISENSE__
void __syncthreads(); // This avoids that usages of __syncthreads are underlined red
#endif

#include "raytracer.hpp"

#include "sampling/sampler.hpp"

#include "utility/performance_monitoring.hpp"
#include "integrators/direct_lighting_integrator.hpp"
#include "integrators/path_integrator.hpp"
#include "camera/pixel_sampler.hpp"
#include "scene/environmentmap.hpp"
#include "utility/qualifiers.hpp"
#include "utility/debugging.hpp"
#include "medium/homogeneous_medium.hpp"
#include "shapes/circle.hpp"
#include "shapes/sphere.hpp"
#include "shapes/cuboid.hpp"
#include "medium/heterogenous_medium.hpp"
#include <random>
#include "medium/nvdb_medium.hpp"
#include "medium/sggx_phase_function.hpp"

namespace rt {
  // Initializes cuRAND random number generators
  __global__ void init(CSampler* sampler, SDeviceFrame* frame) {
    uint16_t y = blockIdx.y;
    uint16_t x = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t samplerId = y * frame->width + x;
    sampler[samplerId].init(samplerId, 0);
  }

  //struct SSharedMemoryInitializer {
  //  D_CALLABLE static void copyScene(void* sharedScene, CDeviceScene* scene) {
  //    if (threadIdx.x == 0) {
  //      char* sharedPtr = (char*)sharedScene;
  //      memcpy(sharedPtr, scene, sizeof(CDeviceScene));
  //      ((CDeviceScene*)sharedPtr)->m_sceneobjects = (CDeviceSceneobject*)(sharedPtr + sizeof(CDeviceScene)); // Sceneobjects are located behind CDeviceScene
  //      memcpy(((CDeviceScene*)sharedPtr)->m_sceneobjects, scene->m_sceneobjects, scene->m_numSceneobjects * sizeof(CDeviceSceneobject)); // Copy value of sceneobjects
  //      uint32_t shapeMaxSize = glm::max(glm::max(glm::max(sizeof(CCircle), sizeof(CCuboid)), sizeof(CRectangle)), sizeof(Sphere));
  //      uint32_t propertyMaxSize = glm::max(sizeof(CMaterial), sizeof(CMedium));
  //      uint32_t dataSize = shapeMaxSize + propertyMaxSize;
  //      for (size_t offset = 0; offset < scene->m_numSceneobjects; ++offset) {
  //        CDeviceSceneobject* src = scene->m_sceneobjects + offset;
  //        CDeviceSceneobject* dst = ((CDeviceScene*)sharedPtr)->m_sceneobjects + offset;
  //        CShape* shapeDst = (CShape*)((char*)((CDeviceScene*)sharedPtr)->m_sceneobjects + scene->m_numSceneobjects * sizeof(CDeviceSceneobject) + offset * dataSize); // Shapes and materials/medium properties are stored after all scene objects
  //        if (src->m_shape) {
  //          switch (src->m_shape->shape()) {
  //          case EShape::CIRCLE:
  //            memcpy(shapeDst, src->m_shape, sizeof(CCircle));
  //            break;
  //          case EShape::CUBOID:
  //            memcpy(shapeDst, src->m_shape, sizeof(CCuboid));
  //            break;
  //          case EShape::RECTANGLE:
  //            memcpy(shapeDst, src->m_shape, sizeof(CRectangle));
  //            break;
  //          case EShape::SPHERE:
  //            memcpy(shapeDst, src->m_shape, sizeof(Sphere));
  //            break;
  //          }
  //        }

  //        if (src->m_material) {
  //          CMaterial* materialDst = (CMaterial*)(shapeDst + 1);
  //          memcpy(materialDst, src->m_material, sizeof(CMaterial));
  //        }
  //        else if (src->m_medium) {
  //          CMedium* mediumDst = (CMedium*)(shapeDst + 1);
  //          memcpy(mediumDst, src->m_medium, sizeof(CMedium));
  //        }
  //      }
  //    }
  //    __syncthreads();
  //  }
  //};

  // Raytracing
  __global__ void renderFrame(CDeviceScene* scene, CCamera* camera, CSampler* sampler, uint16_t numSamples, SDeviceFrame* frame) {
    uint16_t y = blockIdx.y;
    uint16_t x = blockIdx.x * blockDim.x + threadIdx.x;


    if (y < frame->height && x < frame->width) {
      //extern __shared__ char sharedScene[];
      //SSharedMemoryInitializer::copyScene(sharedScene, scene);

      uint32_t currentPixel = frame->bpp * (y * frame->width + x);
      uint32_t samplerId = y * frame->width + x;

      CPixelSampler pixelSampler(camera, x, y, &(sampler[samplerId]));
      //CPathIntegrator integrator((CDeviceScene*)sharedScene, &pixelSampler, &(sampler[samplerId]), numSamples);
      CPathIntegrator integrator(scene, &pixelSampler, &(sampler[samplerId]), numSamples);
      glm::vec3 L = integrator.Li();

      frame->data[currentPixel + 0] += L.r;
      frame->data[currentPixel + 1] += L.g;
      frame->data[currentPixel + 2] += L.b;

    }
  }

  D_CALLABLE inline float computeTonemapFactor(SDeviceFrame* frame, uint16_t x, uint16_t y) {
    constexpr uint8_t filterSize = 11;
    float filterHalf = (float)filterSize / 2;
    float alpha = -glm::log(0.5f) / (filterHalf * filterHalf); // 0.02: From webers law
    float weights[filterSize][filterSize];
    float sum = 0.f;
    for (int8_t dX = 0; dX < filterSize; ++dX) {
      for (int8_t dY = 0; dY < filterSize; ++dY) {
        int32_t currX = x + dX - filterHalf;
        int32_t currY = y + dY - filterHalf;
        if (currX < 0 || currX >= frame->width || currY < 0 || currY >= frame->height) {
          weights[dY][dX] = 0.f;
        }
        else {
          float distance = (float)dX * dX + (float)dY * dY;
          float weight = glm::exp(-alpha * distance);
          sum += weight;
          weights[dY][dX] = weight;
        }
      }
    }

    float sigma(0.f);
    for (int8_t dX = 0; dX < filterSize; ++dX) {
      for (int8_t dY = 0; dY < filterSize; ++dY) {
        int32_t currX = x + dX - filterHalf;
        int32_t currY = y + dY - filterHalf;
        if (!(currX < 0 || currX >= frame->width || currY < 0 || currY >= frame->height)) {
          uint32_t currentPixel = frame->bpp * (currY * frame->width + currX);

          float r = frame->data[currentPixel + 0];
          float g = frame->data[currentPixel + 1];
          float b = frame->data[currentPixel + 2];
          sigma += glm::log(r + g + b) * weights[dY][dX] / sum;
        }
        
      }
    }

    return glm::exp(sigma);
  }

  __global__ void filterFrame(SDeviceFrame* frame) {
    uint16_t y = blockIdx.y;
    uint16_t x = blockIdx.x * blockDim.x + threadIdx.x;

    if (y < frame->height && x < frame->width) {
      uint32_t currentPixel = frame->bpp * (y * frame->width + x);
      float sigma = computeTonemapFactor(frame, x, y);
      
      frame->filtered[currentPixel + 0] = sigma;
    }
  }

  __global__ void computeGlobalTonemapping1(SDeviceFrame* frame, float* avg) {
    uint16_t y = 0;
    uint16_t x = blockIdx.x * blockDim.x + threadIdx.x;

    if (y < frame->height && x < frame->width) {
      float divisor = frame->height * frame->width * frame->bpp;
      avg[x] = 0.f;
      for (uint16_t yIter = y; yIter < frame->height; ++yIter) {
          uint32_t currentPixel = frame->bpp * (yIter * frame->width + x);
          avg[x] += glm::log(frame->data[currentPixel + 0] + frame->data[currentPixel + 1] + frame->data[currentPixel + 2] + FLT_MIN) / divisor;
      }
    }
  }

  __global__ void computeGlobalTonemapping2(SDeviceFrame* frame, float* avg, float* tonemappingFactor) {
    float result = 0.f;
    for (uint16_t i = 0; i < frame->width; ++i) {
      result += avg[i];
    }
    *tonemappingFactor = glm::exp(result) - (frame->width * frame->height * FLT_MIN);
  }

  // Map colors to [0.0f, 1.0f]
  __global__ void applyTonemapping(SDeviceFrame* frame, float* tonemapFactor) {
    uint16_t y = blockIdx.y;
    uint16_t x = blockIdx.x * blockDim.x + threadIdx.x;

    if (y < frame->height && x < frame->width) {
      uint32_t currentPixel = frame->bpp * (y * frame->width + x);
    
      float r = frame->data[currentPixel + 0];
      float g = frame->data[currentPixel + 1];
      float b = frame->data[currentPixel + 2];

      float sigma = frame->filtered[currentPixel + 0];

      frame->data[currentPixel + 0] = r / (r + *tonemapFactor);
      frame->data[currentPixel + 1] = g / (g + *tonemapFactor);
      frame->data[currentPixel + 2] = b / (b + *tonemapFactor);
    }
  }

  

  // Corrects nonliniar monitor output
  __global__ void correctGamma(SDeviceFrame* frame, float gamma) {
    uint16_t y = blockIdx.y;
    uint16_t x = blockIdx.x * blockDim.x + threadIdx.x;

    if (y < frame->height && x < frame->width) {
      uint32_t currentPixel = frame->bpp * (y * frame->width + x);

      float r = frame->data[currentPixel + 0];
      float g = frame->data[currentPixel + 1];
      float b = frame->data[currentPixel + 2];

      frame->data[currentPixel + 0] = glm::pow(r, 1 / gamma);
      frame->data[currentPixel + 1] = glm::pow(g, 1 / gamma);
      frame->data[currentPixel + 2] = glm::pow(b, 1 / gamma);
    }
  }

  // Maps [0.0f, 1.0f] to [0, 255], required for jpg/png output
  __global__ void fillByteFrame(SDeviceFrame* frame) {
    uint16_t y = blockIdx.y;
    uint16_t x = blockIdx.x * blockDim.x + threadIdx.x;

    if (y < frame->height && x < frame->width) {
      uint32_t currentPixel = frame->bpp * (y * frame->width + x);

      frame->dataBytes[currentPixel + 0] = glm::round(frame->data[currentPixel + 0] * 255.0f);
      frame->dataBytes[currentPixel + 1] = glm::round(frame->data[currentPixel + 1] * 255.0f);
      frame->dataBytes[currentPixel + 2] = glm::round(frame->data[currentPixel + 2] * 255.0f);
    }
  }

  Raytracer::Raytracer(uint16_t frameWidth, uint16_t frameHeight) :
    m_frameWidth(frameWidth),
    m_frameHeight(frameHeight),
    m_bpp(3),
    m_scene(),
    //m_hostCamera(frameWidth, frameHeight, 90, glm::vec3(-0.5f, 0.2f, 0.5f), glm::vec3(0.0f, 0.1f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f)),
    m_hostCamera(frameWidth, frameHeight, 90, glm::vec3(-450.f, 73.f, 450.f), glm::vec3(-10.f, 73.f, -43.f), glm::vec3(0.0f, 1.0f, 0.0f)),
    //m_hostCamera(frameWidth, frameHeight, 160, glm::vec3(0.10f, 0.15f, 0.01f), glm::vec3(0.0f, 0.1f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f)),
    m_numSamples(32), // higher -> less noise
    m_tonemappingFactor(100.f),
    m_gamma(2.0f),
    m_deviceCamera(nullptr),
    m_deviceFrameData(nullptr),
    m_deviceSampler(nullptr),
    m_blockSize(128) {
    // Add scene objects
    m_scene.addSceneobject(CHostSceneobject(new CCircle(glm::vec3(0.0f, 0.0f, 0.0f), FLT_MAX, glm::vec3(0.0f, 1.0f, 0.0f)), glm::vec3(0.3f, 0.3f, 0.3f), 0.99f, glm::vec3(0.1f), 0.99f, 0.99f, 1.00029f, 1.2f));
    //m_scene.addSceneobject(CHostSceneobject(new CRectangle(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec2(0.5f, 0.2f), glm::vec3(0.0f, 1.0f, 0.0f)), glm::vec3(0.3f, 0.3f, 0.3f), 0.99f, glm::vec3(0.1f), 0.99f, 0.99f, 1.00029f, 1.2f));
    float lightness = 0.75f / 255.0f;
    //m_scene.addSceneobject(CHostSceneobject(EShape::SPHERE, getSpherePosition(0.08f, 0, 6), 0.08f, glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(lightness, lightness, 0.85f), 0.01f, glm::vec3(0.9f), 0.01f, 0.01f, 1.00029f, 1.5f)); // blue sphere
    //m_scene.addSceneobject(CHostSceneobject(EShape::SPHERE, getSpherePosition(0.08f, 1, 6), 0.08f, glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.85f, lightness, 0.85f), 0.01f,  glm::vec3(0.9f), 0.01f, 0.01f, 1.00029f, 1.5f)); // violet sphere
    //m_scene.addSceneobject(CHostSceneobject(EShape::SPHERE, getSpherePosition(0.08f, 2, 6), 0.08f, glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.85f, lightness, lightness), 0.01f,  glm::vec3(0.9f), 0.01f, 0.01f, 1.00029f, 1.5f)); // red sphere
    //m_scene.addSceneobject(CHostSceneobject(EShape::SPHERE, getSpherePosition(0.08f, 3, 6), 0.08f, glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.85f, 0.85f, lightness), 0.01f,  glm::vec3(0.9f), 0.01f, 0.01f, 1.00029f, 1.5f)); // yellow sphere
    //m_scene.addSceneobject(CHostSceneobject(EShape::SPHERE, getSpherePosition(0.08f, 4, 6), 0.08f, glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(lightness, 0.85f, lightness), 0.01f,  glm::vec3(0.9f), 0.01f, 0.01f, 1.00029f, 1.5f)); // green sphere
    //m_scene.addSceneobject(CHostSceneobject(EShape::SPHERE, getSpherePosition(0.08f, 5, 6), 0.08f, glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(lightness, 0.85f, 0.85f), 0.01f, glm::vec3(0.9f), 0.01f, 0.01f, 1.00029f, 1.5f)); // cyan sphere
    //m_scene.addSceneobject(CHostSceneobject(new CRectangle(getSpherePosition(0.08f, 0, 6), glm::vec2(0.08f), -glm::normalize(getSpherePosition(0.08f, 0, 6))), glm::vec3(lightness, lightness, 0.85f), 0.01f, glm::vec3(0.9f), 0.01f, 0.01f, 1.00029f, 1.5f)); // blue sphere
    //m_scene.addSceneobject(CHostSceneobject(new CRectangle(getSpherePosition(0.08f, 1, 6), glm::vec2(0.08f), -glm::normalize(getSpherePosition(0.08f, 1, 6))), glm::vec3(0.85f, lightness, 0.85f), 0.01f,  glm::vec3(0.9f), 0.01f, 0.01f, 1.00029f, 1.5f)); // violet sphere
    //m_scene.addSceneobject(CHostSceneobject(new CRectangle(getSpherePosition(0.08f, 2, 6), glm::vec2(0.08f), -glm::normalize(getSpherePosition(0.08f, 2, 6))), glm::vec3(0.85f, lightness, lightness), 0.01f,  glm::vec3(0.9f), 0.01f, 0.01f, 1.00029f, 1.5f)); // red sphere
    //m_scene.addSceneobject(CHostSceneobject(new CRectangle(getSpherePosition(0.08f, 3, 6), glm::vec2(0.08f), -glm::normalize(getSpherePosition(0.08f, 3, 6))), glm::vec3(0.85f, 0.85f, lightness), 0.01f,  glm::vec3(0.9f), 0.01f, 0.01f, 1.00029f, 1.5f)); // yellow sphere
    //m_scene.addSceneobject(CHostSceneobject(new CRectangle(getSpherePosition(0.08f, 4, 6), glm::vec2(0.08f), -glm::normalize(getSpherePosition(0.08f, 4, 6))), glm::vec3(lightness, 0.85f, lightness), 0.01f,  glm::vec3(0.9f), 0.01f, 0.01f, 1.00029f, 1.5f)); // green sphere
    //m_scene.addSceneobject(CHostSceneobject(new CRectangle(getSpherePosition(0.08f, 5, 6), glm::vec2(0.08f), -glm::normalize(getSpherePosition(0.08f, 5, 6))), glm::vec3(lightness, 0.85f, 0.85f), 0.01f, glm::vec3(0.9f), 0.01f, 0.01f, 1.00029f, 1.5f)); // cyan sphere
    //m_scene.addSceneobject(CHostSceneobject(new Sphere(glm::vec3(0.f, 0.15f, 0.0f), 0.15f, glm::vec3(0.0f, 1.0f, 0.0f)), new CHomogeneousMedium(glm::vec3(5.f, 3.f, 3.f), glm::vec3(5.f, 3.f, 3.f), 0.0f))); // volume
    //m_scene.addSceneobject(CHostSceneobject(new CCuboid(glm::vec3(0.f, 0.15f, 0.0f), glm::vec3(0.25f), glm::vec3(0.0f, 1.0f, 0.0f)), new CHomogeneousMedium(glm::vec3(5.f, 3.f, 3.f), glm::vec3(5.f, 3.f, 3.f), 0.0f))); // volume
    //std::random_device rd{};
    //std::mt19937 gen{ rd() };

    //constexpr size_t oneDimSize = 100;
    //constexpr size_t volumeSize = oneDimSize * oneDimSize * oneDimSize;
    //std::vector<float> d;
    //d.reserve(volumeSize);
    //for (size_t i = 0; i < volumeSize; ++i) {
    //  //uint16_t x = i % oneDimSize;
    //  //uint16_t y = (i / oneDimSize) % oneDimSize;
    //  //uint16_t z = i / (oneDimSize * oneDimSize);
    //  //float xf = (float)x / oneDimSize;
    //  //float yf = (float)y / oneDimSize;
    //  //float zf = (float)z / oneDimSize;
    //  //glm::vec3 origin((float)((oneDimSize - 1) / 2.f) / oneDimSize);
    //  //glm::vec3 pos((float)xf, (float)yf, (float)zf);
    //  //float dist = glm::length(pos - origin);
    //  //std::normal_distribution<> nd{ 1.f / glm::max(FLT_MIN, dist), 1.f / glm::max(FLT_MIN, dist) };
    //  //std::normal_distribution<> nd{ glm::length(origin) - dist, glm::length(origin) - dist };
    //  std::normal_distribution<> nd{ 0.f, 1.f };
    //  d.push_back(glm::abs(nd(gen)));
    //  //d[i] = 1.f;
    //}
    //glm::vec3 volumePos(-449.5f, 0.15f, 449.5f);
    //glm::vec3 volumePos(-10.f, 73.f, -43.f);
    //glm::vec3 size(0.25f);
    //glm::vec3 size(200.f);
    //m_scene.addSceneobject(CHostSceneobject(new CCuboid(volumePos, size, glm::vec3(0.0f, 1.0f, 0.0f)), new CHeterogenousMedium(glm::vec3(0.f, 0.f, 0.f), glm::vec3(3.f, 3.f, 3.f), -0.5f, 10, 10, 10, volumePos, size, d.data()))); // volume
    //m_scene.addSceneobject(CHostSceneobject(new CNVDBMedium("../../raytracer/assets/wdas_cloud/wdas_cloud_sixteenth.nvdb", glm::vec3(0.f, 0.f, 0.f), glm::vec3(50.f, 50.f, 50.f), 0.f))); // volume
    //m_scene.addSceneobject(CHostSceneobject(new CNVDBMedium("../../raytracer/assets/wdas_cloud/wdas_cloud_sixteenth.nvdb", glm::vec3(0.f, 0.f, 0.f), glm::vec3(50.f, 50.f, 50.f), SSGGXDistributionParameters{2.f, 2.f, 2.f, -1.f, 0.f, -1.f}, SSGGXDistributionParameters{ 2.f, 2.f, 2.f, -1.f, 0.f, -1.f }))); // volume SGGX
    m_scene.addSceneobject(CHostSceneobject(new CCuboid(glm::vec3(-400.f, 50.f, 300.f), glm::vec3(20.f), glm::vec3(0.f, 1.f, 0.f)), glm::vec3(lightness, lightness, 0.85f), 0.01f, glm::vec3(0.9f), 0.01f, 0.01f, 1.00029f, 1.5f)); // as normal reference
    m_scene.addSceneobject(CHostSceneobject(new CNVDBMedium("../../raytracer/assets/wdas_cloud/wdas_cloud_sixteenth.nvdb", glm::vec3(0.f, 0.f, 0.f), glm::vec3(50.f, 50.f, 50.f), 1.f, 0.0001f))); // volume SGGX
    //m_scene.addLightsource(CHostSceneobject(EShape::PLANE, glm::vec3(0.0f, 0.3f, 0.0f), 0.2f, glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(1.0f))); // Light
    //glm::vec3 light1Pos = getSpherePosition(0.1f, 0, 6) + glm::vec3(0.0f, 0.2f, 0.0f);
    //m_scene.addLightsource(CHostSceneobject(EShape::PLANE, light1Pos, 0.05f, -glm::normalize(light1Pos), glm::vec3(10.0f)));
    //glm::vec3 light2Pos = getSpherePosition(0.1f, 1, 6) + glm::vec3(0.0f, 0.2f, 0.0f);
    //m_scene.addLightsource(CHostSceneobject(EShape::PLANE, light2Pos, 0.05f, -glm::normalize(light2Pos), glm::vec3(10.0f)));
    //glm::vec3 light3Pos = getSpherePosition(0.1f, 2, 6) + glm::vec3(0.0f, 0.2f, 0.0f);
    //m_scene.addLightsource(CHostSceneobject(EShape::PLANE, light3Pos, 0.05f, -glm::normalize(light3Pos), glm::vec3(10.0f)));
    //glm::vec3 light4Pos = getSpherePosition(0.1f, 3, 6) + glm::vec3(0.0f, 0.2f, 0.0f);
    //m_scene.addLightsource(CHostSceneobject(EShape::PLANE, light4Pos, 0.05f, -glm::normalize(light4Pos), glm::vec3(10.0f)));
    //glm::vec3 light5Pos = getSpherePosition(0.1f, 4, 6) + glm::vec3(0.0f, 0.2f, 0.0f);
    //m_scene.addLightsource(CHostSceneobject(EShape::PLANE, light5Pos, 0.05f, -glm::normalize(light5Pos), glm::vec3(10.0f)));
    //glm::vec3 light6Pos = getSpherePosition(0.1f, 5, 6) + glm::vec3(0.0f, 0.2f, 0.0f);
    //m_scene.addLightsource(CHostSceneobject(EShape::PLANE, light6Pos, 0.05f, -glm::normalize(light6Pos), glm::vec3(10.0f)));

    //glm::vec3 light2Pos = getSpherePosition(0.1f, 1, 6) + glm::vec3(0.0f, 0.1f, 0.0f);
    //m_scene.addLightsource(CHostSceneobject(EShape::PLANE, light2Pos, 0.05f, -glm::normalize(light2Pos), glm::vec3(2.0f, 0.0f, 0.5f)));
    //glm::vec3 light3Pos = getSpherePosition(0.1f, 2, 6) + glm::vec3(0.0f, 0.1f, 0.0f);
    //m_scene.addLightsource(CHostSceneobject(EShape::PLANE, light3Pos, 0.05f, -glm::normalize(light3Pos), glm::vec3(2.0f)));

    // Add environment map
    //m_scene.setEnvironmentMap(CEnvironmentMap("./../../raytracer/assets/sunflowers_1k_edit.hdr"));
    m_scene.setEnvironmentMap(CEnvironmentMap("./../../raytracer/assets/envmap.hdr"));

    allocateDeviceMemory();
    copyToDevice();
    initDeviceData();
  }

  Raytracer::~Raytracer() {
    freeDeviceMemory();
  }

  // Renderpipeline
  SFrame Raytracer::renderFrame() {
    CDeviceScene* scene = m_scene.deviceScene();
    dim3 grid(m_frameWidth / m_blockSize, m_frameHeight);
    uint32_t shapeMaxSize = glm::max(glm::max(glm::max(sizeof(CCircle), sizeof(CCuboid)), sizeof(CRectangle)), sizeof(Sphere));
    uint32_t propertyMaxSize = glm::max(sizeof(CMaterial), sizeof(CMedium));
    uint32_t dataSize = shapeMaxSize + propertyMaxSize;
    for (uint16_t sample = 0; sample < m_numSamples; ++sample) {
      std::cout << "Sample " << sample + 1 << "/" << m_numSamples << std::endl;
      //CPerformanceMonitoring::startMeasurement("renderFrame");
      //rt::renderFrame << <grid, m_blockSize, sizeof(CDeviceScene) + m_scene.sceneobjects().size() * (sizeof(CDeviceSceneobject) + dataSize) >> > (scene, m_deviceCamera, m_deviceSampler, m_numSamples, m_deviceFrame);
      rt::renderFrame << <grid, m_blockSize >> > (scene, m_deviceCamera, m_deviceSampler, m_numSamples, m_deviceFrame);
      GPU_ASSERT(cudaDeviceSynchronize());
      //CPerformanceMonitoring::endMeasurement("renderFrame");
    }
    rt::filterFrame << <grid, m_blockSize >> > (m_deviceFrame);
    GPU_ASSERT(cudaDeviceSynchronize());

    dim3 reductionGrid(m_frameWidth / m_blockSize, 1);;
    rt::computeGlobalTonemapping1 << <reductionGrid, m_blockSize >> > (m_deviceFrame, m_deviceAverage);
    GPU_ASSERT(cudaDeviceSynchronize());

    rt::computeGlobalTonemapping2 << <1, 1 >> > (m_deviceFrame, m_deviceAverage, m_deviceTonemappingValue);
    GPU_ASSERT(cudaDeviceSynchronize());

    //CPerformanceMonitoring::startMeasurement("applyTonemapping");
    //rt::applyTonemapping << <grid, m_blockSize >> > (m_deviceFrame, m_tonemappingFactor);
    rt::applyTonemapping << <grid, m_blockSize >> > (m_deviceFrame, m_deviceTonemappingValue);
    GPU_ASSERT(cudaDeviceSynchronize());
    //CPerformanceMonitoring::endMeasurement("applyTonemapping");

    //CPerformanceMonitoring::startMeasurement("correctGamma");
    rt::correctGamma << <grid, m_blockSize >> > (m_deviceFrame, m_gamma);
    GPU_ASSERT(cudaDeviceSynchronize());
    //CPerformanceMonitoring::endMeasurement("correctGamma");

    //CPerformanceMonitoring::startMeasurement("fillByteFrame");
    rt::fillByteFrame << <grid, m_blockSize >> > (m_deviceFrame);
    GPU_ASSERT(cudaDeviceSynchronize());
    //CPerformanceMonitoring::endMeasurement("fillByteFrame");

    SFrame frame = retrieveFrame();
    return frame;
  }

  // Distributes N spheres evenly around circle
  glm::vec3 Raytracer::getSpherePosition(float sphereRadius, uint8_t index, uint8_t maxSpheres) {
    float x = 4.0f * sphereRadius * std::cos(2 * M_PI / maxSpheres * index);
    float z = -4.0f * sphereRadius * std::sin(2 * M_PI / maxSpheres * index);
    float y = sphereRadius;
    return glm::vec3(x, y, z);
  }

  void Raytracer::allocateDeviceMemory() {
    m_scene.allocateDeviceMemory();
    cudaMalloc(&m_deviceSampler, sizeof(CSampler) * m_frameWidth * m_frameHeight);
    cudaMalloc(&m_deviceCamera, sizeof(CCamera));
    cudaMalloc(&m_deviceFrame, sizeof(SDeviceFrame));
    cudaMalloc(&m_deviceFrameData, sizeof(float)*m_hostCamera.sensorWidth()*m_hostCamera.sensorHeight()*m_bpp);
    cudaMalloc(&m_deviceFilteredFrame, sizeof(float)*m_hostCamera.sensorWidth()*m_hostCamera.sensorHeight()*m_bpp);
    cudaMalloc(&m_deviceFrameDataBytes, sizeof(uint8_t)*m_hostCamera.sensorWidth()*m_hostCamera.sensorHeight()*m_bpp);
    cudaMalloc(&m_deviceAverage, sizeof(float)*m_frameWidth);
    cudaMalloc(&m_deviceTonemappingValue, sizeof(float));
  }

  void Raytracer::copyToDevice() {
    m_scene.copyToDevice();
    CCamera deviceCamera = m_hostCamera;
    cudaMemcpy(m_deviceCamera, &deviceCamera, sizeof(CCamera), cudaMemcpyHostToDevice);
    
    SDeviceFrame f;
    f.width = m_hostCamera.sensorWidth();
    f.height = m_hostCamera.sensorHeight();
    f.bpp = m_bpp;
    f.data = m_deviceFrameData;
    f.filtered = m_deviceFilteredFrame;
    f.dataBytes = m_deviceFrameDataBytes;
    cudaMemcpy(m_deviceFrame, &f, sizeof(SDeviceFrame), cudaMemcpyHostToDevice);
  }

  void Raytracer::initDeviceData() {
    //CPerformanceMonitoring::startMeasurement("init");
    dim3 grid(m_frameWidth / m_blockSize, m_frameHeight);
    init << <grid, m_blockSize >> > (m_deviceSampler, m_deviceFrame);
    cudaError_t e = cudaDeviceSynchronize();
    //CPerformanceMonitoring::endMeasurement("init");
  }

  void Raytracer::freeDeviceMemory() {
    m_scene.freeDeviceMemory();
    cudaFree(m_deviceCamera);
    cudaFree(m_deviceFrameData);
    cudaFree(m_deviceFrame);
    cudaFree(m_deviceAverage);
    cudaFree(m_deviceTonemappingValue);
  }
  SFrame Raytracer::retrieveFrame() const {
    SFrame frame;
    uint32_t entries = m_frameWidth * m_frameHeight * m_bpp;
    frame.width = m_frameWidth;
    frame.height = m_frameHeight;
    frame.bpp = m_bpp;
    frame.data.resize(entries);
    cudaMemcpy(frame.data.data(), m_deviceFrameData, entries * sizeof(float), cudaMemcpyDeviceToHost);
    frame.dataBytes.resize(entries);
    cudaMemcpy(frame.dataBytes.data(), m_deviceFrameDataBytes, entries * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    return frame;
  }
}