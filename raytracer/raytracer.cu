
#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

#include "raytracer.hpp"

#include "sampling/sampler.hpp"

#include "utility/performance_monitoring.hpp"
#include "integrators/direct_lighting_integrator.hpp"
#include "integrators/path_integrator.hpp"
#include "camera/pixel_sampler.hpp"
#include "scene/environmentmap.hpp"
#include "utility/qualifiers.hpp"
#include "utility/debugging.hpp"
#include "shapes/circle.hpp"
#include "shapes/cuboid.hpp"
#include "medium/nvdb_medium.hpp"
#include "backend/rt_backend.hpp"
#include <optix/optix_stubs.h>
#include "texture/texture_manager.hpp"

namespace rt {
  // Initializes cuRAND random number generators
  __global__ void init(CSampler* sampler, SDeviceFrame* frame) {
    uint16_t y = blockIdx.y;
    uint16_t x = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t samplerId = y * frame->width + x;
    sampler[samplerId].init(samplerId, 0);
  }

  __global__ void clearBuffer(SDeviceFrame* frame) {
    uint16_t y = blockIdx.y;
    uint16_t x = blockIdx.x * blockDim.x + threadIdx.x;

    if (y < frame->height && x < frame->width) {
      uint32_t currentPixel = frame->bpp * (y * frame->width + x);
      frame->data[currentPixel + 0] = 0.f;
      frame->data[currentPixel + 1] = 0.f;
      frame->data[currentPixel + 2] = 0.f;
    }
  }

  // Raytracing
  __global__ void renderFrame(CDeviceScene* scene, CCamera* camera, CSampler* sampler, uint16_t numSamples, SDeviceFrame* frame) {
    //uint16_t y = blockIdx.y;
    //uint16_t x = blockIdx.x * blockDim.x + threadIdx.x;


    //if (y < frame->height && x < frame->width) {
    //  //extern __shared__ char sharedScene[];
    //  //SSharedMemoryInitializer::copyScene(sharedScene, scene);

    //  uint32_t currentPixel = frame->bpp * (y * frame->width + x);
    //  uint32_t samplerId = y * frame->width + x;

    //  CPixelSampler pixelSampler(camera, x, y, &(sampler[samplerId]));
    //  //CPathIntegrator integrator((CDeviceScene*)sharedScene, &pixelSampler, &(sampler[samplerId]), numSamples);
    //  CPathIntegrator integrator(scene, &pixelSampler, &(sampler[samplerId]), numSamples);
    //  glm::vec3 L = integrator.Li();

    //  frame->data[currentPixel + 0] += L.r;
    //  frame->data[currentPixel + 1] += L.g;
    //  frame->data[currentPixel + 2] += L.b;

    //}
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
    //m_hostCamera(frameWidth, frameHeight, 90, glm::vec3(-0.5f, 0.5f, 0.5f), glm::vec3(0.0f, 0.5f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f)),
    m_hostCamera(frameWidth, frameHeight, 90, glm::vec3(-450.f, 73.f, 450.f), glm::vec3(-10.f, 73.f, -43.f), glm::vec3(0.0f, 1.0f, 0.0f)),
    //m_hostCamera(frameWidth, frameHeight, 90, glm::vec3(-5.f, 1.f, 5.f), glm::vec3(0.f, 1.f, 0.f), glm::vec3(0.0f, 1.0f, 0.0f)),
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
    float lightness = 0.75f / 255.0f;
    //m_scene.addSceneobject(CHostSceneobject(new CCuboid(glm::vec3(-400.f, 50.f, 300.f), glm::vec3(20.f), glm::vec3(0.f, 1.f, 0.f)), glm::vec3(lightness, lightness, 0.85f), 0.01f, glm::vec3(0.9f), 0.01f, 0.01f, 1.00029f, 1.5f)); // as normal reference
    //m_scene.addSceneobject(CHostSceneobject(new Sphere(glm::vec3(2.f, 0.5f, 0.f), 0.25f, glm::vec3(0.f, 1.f, 0.f)), glm::vec3(lightness, lightness, 0.85f), 0.01f, glm::vec3(0.9f), 0.01f, 0.01f, 1.00029f, 1.5f));
    //m_scene.addSceneobject(CHostSceneobject(new Sphere(glm::vec3(0.f, 0.5f, 2.f), 0.25f, glm::vec3(0.f, 1.f, 0.f)), glm::vec3(lightness, lightness, 0.85f), 0.01f, glm::vec3(0.9f), 0.01f, 0.01f, 1.00029f, 1.5f));
    //m_scene.addSceneobject(CHostSceneobject(new Sphere(glm::vec3(-2.f, 0.5f, 0.f), 0.25f, glm::vec3(0.f, 1.f, 0.f)), glm::vec3(lightness, lightness, 0.85f), 0.01f, glm::vec3(0.9f), 0.01f, 0.01f, 1.00029f, 1.5f));
    //m_scene.addSceneobject(CHostSceneobject(new Sphere(glm::vec3(0.f, 0.5f, -2.f), 0.25f, glm::vec3(0.f, 1.f, 0.f)), glm::vec3(lightness, lightness, 0.85f), 0.01f, glm::vec3(0.9f), 0.01f, 0.01f, 1.00029f, 1.5f));

    m_scene.addSceneobject(CHostSceneobject(new CNVDBMedium("../../raytracer/assets/wdas_cloud/wdas_cloud_sixteenth.nvdb", glm::vec3(0.f, 0.f, 0.f), glm::vec3(50.f, 50.f, 50.f), 1.f, 0.0001f, glm::vec3(0.f, 200.f, 0.f), glm::vec3(0.f, 1.f, 0.f), glm::vec3(1.f)))); // volume SGGX
    //m_scene.addSceneobject(CHostSceneobject(new CNVDBMedium("../../raytracer/assets/Smoke_Plume_01/Smoke_Plume_01/embergen_smoke_plume_a_0.nvdb", glm::vec3(0.f, 0.f, 0.f), glm::vec3(50.f, 50.f, 50.f), 1.f, 0.0001f, glm::vec3(0.f, 200.f, 0.f), glm::vec3(0.f, 0.f, 1.f), glm::vec3(0.5f)))); // volume SGGX
    //m_scene.addSceneobjectsFromAssimp("../../raytracer/assets/teapot.obj");
    //m_scene.addSceneobjectsFromAssimp("../../raytracer/assets/chestnut", "AL05a.obj", glm::vec3(100.f, 0.f, 0.f), glm::vec3(0.f, 0.f, 1.f), glm::vec3(1.f));
    //m_scene.addSceneobjectsFromAssimp("../../raytracer/assets/chestnut", "AL05m.obj", glm::vec3(100.f, 0.f, 0.f), glm::vec3(0.f, 0.f, 1.f), glm::vec3(1.f));
    //m_scene.addSceneobjectsFromAssimp("../../raytracer/assets/chestnut", "AL05y.obj", glm::vec3(100.f, 0.f, 0.f), glm::vec3(0.f, 0.f, 1.f), glm::vec3(1.f));
    //m_scene.addSceneobjectsFromAssimp("../../raytracer/assets/pine", "scrubPine.obj", glm::vec3(-100.f, 0.f, 0.f), glm::vec3(0.f, 1.f, 0.f), glm::vec3(0.3f));
    //m_scene.addSceneobjectsFromAssimp("../../raytracer/assets/test", "alpha_test.obj", glm::vec3(0.f), glm::vec3(0.f, 1.f, 0.f), glm::vec3(1.f));
    //m_scene.addSceneobjectsFromAssimp("../../raytracer/assets/hairball", "hairball.obj", glm::vec3(0.f), glm::vec3(0.f, 1.f, 0.f), glm::vec3(1.f));
    //m_scene.addSceneobjectsFromAssimp("../../raytracer/assets/sponza", "sponza.obj", glm::vec3(0.f), glm::vec3(0.f, 1.f, 0.f), glm::vec3(0.1f));
    //m_scene.addSceneobjectsFromAssimp("../../raytracer/assets/San_Miguel", "san-miguel.obj", glm::vec3(0.f), glm::vec3(0.f, 1.f, 0.f), glm::vec3(1.f));
    //m_scene.addSceneobjectsFromAssimp("../../raytracer/assets/test", "normal_map_test.obj", glm::vec3(0.f), glm::vec3(0.f, 1.f, 0.f), glm::vec3(1.f));

    // Add environment map
    //m_scene.setEnvironmentMap(CEnvironmentMap("./../../raytracer/assets/sunflowers_1k_edit.hdr"));
    m_scene.setEnvironmentMap(CEnvironmentMap("./../../raytracer/assets/envmap.hdr"));

    allocateDeviceMemory();
    initOptix();
    copyToDevice();
    initDeviceData();
  }

  Raytracer::~Raytracer() {
    freeDeviceMemory();
    CRTBackend::instance()->release();
  }

  // Renderpipeline
  SFrame Raytracer::renderFrame(const std::function<bool()>& keyCallback) {
    //CDeviceScene* scene = m_scene.deviceScene();
    dim3 grid(m_frameWidth / m_blockSize, m_frameHeight);
    rt::clearBuffer << <grid, m_blockSize >> > (m_deviceFrame);
    CUDA_ASSERT(cudaDeviceSynchronize());
    bool abortRendering = false;
    for (uint16_t sample = 0; sample < m_numSamples; ++sample) {
      std::cout << "Sample " << sample + 1 << "/" << m_numSamples << std::endl;
      //CPerformanceMonitoring::startMeasurement("renderFrame");
      //rt::renderFrame << <grid, m_blockSize, sizeof(CDeviceScene) + m_scene.sceneobjects().size() * (sizeof(CDeviceSceneobject) + dataSize) >> > (scene, m_deviceCamera, m_deviceSampler, m_numSamples, m_deviceFrame);
      OPTIX_ASSERT(optixLaunch(
        CRTBackend::instance()->pipeline(),
        0,             // stream
        reinterpret_cast<CUdeviceptr>(m_deviceLaunchParams),
        sizeof(SLaunchParams),
        &CRTBackend::instance()->sbt(),
        m_frameWidth,  // launch width
        m_frameHeight, // launch height
        1       // launch depth
      ));
      CUDA_ASSERT(cudaDeviceSynchronize());
      abortRendering = keyCallback();
      if (abortRendering) {
        return retrieveFrame();
      }
      //CPerformanceMonitoring::endMeasurement("renderFrame");
    }
    //rt::filterFrame << <grid, m_blockSize >> > (m_deviceFrame);
    //CUDA_ASSERT(cudaDeviceSynchronize());

    dim3 reductionGrid(m_frameWidth / m_blockSize, 1);
    rt::computeGlobalTonemapping1 << <reductionGrid, m_blockSize >> > (m_deviceFrame, m_deviceAverage);
    CUDA_ASSERT(cudaDeviceSynchronize());

    rt::computeGlobalTonemapping2 << <1, 1 >> > (m_deviceFrame, m_deviceAverage, m_deviceTonemappingValue);
    CUDA_ASSERT(cudaDeviceSynchronize());

    ////CPerformanceMonitoring::startMeasurement("applyTonemapping");
    ////rt::applyTonemapping << <grid, m_blockSize >> > (m_deviceFrame, m_tonemappingFactor);
    rt::applyTonemapping << <grid, m_blockSize >> > (m_deviceFrame, m_deviceTonemappingValue);
    CUDA_ASSERT(cudaDeviceSynchronize());
    ////CPerformanceMonitoring::endMeasurement("applyTonemapping");

    ////CPerformanceMonitoring::startMeasurement("correctGamma");
    rt::correctGamma << <grid, m_blockSize >> > (m_deviceFrame, m_gamma);
    CUDA_ASSERT(cudaDeviceSynchronize());
    ////CPerformanceMonitoring::endMeasurement("correctGamma");

    ////CPerformanceMonitoring::startMeasurement("fillByteFrame");
    rt::fillByteFrame << <grid, m_blockSize >> > (m_deviceFrame);
    CUDA_ASSERT(cudaDeviceSynchronize());
    //CPerformanceMonitoring::endMeasurement("fillByteFrame");

    SFrame frame = retrieveFrame();
    return frame;
  }

  SFrame Raytracer::renderPreview() {
    dim3 grid(m_frameWidth / m_blockSize, m_frameHeight);

    rt::clearBuffer << <grid, m_blockSize >> > (m_deviceFrame);
    CUDA_ASSERT(cudaDeviceSynchronize());

    OPTIX_ASSERT(optixLaunch(
      CRTBackend::instance()->pipeline(),
      0,             // stream
      reinterpret_cast<CUdeviceptr>(m_deviceLaunchParams),
      sizeof(SLaunchParams),
      &CRTBackend::instance()->sbt(),
      m_frameWidth,  // launch width
      m_frameHeight, // launch height
      1       // launch depth
    ));
    CUDA_ASSERT(cudaDeviceSynchronize());

    dim3 reductionGrid(m_frameWidth / m_blockSize, 1);
    rt::computeGlobalTonemapping1 << <reductionGrid, m_blockSize >> > (m_deviceFrame, m_deviceAverage);
    CUDA_ASSERT(cudaDeviceSynchronize());

    rt::computeGlobalTonemapping2 << <1, 1 >> > (m_deviceFrame, m_deviceAverage, m_deviceTonemappingValue);
    CUDA_ASSERT(cudaDeviceSynchronize());

    rt::applyTonemapping << <grid, m_blockSize >> > (m_deviceFrame, m_deviceTonemappingValue);
    CUDA_ASSERT(cudaDeviceSynchronize());

    rt::correctGamma << <grid, m_blockSize >> > (m_deviceFrame, m_gamma);
    CUDA_ASSERT(cudaDeviceSynchronize());

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

  void Raytracer::updateCamera(EPressedKey pressedKeys, const glm::vec2& mouseMoveDir) {
    glm::vec3 posCamSpace(0.f);
    if (pressedKeys & EPressedKey::W) {
      posCamSpace += glm::vec3(0.f, 0.f, 0.5f);
    }
    if (pressedKeys & EPressedKey::S) {
      posCamSpace -= glm::vec3(0.f, 0.f, 0.5f);
    }
    if (pressedKeys & EPressedKey::A) {
      posCamSpace -= glm::vec3(0.5f, 0.f, 0.f);
    }
    if (pressedKeys & EPressedKey::D) {
      posCamSpace += glm::vec3(0.5f, 0.f, 0.f);
    }
    if (pressedKeys & EPressedKey::Q) {
      posCamSpace -= glm::vec3(0.f, 0.5f, 0.f);
    }
    if (pressedKeys & EPressedKey::E) {
      posCamSpace += glm::vec3(0.f, 0.5f, 0.f);
    }

    // Move camera only along along three axes around up vector
    {
      glm::vec3 viewDir = glm::vec3(m_hostCamera.viewToWorld() * glm::vec4(0.f, 0.f, -1.f, 0.f));
      glm::vec3 moveDirRight = glm::cross(viewDir, m_hostCamera.up());
      glm::vec3 moveDirForward = glm::cross(m_hostCamera.up(), moveDirRight);

      glm::mat4 moveToWorld = glm::mat4(glm::vec4(glm::normalize(moveDirRight), 0.f), glm::vec4(glm::normalize(m_hostCamera.up()), 0.f), glm::vec4(glm::normalize(moveDirForward), 0.f), glm::vec4(m_hostCamera.position(), 1.f));
      glm::vec3 posWorldSpace = glm::vec3(moveToWorld * glm::vec4(posCamSpace, 1.f));

      //glm::vec3 posWorldSpace = glm::vec3(m_hostCamera.viewToWorld() * glm::vec4(posCamSpace, 1.f));
      m_hostCamera.updatePosition(posWorldSpace);
    }

    {
      glm::vec3 viewDir(0.f, 0.f, -1.f);
      viewDir += glm::vec3(mouseMoveDir.x, mouseMoveDir.y, 0.f) * 0.03f;
      viewDir = glm::normalize(viewDir);
      glm::vec3 lookAtCamSpace = viewDir;
      glm::vec3 lookAtWorldSpace = glm::vec3(m_hostCamera.viewToWorld() * glm::vec4(lookAtCamSpace, 1.f));
      m_hostCamera.updateLookAt(lookAtWorldSpace);
    }




    cudaMemcpy(m_deviceCamera, &m_hostCamera, sizeof(CCamera), cudaMemcpyHostToDevice);
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
    cudaMalloc(&m_deviceLaunchParams, sizeof(SLaunchParams));
    CTextureManager::allocateDeviceMemory();
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

    SLaunchParams launchParams;
    launchParams.width = m_hostCamera.sensorWidth();
    launchParams.height = m_hostCamera.sensorHeight();
    launchParams.bpp = m_bpp;
    launchParams.data = m_deviceFrameData;
    launchParams.filtered = m_deviceFilteredFrame;
    launchParams.dataBytes = m_deviceFrameDataBytes;
    launchParams.scene = m_scene.deviceScene();
    launchParams.camera = m_deviceCamera;
    launchParams.sampler = m_deviceSampler;
    launchParams.numSamples = m_numSamples;
    cudaMemcpy(m_deviceLaunchParams, &launchParams, sizeof(SLaunchParams), cudaMemcpyHostToDevice);

    CTextureManager::copyToDevice();
  }

  void Raytracer::initDeviceData() {
    //CPerformanceMonitoring::startMeasurement("init");
    dim3 grid(m_frameWidth / m_blockSize, m_frameHeight);
    init << <grid, m_blockSize >> > (m_deviceSampler, m_deviceFrame);
    cudaError_t e = cudaDeviceSynchronize();
    //CPerformanceMonitoring::endMeasurement("init");
  }

  void Raytracer::initOptix() {
    CRTBackend* rtBackend = CRTBackend::instance();
    rtBackend->init();
#ifdef DEBUG
    std::string modulePath = "cuda_to_ptx.dir/Debug/shaders.optix.ptx";
#endif
#ifdef RELEASE
    std::string modulePath = "cuda_to_ptx.dir/Release/shaders.optix.ptx";
#endif
    rtBackend->createModule(modulePath);
    rtBackend->createProgramGroups();
    rtBackend->createPipeline();
    const std::vector <SRecord<const CDeviceSceneobject*>> sbtHitRecords = m_scene.getSBTHitRecords();
    rtBackend->createSBT(sbtHitRecords);
    m_scene.buildOptixAccel();
  }

  void Raytracer::freeDeviceMemory() {
    m_scene.freeDeviceMemory();
    CUDA_ASSERT(cudaFree(m_deviceCamera));
    CUDA_ASSERT(cudaFree(m_deviceFrameData));
    CUDA_ASSERT(cudaFree(m_deviceFrame));
    CUDA_ASSERT(cudaFree(m_deviceAverage));
    CUDA_ASSERT(cudaFree(m_deviceTonemappingValue));
    CUDA_ASSERT(cudaFree(m_deviceLaunchParams));

    CTextureManager::freeDeviceMemory();
  }
  SFrame Raytracer::retrieveFrame() const {
    SFrame frame;
    uint32_t entries = m_frameWidth * m_frameHeight * m_bpp;
    frame.width = m_frameWidth;
    frame.height = m_frameHeight;
    frame.bpp = m_bpp;
    frame.data.resize(entries);
    CUDA_ASSERT(cudaMemcpy(frame.data.data(), m_deviceFrameData, entries * sizeof(float), cudaMemcpyDeviceToHost));
    frame.dataBytes.resize(entries);
    CUDA_ASSERT(cudaMemcpy(frame.dataBytes.data(), m_deviceFrameDataBytes, entries * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    return frame;
  }
}