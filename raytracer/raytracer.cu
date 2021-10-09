
#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>

#include "device_launch_parameters.h"

#include "raytracer.hpp"

#include "sampling/sampler.hpp"

#include "utility/performance_monitoring.hpp"
#include "integrators/direct_lighting_integrator.hpp"
#include "camera/pixel_sampler.hpp"

namespace rt {
  // Initializes cuRAND random number generators
  __global__ void init(CSampler* sampler, SDeviceFrame* frame) {
    uint16_t y = blockIdx.y;
    uint16_t x = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t samplerId = y * frame->width + x;
    sampler[samplerId].init(samplerId, 0);
  }

  // Raytracing
  __global__ void renderFrame(CDeviceScene* scene, CCamera* camera, CSampler* sampler, uint16_t numSamples, SDeviceFrame* frame) {
    uint16_t y = blockIdx.y;
    uint16_t x = blockIdx.x * blockDim.x + threadIdx.x;

    if (y < frame->height && x < frame->width) {
      uint32_t currentPixel = frame->bpp * (y * frame->width + x);
      uint32_t samplerId = y * frame->width + x;

      CPixelSampler pixelSampler(camera, x, y, &(sampler[samplerId]));
      CDirectLightingIntegrator integrator(scene, &pixelSampler, &(sampler[samplerId]), numSamples);
      glm::vec3 L = integrator.Li(EIntegrationStrategy::MULTIPLE_IMPORTANCE_SAMPLE);

      frame->data[currentPixel + 0] += L.r;
      frame->data[currentPixel + 1] += L.g;
      frame->data[currentPixel + 2] += L.b;
    }
  }

  // Map colors to [0.0f, 1.0f]
  __global__ void applyTonemapping(SDeviceFrame* frame, float tonemapFactor) {
    uint16_t y = blockIdx.y;
    uint16_t x = blockIdx.x * blockDim.x + threadIdx.x;

    if (y < frame->height && x < frame->width) {
      uint32_t currentPixel = frame->bpp * (y * frame->width + x);
    
      float r = frame->data[currentPixel + 0];
      float g = frame->data[currentPixel + 1];
      float b = frame->data[currentPixel + 2];

      frame->data[currentPixel + 0] = r / (r + tonemapFactor);
      frame->data[currentPixel + 1] = g / (g + tonemapFactor);
      frame->data[currentPixel + 2] = b / (b + tonemapFactor);
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
    m_hostCamera(frameWidth, frameHeight, 90, glm::vec3(0.0f, 0.25f, 0.5f), glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f)),
    m_numSamples(100), // higher -> less noise
    m_tonemappingFactor(1.0f),
    m_gamma(2.0f),
    m_deviceCamera(nullptr),
    m_deviceFrameData(nullptr),
    m_deviceSampler(nullptr),
    m_blockSize(128) {
    // Add scene objects
    m_scene.addSceneobject(CHostSceneobject(EShape::PLANE, glm::vec3(0.0f, 0.0f, 0.0f), 5000.f, glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.95f, 0.95f, 0.95f), 0.99f, glm::vec3(0.8f), 0.99f, 0.99f, 1.00029f, 1.2f));
    float lightness = 50.0f / 255.0f;
    //m_scene.addSceneobject(CHostSceneobject(EShape::SPHERE, getSpherePosition(0.05f, 0, 6), 0.05f, glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(lightness, lightness, 0.95f), 0.01f, glm::vec3(0.9f), 0.01f, 0.01f, 1.00029f, 1.5f)); // blue sphere
    //m_scene.addSceneobject(CHostSceneobject(EShape::SPHERE, getSpherePosition(0.05f, 1, 6), 0.05f, glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.95f, lightness, 0.95f), 0.01f,  glm::vec3(0.9f), 0.01f, 0.01f, 1.00029f, 1.5f)); // violet sphere
    //m_scene.addSceneobject(CHostSceneobject(EShape::SPHERE, getSpherePosition(0.05f, 2, 6), 0.05f, glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.95f, lightness, lightness), 0.01f,  glm::vec3(0.9f), 0.01f, 0.01f, 1.00029f, 1.5f)); // red sphere
    //m_scene.addSceneobject(CHostSceneobject(EShape::SPHERE, getSpherePosition(0.05f, 3, 6), 0.05f, glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.95f, 0.95f, lightness), 0.01f,  glm::vec3(0.9f), 0.01f, 0.01f, 1.00029f, 1.5f)); // yellow sphere
    //m_scene.addSceneobject(CHostSceneobject(EShape::SPHERE, getSpherePosition(0.05f, 4, 6), 0.05f, glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(lightness, 0.95f, lightness), 0.01f,  glm::vec3(0.9f), 0.01f, 0.01f, 1.00029f, 1.5f)); // green sphere
    //m_scene.addSceneobject(CHostSceneobject(EShape::SPHERE, getSpherePosition(0.05f, 5, 6), 0.05f, glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(lightness, 0.95f, 0.95f), 0.01f, glm::vec3(0.9f), 0.01f, 0.01f, 1.00029f, 1.5f)); // cyan sphere
    m_scene.addSceneobject(CHostSceneobject(EShape::SPHERE, glm::vec3(0.f, 0.15f, 0.0f), 0.15f, glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(10.f, 10.0f, 0.0f), glm::vec3(0.f, 0.0f, 0.0f), -0.0f)); // volume
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

    glm::vec3 light2Pos = getSpherePosition(0.1f, 1, 6) + glm::vec3(0.0f, 0.1f, 0.0f);
    m_scene.addLightsource(CHostSceneobject(EShape::PLANE, light2Pos, 0.05f, -glm::normalize(light2Pos), glm::vec3(2.0f, 0.0f, 0.5f)));
    glm::vec3 light3Pos = getSpherePosition(0.1f, 2, 6) + glm::vec3(0.0f, 0.1f, 0.0f);
    m_scene.addLightsource(CHostSceneobject(EShape::PLANE, light3Pos, 0.05f, -glm::normalize(light3Pos), glm::vec3(2.0f)));

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
    cudaError_t e;
    for (uint16_t sample = 0; sample < m_numSamples; ++sample) {
      std::cout << "Sample " << sample + 1 << "/" << m_numSamples << std::endl;
      //CPerformanceMonitoring::startMeasurement("renderFrame");
      rt::renderFrame << <grid, m_blockSize >> > (scene, m_deviceCamera, m_deviceSampler, m_numSamples, m_deviceFrame);
      e = cudaDeviceSynchronize();
      //CPerformanceMonitoring::endMeasurement("renderFrame");
    }
    //CPerformanceMonitoring::startMeasurement("applyTonemapping");
    rt::applyTonemapping << <grid, m_blockSize >> > (m_deviceFrame, m_tonemappingFactor);
    cudaDeviceSynchronize();
    //CPerformanceMonitoring::endMeasurement("applyTonemapping");

    //CPerformanceMonitoring::startMeasurement("correctGamma");
    rt::correctGamma << <grid, m_blockSize >> > (m_deviceFrame, m_gamma);
    cudaDeviceSynchronize();
    //CPerformanceMonitoring::endMeasurement("correctGamma");

    //CPerformanceMonitoring::startMeasurement("fillByteFrame");
    rt::fillByteFrame << <grid, m_blockSize >> > (m_deviceFrame);
    cudaDeviceSynchronize();
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
    cudaMalloc(&m_deviceFrameDataBytes, sizeof(uint8_t)*m_hostCamera.sensorWidth()*m_hostCamera.sensorHeight()*m_bpp);
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