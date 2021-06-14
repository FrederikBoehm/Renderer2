
#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>

#include "device_launch_parameters.h"

#include "raytracer.hpp"

namespace rt {
  __global__ void init(CCamera* camera) {
    camera->initCurandState();
  }

  __global__ void renderFrame(CDeviceScene* scene, CCamera* camera, SDeviceFrame* frame) {
    uint16_t y = blockIdx.y;
    uint16_t x = blockIdx.x;
    uint16_t numSamples = 1;
    float* data = frame->data;

    if (y < frame->height && x < frame->width) {
      glm::vec3 pixelColor(0.0f);
      for (uint16_t sample = 0; sample < numSamples; ++sample) {
        SSurfaceInteraction si = scene->intersect(camera->samplePixel(x, y));
        pixelColor.r += si.surfaceAlbedo.r;
        pixelColor.g += si.surfaceAlbedo.g;
        pixelColor.b += si.surfaceAlbedo.b;
      }
      pixelColor /= numSamples;
      uint32_t currentPixel = frame->bpp * (y * frame->width + x);
      frame->data[currentPixel + 0] = pixelColor.r;
      frame->data[currentPixel + 1] = pixelColor.g;
      frame->data[currentPixel + 2] = pixelColor.b;
    }
  }

  Raytracer::Raytracer(uint16_t frameWidth, uint16_t frameHeight) :
    m_frameWidth(frameWidth),
    m_frameHeight(frameHeight),
    m_bpp(3),
    m_scene(),
    m_hostCamera(frameWidth, frameHeight, 90, glm::vec3(0.0f, 0.25f, 1.0f), glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f)),
    m_deviceCamera(nullptr) {
    // Add scene objects
    m_scene.addSceneobject(CHostSceneobject(EShape::PLANE, glm::vec3(0.0f, 0.0f, 0.0f), 5.0f, glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(1.0f)));
    float lightness = 100.0f / 255.0f;
    m_scene.addSceneobject(CHostSceneobject(EShape::SPHERE, getSpherePosition(0.05f, 0, 6), 0.05f, glm::vec3(), glm::vec3(lightness, lightness, 1.0f)));
    m_scene.addSceneobject(CHostSceneobject(EShape::SPHERE, getSpherePosition(0.05f, 1, 6), 0.05f, glm::vec3(), glm::vec3(1.0f, lightness, 1.0f)));
    m_scene.addSceneobject(CHostSceneobject(EShape::SPHERE, getSpherePosition(0.05f, 2, 6), 0.05f, glm::vec3(), glm::vec3(1.0f, lightness, lightness)));
    m_scene.addSceneobject(CHostSceneobject(EShape::SPHERE, getSpherePosition(0.05f, 3, 6), 0.05f, glm::vec3(), glm::vec3(1.0f, 1.0f, lightness)));
    m_scene.addSceneobject(CHostSceneobject(EShape::SPHERE, getSpherePosition(0.05f, 4, 6), 0.05f, glm::vec3(), glm::vec3(lightness, 1.0f, lightness)));
    m_scene.addSceneobject(CHostSceneobject(EShape::SPHERE, getSpherePosition(0.05f, 5, 6), 0.05f, glm::vec3(), glm::vec3(lightness, 1.0f, 1.0f)));

    allocateDeviceMemory();
    copyToDevice();
    initDeviceData();
  }

  Raytracer::~Raytracer() {
    freeDeviceMemory();
  }

  SFrame Raytracer::renderFrame() {
    // TODO: Measure execution time
    cudaDeviceSynchronize();
    CDeviceScene* scene = m_scene.deviceScene();
    dim3 grid(m_frameWidth, m_frameHeight);
    rt::renderFrame << <grid, 1 >> > (scene, m_deviceCamera, m_deviceFrame);
    cudaError_t error = cudaDeviceSynchronize();
    SFrame frame = retrieveFrame();
    return frame;
  }

  glm::vec3 Raytracer::getSpherePosition(float sphereRadius, uint8_t index, uint8_t maxSpheres) {
    float x = 4.0f * sphereRadius * std::cos(2 * M_PI / maxSpheres * index);
    float z = -4.0f * sphereRadius * std::sin(2 * M_PI / maxSpheres * index);
    float y = sphereRadius;
    return glm::vec3(x, y, z);
  }

  void Raytracer::allocateDeviceMemory() {
    m_scene.allocateDeviceMemory();
    cudaMalloc(&m_deviceCamera, sizeof(CCamera));
    cudaMalloc(&m_deviceFrame, sizeof(SDeviceFrame));
    cudaMalloc(&m_deviceFrameData, sizeof(float)*m_hostCamera.sensorWidth()*m_hostCamera.sensorHeight()*m_bpp);
  }

  void Raytracer::copyToDevice() {
    m_scene.copyToDevice();
    cudaMemcpy(m_deviceCamera, &m_hostCamera, sizeof(CCamera), cudaMemcpyHostToDevice);
    
    SDeviceFrame f;
    f.width = m_hostCamera.sensorWidth();
    f.height = m_hostCamera.sensorHeight();
    f.bpp = m_bpp;
    f.data = m_deviceFrameData;
    cudaMemcpy(m_deviceFrame, &f, sizeof(SDeviceFrame), cudaMemcpyHostToDevice);
  }

  void Raytracer::initDeviceData() {
    init << <1, 1 >> > (m_deviceCamera);
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
    return frame;
  }
}