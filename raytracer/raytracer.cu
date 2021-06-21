
#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>

#include "device_launch_parameters.h"

#include "raytracer.hpp"

#include "sampling/sampler.hpp"

namespace rt {
  __global__ void init(CSampler* sampler) {
    sampler->init();
  }

  __global__ void renderFrame(CDeviceScene* scene, CCamera* camera, CSampler* sampler, uint16_t numSamples, SDeviceFrame* frame) {
    uint16_t y = blockIdx.y;
    uint16_t x = blockIdx.x;

    if (y < frame->height && x < frame->width) {
      uint32_t currentPixel = frame->bpp * (y * frame->width + x);

      Ray eyeRay = camera->samplePixel(x, y);
      SSurfaceInteraction si = scene->intersect(eyeRay);
      if (si.hitInformation.hit) {
        if (si.material.Le() != glm::vec3(0.0f)) {
          glm::vec3 le = si.material.Le() / (float)numSamples; // TODO: Maybe divide by numSamples at the very end
          frame->data[currentPixel + 0] += le.r;
          frame->data[currentPixel + 1] += le.g;
          frame->data[currentPixel + 2] += le.b;
        }
        else {
          glm::vec3 tangentSpaceDirection = sampler->uniformSampleHemisphere();
          // Construct tangent space
          glm::vec3 notN = normalize(glm::vec3(si.hitInformation.normal.x + 1.0f, si.hitInformation.normal.x + 2.0f, si.hitInformation.normal.x + 3.0f));
          glm::vec3 tangent = glm::normalize(glm::cross(notN, si.hitInformation.normal));
          glm::vec3 bitangent = glm::normalize(glm::cross(si.hitInformation.normal, tangent));

          glm::mat4 tangentToWorld(glm::vec4(tangent, 0.0f), glm::vec4(bitangent, 0.0f), glm::vec4(si.hitInformation.normal, 0.0f), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
          glm::vec3 worldSpaceDirection = glm::normalize(glm::vec3(tangentToWorld * glm::vec4(tangentSpaceDirection, 0.0f)));

          Ray shadowRay = Ray(si.hitInformation.pos + FLT_EPSILON * si.hitInformation.normal, worldSpaceDirection);
          SSurfaceInteraction si2 = scene->intersect(shadowRay);


          glm::vec3 f = si.material.f(si.hitInformation, -eyeRay.m_direction, shadowRay.m_direction);
          glm::vec3 Le = si2.material.Le();
          float cosine = glm::max(glm::dot(si.hitInformation.normal, shadowRay.m_direction), 0.0f);
          float pdf = sampler->uniformHemispherePdf();

          glm::vec3 L = f * Le * cosine / ((float)numSamples * pdf);
          frame->data[currentPixel + 0] += L.r;
          frame->data[currentPixel + 1] += L.g;
          frame->data[currentPixel + 2] += L.b;
        }
      }
    }
  }

  __global__ void applyTonemapping(SDeviceFrame* frame, float tonemapFactor) {
    uint16_t y = blockIdx.y;
    uint16_t x = blockIdx.x;

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

  __global__ void correctGamma(SDeviceFrame* frame, float gamma) {
    uint16_t y = blockIdx.y;
    uint16_t x = blockIdx.x;

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

  __global__ void fillByteFrame(SDeviceFrame* frame) {
    uint16_t y = blockIdx.y;
    uint16_t x = blockIdx.x;

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
    m_numSamples(100),
    m_tonemappingFactor(1.0f),
    m_gamma(2.0f),
    m_deviceCamera(nullptr),
    m_deviceFrameData(nullptr),
    m_deviceSampler(nullptr) {
    // Add scene objects
    m_scene.addSceneobject(CHostSceneobject(EShape::PLANE, glm::vec3(0.0f, 0.0f, 0.0f), 5000.f, glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.7f), glm::vec3(0.8f), 2.0f));
    float lightness = 50.0f / 255.0f;
    m_scene.addSceneobject(CHostSceneobject(EShape::SPHERE, getSpherePosition(0.05f, 0, 6), 0.05f, glm::vec3(), glm::vec3(lightness, lightness, 0.85f), glm::vec3(0.9f), 1000.0f));
    m_scene.addSceneobject(CHostSceneobject(EShape::SPHERE, getSpherePosition(0.05f, 1, 6), 0.05f, glm::vec3(), glm::vec3(0.85f, lightness, 0.85f), glm::vec3(0.9f), 1000.0f));
    m_scene.addSceneobject(CHostSceneobject(EShape::SPHERE, getSpherePosition(0.05f, 2, 6), 0.05f, glm::vec3(), glm::vec3(0.85f, lightness, lightness), glm::vec3(0.9f), 1000.0f));
    m_scene.addSceneobject(CHostSceneobject(EShape::SPHERE, getSpherePosition(0.05f, 3, 6), 0.05f, glm::vec3(), glm::vec3(0.85f, 0.85f, lightness), glm::vec3(0.9f), 1000.0f));
    m_scene.addSceneobject(CHostSceneobject(EShape::SPHERE, getSpherePosition(0.05f, 4, 6), 0.05f, glm::vec3(), glm::vec3(lightness, 0.85f, lightness), glm::vec3(0.9f), 1000.0f));
    m_scene.addSceneobject(CHostSceneobject(EShape::SPHERE, getSpherePosition(0.05f, 5, 6), 0.05f, glm::vec3(), glm::vec3(lightness, 0.85f, 0.85f), glm::vec3(0.9f), 1000.0f));
    m_scene.addSceneobject(CHostSceneobject(EShape::PLANE, glm::vec3(0.0f, 0.3f, 0.0f), 0.3f, glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(3.0f))); // Light

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
    for (uint16_t sample = 0; sample < m_numSamples; ++sample) {
      std::cout << "Sample " << sample + 1 << "/" << m_numSamples << std::endl;
      rt::renderFrame << <grid, 1 >> > (scene, m_deviceCamera, m_deviceSampler, m_numSamples, m_deviceFrame);
      cudaError_t error = cudaDeviceSynchronize();
    }
    rt::applyTonemapping << <grid, 1 >> > (m_deviceFrame, m_tonemappingFactor);
    cudaDeviceSynchronize();

    rt::correctGamma << <grid, 1 >> > (m_deviceFrame, m_gamma);
    cudaDeviceSynchronize();

    rt::fillByteFrame << <grid, 1 >> > (m_deviceFrame);
    cudaDeviceSynchronize();

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
    cudaMalloc(&m_deviceSampler, sizeof(CSampler));
    cudaMalloc(&m_deviceCamera, sizeof(CCamera));
    cudaMalloc(&m_deviceFrame, sizeof(SDeviceFrame));
    cudaMalloc(&m_deviceFrameData, sizeof(float)*m_hostCamera.sensorWidth()*m_hostCamera.sensorHeight()*m_bpp);
    cudaMalloc(&m_deviceFrameDataBytes, sizeof(uint8_t)*m_hostCamera.sensorWidth()*m_hostCamera.sensorHeight()*m_bpp);
  }

  void Raytracer::copyToDevice() {
    m_scene.copyToDevice();
    CCamera deviceCamera = m_hostCamera;
    deviceCamera.setSampler(m_deviceSampler);
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
    init << <1, 1 >> > (m_deviceSampler);
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