#include "pipeline.hpp"

#include "raytracer.hpp"
#include "../visualisation/visualisation.hpp"
#include <iostream>
#include <fstream>
#include "utility/performance_monitoring.hpp"


#include <vector>
#include <glm/glm.hpp>
#include "filtering/openvdb_backend.hpp"
#include "backend/config_loader.hpp"
#include "filter.hpp"
#include "backend/asset_manager.hpp"
#include "backend/rt_backend.hpp"

CPipeline::CPipeline(const char* configPath):
  m_config(CConfigLoader::loadConfig(configPath)) {

}

void CPipeline::run() {
  using namespace rt;
  
  allocateDeviceMemory();
  if (m_config.filteringConfig.filter) {
    filter::CFilter filter(m_config);
    copyToDevice();
    filter.runFiltering();
    CRTBackend::instance()->reset();
  }


  rt::Raytracer raytracer(m_config);
  copyToDevice();

  vis::CVisualisation visualizer(raytracer.getFrameWidth(), raytracer.getFrameHeight());


#ifdef GUI_PLATFORM
  {
    SFrame previewFrame = raytracer.renderPreview();
    visualizer.render(previewFrame);
  }
  bool renderDetailed = true;
  SFrame frame;
  while (true) {
    visualizer.pollEvents();
    EPressedKey pressedKeys = visualizer.getPressedKeys();
    glm::vec2 mouseMoveDirection = visualizer.getMouseMoveDirection();
    while (pressedKeys || mouseMoveDirection != glm::vec2(0.f)) {
      // render preview
      raytracer.updateCamera(pressedKeys, mouseMoveDirection);
      SFrame previewFrame = raytracer.renderPreview();
      visualizer.render(previewFrame);
      visualizer.pollEvents();
      pressedKeys = visualizer.getPressedKeys();
      mouseMoveDirection = visualizer.getMouseMoveDirection();
      renderDetailed = true;
    }

    if (renderDetailed) {
      auto keyCallback = [&visualizer]() -> bool {
        visualizer.pollEvents();
        return visualizer.getPressedKeys() != EPressedKey::NONE || visualizer.getMouseMoveDirection() != glm::vec2(0.f);
      };
      frame = raytracer.renderFrame(keyCallback);
      if (keyCallback()) {
        continue;
      }
      visualizer.writeToFile("./", frame, vis::EImageFormat::JPG);
      visualizer.writeToFile("./", frame, vis::EImageFormat::PNG);
      renderDetailed = false;
    }

    visualizer.render(frame);
  }
#else
  auto keyCallback = []() -> bool {
    return false;
  };
  SFrame frame = raytracer.renderFrame(keyCallback);
  visualizer.writeToFile("./", frame, vis::EImageFormat::JPG);
  visualizer.writeToFile("./", frame, vis::EImageFormat::PNG);

#endif // GUI_PLATFORM

  freeDeviceMemory();
}

void CPipeline::allocateDeviceMemory() {
  m_config.scene->allocateDeviceMemory();
  rt::CAssetManager::allocateDeviceMemory();
}

void CPipeline::copyToDevice() {
  m_config.scene->copyToDevice();
  rt::CAssetManager::copyToDevice();
}

void CPipeline::freeDeviceMemory() {
  m_config.scene->freeDeviceMemory();
  rt::CAssetManager::freeDeviceMemory();
}

