#include "raytracer.hpp"
#include "../visualisation/visualisation.hpp"
#include <iostream>
#include <fstream>
#include "utility/performance_monitoring.hpp"


#include <vector>
#include <glm/glm.hpp>
#include "filtering/openvdb_backend.hpp"
#include "backend/config_loader.hpp"

int main(int argc, char** argv) {
  using namespace rt;

  if (argc < 2) {
    fprintf(stderr, "Error: Provide valid path to config file.\n");
    return 1;
  }

  SConfig config = CConfigLoader::loadConfig(argv[1]);
  if (config.filteringConfig.filter) {

    filter::SOpenvdbBackendConfig openvdbConfig;
    openvdbConfig.boundingBoxes = config.scene->getObjectBBs(rt::ESceneobjectMask::FILTER);
    openvdbConfig.numVoxels = config.filteringConfig.numVoxels;
    if (openvdbConfig.boundingBoxes.size() > 0) {
      filter::COpenvdbBackend* openvdbBackend = filter::COpenvdbBackend::instance();
      openvdbBackend->init(openvdbConfig);
      openvdbBackend->setValues();
      nanovdb::GridHandle<nanovdb::HostBuffer> gridHandle = openvdbBackend->getNanoGridHandle();
      openvdbBackend->writeToFile(gridHandle, "./filtering", "filtered_mesh.nvdb");
    }
    else {
      printf("[WARNING]: No bounding boxes provided --> proceed without filtering.\n");
    }

  }

  rt::Raytracer raytracer(config);

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

  return 0;

}