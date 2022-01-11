#include "raytracer.hpp"
#include "../visualisation/visualisation.hpp"
#include <iostream>
#include <fstream>
#include "utility/performance_monitoring.hpp"


#include <vector>
#include <glm/glm.hpp>

int main() {
  using namespace rt;
  const uint16_t WIDTH = 1920;
  const uint16_t HEIGHT = 1080;

  rt::Raytracer raytracer(WIDTH, HEIGHT);
  vis::CVisualisation visualizer(WIDTH, HEIGHT);


  //for (uint8_t i = 0; i < 10; ++i) {
  //  CPerformanceMonitoring::startMeasurement("renderFrame (Method)");
  //  SFrame frame = raytracer.renderFrame();
  //  CPerformanceMonitoring::endMeasurement("renderFrame (Method)");
  //  //rt::Raytracer raytracer2(WIDTH, HEIGHT);
  //}

  //std::fstream s("./renderFrame_method.csv", std::fstream::out);
  //std::string output = rt::CPerformanceMonitoring::toString();
  //s.write(output.c_str(), output.size());
  //std::cout << output << std::endl;

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