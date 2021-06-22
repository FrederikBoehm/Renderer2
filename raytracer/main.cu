#include "raytracer.hpp"
#include "../visualisation/visualisation.hpp"
#include <iostream>

int main() {
  using namespace rt;
  const uint16_t WIDTH = 1920;
  const uint16_t HEIGHT = 1080;

  rt::Raytracer raytracer(WIDTH, HEIGHT);
  vis::CVisualisation visualizer(WIDTH, HEIGHT);

  SFrame frame = raytracer.renderFrame();

  visualizer.writeToFile("./", frame, vis::EImageFormat::JPG);
  visualizer.writeToFile("./", frame, vis::EImageFormat::PNG);

#ifdef GUI_PLATFORM
  while (true) {
    visualizer.render(frame);
  }
#endif GUI_PLATFORM

  return 0;

}