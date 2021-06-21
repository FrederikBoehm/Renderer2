#include "raytracer.hpp"
#include "../visualisation/visualisation.hpp"
#include <iostream>

int main() {
  using namespace rt;
  const uint16_t WIDTH = 1280;
  const uint16_t HEIGHT = 720;

  rt::Raytracer raytracer(WIDTH, HEIGHT);
  vis::CVisualisation visualizer(WIDTH, HEIGHT);

  SFrame frame = raytracer.renderFrame();

  visualizer.writeToFile("./", frame, vis::EImageFormat::JPG);
  visualizer.writeToFile("./", frame, vis::EImageFormat::PNG);
  visualizer.render(frame);
  bool finish;
  std::cin >> finish;
  return 0;

}