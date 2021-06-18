#include "raytracer.hpp"
#include "../visualisation/visualisation.hpp"

void main() {
  using namespace rt;
  const uint16_t WIDTH = 1280;
  const uint16_t HEIGHT = 720;

  rt::Raytracer raytracer(WIDTH, HEIGHT);
  vis::CVisualisation visualizer(WIDTH, HEIGHT);

  SFrame frame = raytracer.renderFrame();


  visualizer.render(frame);
  while (true) {
    visualizer.render(frame);
  }

}