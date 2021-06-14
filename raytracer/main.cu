#include "raytracer.hpp"
#include "../visualisation/visualisation.hpp"

void main() {
  using namespace rt;
  const uint16_t WIDTH = 1920;
  const uint16_t HEIGHT = 1080;

  rt::Raytracer raytracer(WIDTH, HEIGHT);
  vis::CVisualisation visualizer(WIDTH, HEIGHT);

  SFrame frame = raytracer.renderFrame();

  visualizer.render(frame);
  while (true) {
    visualizer.render(frame);
    //raytracer.renderFrame();
  }

}