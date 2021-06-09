#include "raytracer.hpp"

void main() {
  using namespace rt;
  rt::Raytracer raytracer;


  raytracer.renderFrame();
  while (true) {
    raytracer.renderFrame();
  }

}