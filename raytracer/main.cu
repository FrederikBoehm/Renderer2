#include "raytracer.hpp"
#include "../visualisation/visualisation.hpp"
#include <iostream>
#include <fstream>
#include "utility/performance_monitoring.hpp"

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