#ifndef FRAME_HPP
#define FRAME_HPP
#include <vector>

struct SFrame {
  uint16_t width;
  uint16_t height;
  uint8_t bpp;
  std::vector<float> data;
};
#endif // !FRAME_HPP
