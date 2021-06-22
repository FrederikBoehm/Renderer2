#ifndef FRAME_HPP
#define FRAME_HPP
#include <vector>
#include <cstdint>

struct SFrame {
  uint16_t width;
  uint16_t height;
  uint8_t bpp;
  std::vector<float> data;
  std::vector<uint8_t> dataBytes;
};
#endif // !FRAME_HPP
