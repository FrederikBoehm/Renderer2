
#include "image_writer.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

namespace vis {
  CImageWriter::CImageWriter() {
    stbi_flip_vertically_on_write(true);
    stbi_write_png_compression_level = 0;
  }

  void CImageWriter::writeToFile(const std::string& outputDir, const SFrame& frame, EImageFormat format) const {
    switch (format) {
    case EImageFormat::JPG:
      stbi_write_jpg((outputDir + "output.jpg").c_str(), frame.width, frame.height, frame.bpp, frame.dataBytes.data(), 100);
      break;
    case EImageFormat::PNG:
      auto ret = stbi_write_png((outputDir + "output.png").c_str(), frame.width, frame.height, frame.bpp, frame.dataBytes.data(), frame.bpp * frame.width);
      break;
    }
  }


}