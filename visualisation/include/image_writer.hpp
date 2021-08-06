#ifndef IMAGE_WRITER_HPP
#define IMAGE_WRITER_HPP

#include "../../common/frame.hpp"
#include <string>

namespace vis {
  enum EImageFormat {
    JPG = 1,
    PNG = 2
  };

  class CImageWriter {
  public:
    CImageWriter();

    void writeToFile(const std::string& outputDir, const SFrame& frame, EImageFormat format) const;

  private:
    std::string m_outputDir;

  };
}

#endif // !IMAGE_WRITER_HPP
