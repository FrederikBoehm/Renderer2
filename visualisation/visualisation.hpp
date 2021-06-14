#ifndef VISUALISATION_HPP
#define VISUALISATION_HPP

#include "../common/frame.hpp"

#include "include/gl_visualiser.hpp"

namespace vis {
  class CVisualisation {
  public:
    CVisualisation(uint16_t width, uint16_t height);
    ~CVisualisation();

    void render(const SFrame& frame);
  private:
    CGLVisualiser m_glVisualiser;
  };

  inline CVisualisation::CVisualisation(uint16_t width, uint16_t height) :
    m_glVisualiser(width, height) {
    m_glVisualiser.init();
  }

  inline CVisualisation::~CVisualisation() {
    m_glVisualiser.release();
  }

  inline void CVisualisation::render(const SFrame& frame) {
    m_glVisualiser.render(frame);
  }
}
#endif