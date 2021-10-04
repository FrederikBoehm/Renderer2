#ifndef MIS_HPP
#define MIS_HPP
#include "utility/qualifiers.hpp"
#include <cstdint>

namespace rt {
  DH_CALLABLE inline float balanceHeuristic(uint32_t nf, float fPdf, uint32_t ng, float gPdf) {
    return (nf * fPdf) / (nf * fPdf + ng * gPdf);
  }

  DH_CALLABLE inline float powerHeuristic(uint32_t nf, float fPdf, uint32_t ng, float gPdf) {
    float f = nf * fPdf;
    float g = ng * gPdf;
    return (f * f) / (f * f + g * g);
  }
}

#endif