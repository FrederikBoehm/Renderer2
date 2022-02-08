#include "scene/sceneobject_mask.hpp"

namespace rt {
  ESceneobjectMask getMask(const std::string& maskString) {
    if (maskString == "NONE") {
      return ESceneobjectMask::NONE;
    }
    else if (maskString == "FILTER") {
      return ESceneobjectMask::FILTER;
    }
    else if (maskString == "RENDER") {
      return ESceneobjectMask::RENDER;
    }
    else {
      return ESceneobjectMask::ALL;
    }
  }
}