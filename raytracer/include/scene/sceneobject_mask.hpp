#ifndef SCENEOBJECT_MASK_HPP
#define SCENEOBJECT_MASK_HPP
#include <cstdint>
#include <string>
namespace rt {
  enum ESceneobjectMask : uint8_t {
    NONE = 0,
    FILTER = 1,
    RENDER = 2,
    ALL = 255
  };

  ESceneobjectMask getMask(const std::string& maskString);
}
#endif