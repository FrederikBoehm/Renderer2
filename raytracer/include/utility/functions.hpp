#ifndef FUNCTIONS_HPP
#define FUNCTIONS_HPP
#include <glm/glm.hpp>

#include "qualifiers.hpp"

namespace rt {
   template <typename Predicate> DH_CALLABLE int findInterval(int size, const Predicate& pred) {
    int first = 0;
    int len = size;
    while (len > 0) {
      int half = len >> 1;
      int middle = first + half;

      if (pred(middle)) {
        first = middle + 1;
        len -= half + 1;
      }
      else {
        len = half;
      }
    }

    return glm::clamp(first - 1, 0, size - 2);
  }
}

#endif // !FUNCTIONS_HPP
