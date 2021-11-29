#ifndef PRESSED_KEY_HPP
#define PRESSED_KEY_HPP

enum EPressedKey {
  NONE = 0,
  RIGHT = 1,
  LEFT = 2,
  UP = 4,
  DOWN = 8
};

inline EPressedKey& operator|=(EPressedKey& l, EPressedKey r) {
  return (EPressedKey&)((int&)(l) |= (int)(r));
}

inline EPressedKey& operator&=(EPressedKey& l, EPressedKey r) {
  return (EPressedKey&)((int&)(l) &= (int)(r));
}

inline EPressedKey operator~(EPressedKey r) {
  return (EPressedKey)(~(int)(r));
}

inline EPressedKey operator|(EPressedKey l, EPressedKey r) {
  return (EPressedKey)((int)(l) | (int)(r));
}

inline EPressedKey operator&(EPressedKey l, EPressedKey r) {
  return (EPressedKey)((int)(l) & (int)(r));
}
#endif // !PRESSED_KEY_HPP
