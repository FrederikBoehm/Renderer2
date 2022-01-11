#ifndef PRESSED_KEY_HPP
#define PRESSED_KEY_HPP

enum EPressedKey {
  NONE = 0,
  D = 1,
  A = 2,
  W = 4,
  S = 8,
  Q = 16,
  E = 32
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
