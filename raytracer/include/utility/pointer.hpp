#ifndef POINTER_HPP
#define POINTER_HPP

#include "qualifiers.hpp"

//template <typename T>
//CPointer<T> make_pointer(T&& v) {
//  return CPointer<T>((T&&)v);
//}

template <typename T>
class CPointer {
public:
  DH_CALLABLE CPointer(): m_v(nullptr) {
    m_refs = malloc(sizeof(size_t));
    *m_refs = 1;
  }

  DH_CALLABLE CPointer(T& v) {
    m_v = malloc(sizeof(T));
    *m_v = (T&&)v;
    m_refs = malloc(sizeof(size_t));
    *m_refs = 1;
  }

  DH_CALLABLE CPointer(T&& v) {
    m_v = malloc(sizeof(T));
    *m_v = (T&&)v;
    m_refs = malloc(sizeof(size_t));
    *m_refs = 1;
  }

  //DH_CALLABLE CPointer(T* v = nullptr) : m_v(nullptr) {
  //  if (v) {
  //    m_v = v;
  //    m_refs = malloc(sizeof(size_t));
  //    *m_refs = 1;
  //  }
  //}

  DH_CALLABLE CPointer(CPointer<T>& ptr) : m_refs(ptr.m_refs), m_v(ptr.m_v) {
    if (m_v) {
      ++(*m_refs);
    }
  }

  DH_CALLABLE CPointer(CPointer<T>&& ptr) : m_refs(ptr.m_refs), m_v(ptr.m_v) {
    ptr.m_refs = nullptr;
    ptr.m_v = nullptr;
  }

  DH_CALLABLE ~CPointer() {
    if (m_refs == 0) {
      free(m_refs);
      free(m_v);
    }
    else {
      --(*m_refs);
    }
  }

  DH_CALLABLE CPointer<T> operator=(CPointer<T>& ptr) {
    if (m_v) {
      *(this->m_refs) -= 1;
    }
    if (*(this->m_refs) == 0) {
      free(this->m_refs);
      free(this->m_v);
    }
    this->m_refs = ptr.m_refs;
    this->m_v = ptr.m_v;
    if (this->m_v) {
      ++(*m_refs);
    }
    return *this;
  }

  DH_CALLABLE CPointer<T> operator=(CPointer<T>&& ptr) {
    if (m_v) {
      *(this->m_refs) -= 1;
    }
    if (*(this->m_refs) == 0) {
      free(this->m_refs);
      free(this->m_v);
    }
    m_refs = ptr.m_refs;
    m_v = ptr.m_v;

    ptr.m_refs = nullptr;
    ptr.m_v = nullptr;

    return *this;
  }

  DH_CALLABLE CPointer<T> operator=(nullptr_t ptr) {
    if (m_v) {
      *(this->m_refs) -= 1;
    }
    if (*(this->m_refs) == 0) {
      free(this->m_refs);
      free(this->m_v);
    }
    m_refs = ptr;
    m_v = ptr;
    return *this;
  }

  DH_CALLABLE T* operator->() {
    return m_v;
  }

  DH_CALLABLE explicit operator bool() {
    return m_v;
  }


  DH_CALLABLE friend bool operator==(CPointer<T>& l, CPointer<T>& r);
  DH_CALLABLE friend bool operator!=(CPointer<T>& l, CPointer<T>& r);

  DH_CALLABLE friend bool operator==(CPointer<T>& l, T* r);
  DH_CALLABLE friend bool operator!=(CPointer<T>& l, T* r);

private:
  size_t* m_refs;
  T* m_v;
};

template <typename T>
bool operator==(CPointer<T>& l, CPointer<T>& r) {
  return l.m_v == r.m_v;
}

template <typename T>
bool operator!=(CPointer<T>& l, CPointer<T>& r) {
  return l != r;
}

template <typename T>
bool operator==(CPointer<T>& l, T* r) {
  return l.m_v == r;
}

template <typename T>
bool operator!=(CPointer<T>& l, T* r) {
  return l.m_v != r;
}

#endif // !POINTER_HPP
