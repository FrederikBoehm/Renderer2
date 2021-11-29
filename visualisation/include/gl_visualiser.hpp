#ifndef GL_VISUALISER_HPP
#define GL_VISUALISER_HPP

#include <cstdint>

#include "../../common/frame.hpp"
#include "../../common/pressed_key.hpp"

class GLFWwindow;

namespace vis {
  class CGLVisualiser {
  public:
    CGLVisualiser(uint16_t width, uint16_t height);
    ~CGLVisualiser();

    void render(const SFrame& frame);
    void init();
    void release() const;

    EPressedKey getPressedKeys() const;
  private:
    static EPressedKey s_pressedKeys;

    uint16_t m_width;
    uint16_t m_height;
    GLFWwindow* m_window;

    uint32_t m_vao;
    uint32_t m_vbo;
    uint32_t m_ebo;
    uint32_t m_program;
    uint32_t m_texture;

    friend void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

    void initGL();
    void setupVertices();
    void setupShaders();
    void setupTexture();
  };
}
#endif // !GL_VISUALISER_HPP
