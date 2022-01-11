#ifndef GL_VISUALISER_HPP
#define GL_VISUALISER_HPP

#include <cstdint>

#include "../../common/frame.hpp"
#include "../../common/pressed_key.hpp"
#include "glm/glm.hpp"

class GLFWwindow;

namespace vis {
  class CGLVisualiser {
  public:
    CGLVisualiser(uint16_t width, uint16_t height);
    ~CGLVisualiser();

    void render(const SFrame& frame);
    void init();
    void release() const;

    void pollEvents() const;
    EPressedKey getPressedKeys() const;
    glm::vec2 getMouseMoveDirection() const;
  private:
    static EPressedKey s_pressedKeys;
    static glm::vec2 s_previousMousePos;
    static bool s_mousePressed;

    uint16_t m_width;
    uint16_t m_height;
    GLFWwindow* m_window;

    uint32_t m_vao;
    uint32_t m_vbo;
    uint32_t m_ebo;
    uint32_t m_program;
    uint32_t m_texture;

    friend void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    friend void mousePressCallback(GLFWwindow* window, int button, int action, int mods);

    void initGL();
    void setupVertices();
    void setupShaders();
    void setupTexture();
  };
}
#endif // !GL_VISUALISER_HPP
