#include "gl_visualiser.hpp"

#include <iostream>
#include <vector>

#ifdef GUI_PLATFORM
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#endif

namespace vis {

  EPressedKey CGLVisualiser::s_pressedKeys = EPressedKey::NONE;
  bool CGLVisualiser::s_mousePressed = false;
  glm::vec2 CGLVisualiser::s_previousMousePos(0.f);

  CGLVisualiser::CGLVisualiser(uint16_t width, uint16_t height) :
    m_width(width),
    m_height(height),
    m_window(nullptr),
    m_vao(0) {

  }

  CGLVisualiser::~CGLVisualiser() {
#ifdef GUI_PLATFORM
    glfwTerminate();
#endif
  }

  void CGLVisualiser::render(const SFrame& frame) {
#ifdef GUI_PLATFORM
    glfwPollEvents();



    // Render
    // Clear the colorbuffer
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(m_program);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frame.width, frame.height, 0, GL_RGB, GL_FLOAT, frame.data.data());

    glBindVertexArray(m_vao);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_BYTE, 0);

    // Swap the screen buffers
    glfwSwapBuffers(m_window);
#endif
  }

  void CGLVisualiser::init() {
    initGL();
    setupVertices();
    setupShaders();
    setupTexture();
  }

  void CGLVisualiser::initGL() {
#ifdef GUI_PLATFORM


    glfwInit();
    // Set all the required options for GLFW
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

    // Create a GLFWwindow object that we can use for GLFW's functions
    m_window = glfwCreateWindow(m_width, m_height, "raytracer", NULL, NULL);
    glfwMakeContextCurrent(m_window);
    if (m_window == NULL)
    {
      std::cout << "Failed to create GLFW window" << std::endl;
      glfwTerminate();
    }

    // Set the required callback functions
    glfwSetKeyCallback(m_window, &keyCallback);
    glfwSetMouseButtonCallback(m_window, &mousePressCallback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
      std::cout << "Failed to initialize OpenGL context" << std::endl;
    }

    // Define the viewport dimensions
    glViewport(0, 0, m_width, m_height);
#endif // GUI_PLATFORM
  }

  void CGLVisualiser::setupVertices() {
#ifdef GUI_PLATFORM


    glGenVertexArrays(1, &m_vao);
    glBindVertexArray(m_vao);

    glGenBuffers(1, &m_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);

    std::vector<float> vertices = { -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
                                     1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
                                     1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
                                    -1.0f,  1.0f, 0.0f, 0.0f, 1.0f };
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);


    glGenBuffers(1, &m_ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ebo);

    std::vector<uint8_t> indices = { 0, 1, 3,
                                     1, 2, 3 };
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(uint8_t), indices.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, false, 5 * sizeof(float), (void*)(0 * sizeof(float)));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, false, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
#endif // GUI_PLATFORM
  }

  void CGLVisualiser::setupShaders() {
#ifdef GUI_PLATFORM


    const char* vertexSource = "\n"
      "#version 450 core\n"
      "layout(location = 0) in vec3 aPos;\n"
      "layout(location = 1) in vec2 aUV;\n"
      "\n"
      "out vec2 vUV;\n"
      "void main() {\n"
      "  vUV = aUV;\n"
      "  gl_Position = vec4(aPos, 1.0f);\n"
      "}\n";


    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexSource, 0);
    glCompileShader(vertexShader);

    GLint vertex_compiled;
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &vertex_compiled);
    if (vertex_compiled != GL_TRUE)
    {
      GLsizei log_length = 0;
      GLchar message[1024];
      glGetShaderInfoLog(vertexShader, 1024, &log_length, message);
      // Write the error to a log
      std::cout << "[VERTEX COMPILE]" << std::endl;
      std::cout << message << std::endl;
    }

    const char* fragmentSource = "\n"
      "#version 450 core\n"
      "uniform sampler2D tex;\n"
      "\n"
      "in vec2 vUV;\n"
      "out vec4 FragColor;\n"
      "void main() {\n"
      "  FragColor = vec4(texture(tex, vUV).rgb, 1.0);\n"
      "}\n";

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentSource, 0);
    glCompileShader(fragmentShader);

    GLint fragment_compiled;
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &fragment_compiled);
    if (fragment_compiled != GL_TRUE)
    {
      GLsizei log_length = 0;
      GLchar message[1024];
      glGetShaderInfoLog(fragmentShader, 1024, &log_length, message);
      // Write the error to a log
      std::cout << "[FRAGMENT COMPILE]" << std::endl;
      std::cout << message << std::endl;
    }

    m_program = glCreateProgram();

    glAttachShader(m_program, vertexShader);
    glAttachShader(m_program, fragmentShader);
    glLinkProgram(m_program);

    GLint program_linked;
    glGetProgramiv(m_program, GL_LINK_STATUS, &program_linked);
    if (program_linked != GL_TRUE)
    {
      GLsizei log_length = 0;
      GLchar message[1024];
      glGetProgramInfoLog(m_program, 1024, &log_length, message);
      // Write the error to a log
      std::cout << "[PROGRAM LINK]" << std::endl;
      std::cout << message << std::endl;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
#endif // GUI_PLATFORM
  }

  void CGLVisualiser::setupTexture() {
#ifdef GUI_PLATFORM


    glGenTextures(1, &m_texture);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_texture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glUseProgram(m_program);
    glUniform1i(glGetUniformLocation(m_program, "tex"), 0);
    glUseProgram(0);
#endif // GUI_PLATFORM
  }

  void CGLVisualiser::release() const {
#ifdef GUI_PLATFORM


    glDeleteVertexArrays(1, &m_vao);
    glDeleteBuffers(1, &m_vbo);
    glDeleteBuffers(1, &m_ebo);
#endif // GUI_PLATFORM
  }

  void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
#ifdef GUI_PLATFORM
    EPressedKey k = EPressedKey::NONE;
    switch (key) {
      case GLFW_KEY_W:
        k = EPressedKey::W;
        break;
      case GLFW_KEY_A:
        k = EPressedKey::A;
        break;
      case GLFW_KEY_S:
        k = EPressedKey::S;
        break;
      case GLFW_KEY_D:
        k = EPressedKey::D;
        break;
      case GLFW_KEY_Q:
        k = EPressedKey::Q;
        break;
      case GLFW_KEY_E:
        k = EPressedKey::E;
        break;
    }
    switch (action) {
    case GLFW_PRESS:
      CGLVisualiser::s_pressedKeys |= k;
      break;
    case GLFW_RELEASE:
      CGLVisualiser::s_pressedKeys &= ~k;
      break;
    }
#endif // GUI_PLATFORM
  }

  void mousePressCallback(GLFWwindow* window, int button, int action, int mods) {
#ifdef GUI_PLATFORM
    if (button == GLFW_MOUSE_BUTTON_1) {
      if (action == GLFW_PRESS) {
        CGLVisualiser::s_mousePressed = true;
        double x;
        double y;
        glfwGetCursorPos(window, &x, &y);
        CGLVisualiser::s_previousMousePos = glm::vec2(x, y);
      }
      if (action == GLFW_RELEASE) {
        CGLVisualiser::s_mousePressed = false;
      }
    }
#endif // GUI_PLATFORM

  }

  void CGLVisualiser::pollEvents() const {
#ifdef GUI_PLATFORM
    glfwPollEvents();
#endif
  }

  EPressedKey CGLVisualiser::getPressedKeys() const {
#ifdef GUI_PLATFORM
    return s_pressedKeys;
#else
    return EPressedKey::NONE;
#endif // GUI_PLATFORM
  }

  glm::vec2 CGLVisualiser::getMouseMoveDirection() const {
#ifdef GUI_PLATFORM
    if (s_mousePressed) {
      double x;
      double y;
      glfwGetCursorPos(m_window, &x, &y);
      glm::vec2 currentPos(x, y);
      if (currentPos != s_previousMousePos) {
        glm::vec2 diff = currentPos - s_previousMousePos;
        return glm::normalize(glm::vec2(diff.x, -diff.y));
      }
      else {
        return glm::vec2(0.f);
      }
    }
    else {
      return glm::vec2(0.f);
    }
#else
    return glm::vec2(0.f);
#endif
  }
}