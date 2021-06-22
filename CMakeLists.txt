cmake_minimum_required(VERSION 3.10)
project(map_raytracer LANGUAGES CXX CUDA)

set(GUI false CACHE BOOL "Setup project with available GUI")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if (UNIX)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif()


add_subdirectory(visualisation)
add_subdirectory(3rdparty/glm)

if (GUI)
add_subdirectory(3rdparty/glfw)
add_subdirectory(3rdparty/glad)
ENDIF()

add_subdirectory(raytracer)
