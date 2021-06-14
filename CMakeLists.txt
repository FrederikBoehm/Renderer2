cmake_minimum_required(VERSION 3.10)
project(map_raytracer LANGUAGES CXX CUDA)


add_subdirectory(visualisation)
add_subdirectory(3rdparty/glm)
add_subdirectory(3rdparty/glfw)
add_subdirectory(3rdparty/glad)
add_subdirectory(raytracer)
