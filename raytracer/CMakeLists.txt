set(SOURCE_FILES main.cu raytracer.cpp raytracer.cu src/camera/camera.cpp src/camera/camera.cu src/scene/scene.cpp src/scene/scene.cu)
set(HEADER_FILES raytracer.hpp include/camera/camera.hpp include/scene/scene.hpp)


add_executable(raytracer ${SOURCE_FILES} ${HEADER_FILES})
target_include_directories(raytracer PRIVATE include)
target_include_directories(raytracer PRIVATE "../3rdparty/include/glm")

target_link_libraries(raytracer PRIVATE glm::glm)

source_group(TREE ${CMAKE_SOURCE_DIR} FILES ${SOURCE_FILES} ${HEADER_FILES})


target_compile_features(raytracer PRIVATE cxx_std_17)

set_target_properties(raytracer PROPERTIES CUDA_SEPERABLE_COMPILATION ON)
