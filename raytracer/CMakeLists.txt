

set(SOURCE_FILES 
	main.cu
	raytracer.cu
	src/camera/camera.cu
	src/scene/scene.cu
	src/scene/sceneobject.cpp
	src/shapes/shape.cu
	src/shapes/sphere.cu
	src/shapes/plane.cu
	src/intersect/ray.cu
	src/material/material.cpp)
set(HEADER_FILES
	raytracer.hpp
	include/camera/camera.hpp
	include/scene/scene.hpp
	include/scene/sceneobject.hpp
	include/scene/surface_interaction.hpp
	include/shapes/shape.hpp
	include/shapes/sphere.hpp
	include/shapes/plane.hpp
	include/intersect/ray.hpp
	include/intersect/hit_information.hpp
	include/material/material.hpp
	include/utility/qualifiers.hpp)


add_executable(raytracer ${SOURCE_FILES} ${HEADER_FILES})
set_property(TARGET raytracer PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(raytracer PRIVATE include)
target_include_directories(raytracer PRIVATE "../3rdparty/include/glm")

target_link_libraries(raytracer PRIVATE glm)

source_group(TREE ${CMAKE_SOURCE_DIR} FILES ${SOURCE_FILES} ${HEADER_FILES})

#target_compile_options(raytracer PRIVATE "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:DEBUG>>:-G>" "$<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>" "$<$<COMPILE_LANGUAGE:CUDA>:-x>" "$<$<COMPILE_LANGUAGE:CUDA>:cu>")
target_compile_options(raytracer PRIVATE "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:DEBUG>>:-G>")
target_compile_features(raytracer PRIVATE cxx_std_17)

set_target_properties(raytracer PROPERTIES CUDA_SEPERABLE_COMPILATION ON)
