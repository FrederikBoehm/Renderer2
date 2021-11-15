cmake_policy(SET CMP0105 NEW)

set(SOURCE_FILES 
	main.cu
	raytracer.cu
	src/utility/performance_monitoring.cpp
	src/camera/camera.cu
	src/scene/scene.cu
	src/scene/sceneobject.cu
	src/shapes/shape.cu
	src/shapes/sphere.cu
	src/shapes/circle.cu
	src/intersect/ray.cu
	src/material/material.cu
	src/material/lambertian_brdf.cu
	src/material/blinn_phong_brdf.cu
	src/sampling/sampler.cu
	src/material/oren_nayar_brdf.cu
	src/material/microfacet_brdf.cu
	src/material/fresnel.cu
	src/material/microfacet_distribution.cu
	src/sampling/distribution_1d.cu
	src/integrators/direct_lighting_integrator.cu
	src/camera/pixel_sampler.cu
	src/medium/homogeneous_medium.cu
	src/medium/henyey_greenstein_phase_function.cu
	src/texture/texture.cu
	src/sampling/distribution_2d.cu
	src/scene/environmentmap.cu
	src/integrators/path_integrator.cu
	src/medium/medium.cu
	src/medium/heterogenous_medium.cu
	src/shapes/cuboid.cu
	src/shapes/rectangle.cu
	src/medium/nvdb_medium.cu)
set(HEADER_FILES
	raytracer.hpp
	include/utility/performance_monitoring.hpp
	include/camera/camera.hpp
	include/scene/scene.hpp
	include/scene/sceneobject.hpp
	include/scene/interaction.hpp
	include/shapes/shape.hpp
	include/shapes/sphere.hpp
	include/shapes/circle.hpp
	include/intersect/ray.hpp
	include/intersect/hit_information.hpp
	include/material/material.hpp
	include/material/lambertian_brdf.hpp
	include/material/blinn_phong_brdf.hpp
	include/utility/qualifiers.hpp
	include/sampling/sampler.hpp
	include/material/oren_nayar_brdf.hpp
	include/material/brdf_functions.hpp
	include/material/microfacet_brdf.hpp
	include/material/fresnel.hpp
	include/material/microfacet_distribution.hpp
	include/sampling/distribution_1d.hpp
	include/utility/functions.hpp
	include/integrators/direct_lighting_integrator.hpp
	include/camera/pixel_sampler.hpp
	include/sampling/mis.hpp
	include/medium/homogeneous_medium.hpp
	include/medium/henyey_greenstein_phase_function.hpp
	include/texture/texture.hpp
	include/sampling/distribution_2d.hpp
	include/scene/environmentmap.hpp
	include/integrators/path_integrator.hpp
	include/integrators/objects.hpp
	include/utility/debugging.hpp
	include/medium/medium.hpp
	include/medium/heterogenous_medium.hpp
	include/shapes/cuboid.hpp
	include/shapes/rectangle.hpp
	include/medium/nvdb_medium.hpp)


add_executable(raytracer ${SOURCE_FILES} ${HEADER_FILES})
set_property(TARGET raytracer PROPERTY CUDA_SEPARABLE_COMPILATION ON)
#set_target_properties(raytracer PROPERTIES CUDA_SEPERABLE_COMPILATION ON)
target_include_directories(raytracer PRIVATE include)
target_include_directories(raytracer PRIVATE "../3rdparty/glm")
target_include_directories(raytracer PRIVATE "../3rdparty")
target_include_directories(raytracer PRIVATE "../3rdparty/vcpkg/installed/x64-windows/include")

get_target_property(NANOVDB_INCLUDE_DEBUG lib_openvdb_external_dbg INCLUDE_DIRECTORIES)
get_target_property(NANOVDB_INCLUDE_RELEASE lib_openvdb_external_rel INCLUDE_DIRECTORIES)
target_include_directories(raytracer PRIVATE $<$<CONFIG:Debug>:${NANOVDB_INCLUDE_DEBUG}>
											 $<$<CONFIG:Release>:${NANOVDB_INCLUDE_RELEASE}>)
#target_include_directories(raytracer PRIVATE ${NANOVDB_INCLUDE_RELEASE})

target_link_libraries(raytracer PRIVATE glm)
target_link_libraries(raytracer PRIVATE visualisation)
#message("[DEBUG] Adding link library openvdb")
#target_link_libraries(raytracer PRIVATE debug lib_openvdb_external_dbg optimized lib_openvdb_external_rel)
#target_link_libraries(raytracer PRIVATE lib_openvdb_external_rel)
#target_link_libraries(raytracer PRIVATE openvdb)

source_group(TREE ${CMAKE_SOURCE_DIR} FILES ${SOURCE_FILES} ${HEADER_FILES})

#message("[DEBUG] Setting target compile options for raytracer")
target_compile_options(raytracer PRIVATE "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:DEBUG>>:-G>")
target_compile_features(raytracer PRIVATE cxx_std_17)
target_compile_definitions(raytracer PRIVATE NANOVDB_USE_INTRINSICS=1 BOOST_PP_VARIADICS=0)

if (UNIX)
set_target_properties(raytracer PROPERTIES CUDA_ARCHITECTURES "75")
endif()

if (LTO)
target_link_options(raytracer PRIVATE "$<$<CONFIG:RELEASE>:$<DEVICE_LINK:-dlto>>")
endif()

if (GUI)
target_compile_definitions(raytracer PRIVATE GUI_PLATFORM)
endif()

