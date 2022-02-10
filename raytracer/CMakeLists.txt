cmake_policy(SET CMP0105 NEW)

set(OPTIX_SOURCE_FILES
	src/backend/shaders.optix.cu
)

add_library(cuda_to_ptx OBJECT ${OPTIX_SOURCE_FILES})
set_property(TARGET cuda_to_ptx PROPERTY CUDA_PTX_COMPILATION ON)
target_compile_features(cuda_to_ptx PRIVATE cxx_std_17)
#target_compile_definitions(cuda_to_ptx PRIVATE NANOVDB_USE_INTRINSICS=1 BOOST_PP_VARIADICS=0)
target_compile_definitions(cuda_to_ptx PRIVATE -DNANOVDB_USE_INTRINSICS=1 _ENABLE_EXTENDED_ALIGNED_STORAGE)
#target_compile_options(cuda_to_ptx PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-DNANOVDB_USE_INTRINSICS>")
set_property(TARGET cuda_to_ptx PROPERTY CUDA_ARCHITECTURES 52-virtual)

target_include_directories(cuda_to_ptx PRIVATE include)
target_include_directories(cuda_to_ptx PRIVATE "../3rdparty/glm")
target_include_directories(cuda_to_ptx PRIVATE "../3rdparty")
target_include_directories(cuda_to_ptx PRIVATE "../3rdparty/assimp/include")
target_include_directories(cuda_to_ptx PRIVATE "${CMAKE_BINARY_DIR}/3rdparty/assimp/include")

get_target_property(NANOVDB_INCLUDE_DEBUG lib_openvdb_external_dbg INCLUDE_DIRECTORIES)
get_target_property(NANOVDB_INCLUDE_RELEASE lib_openvdb_external_rel INCLUDE_DIRECTORIES)

target_include_directories(cuda_to_ptx PRIVATE $<$<CONFIG:Debug>:${NANOVDB_INCLUDE_DEBUG}>
											 $<$<CONFIG:Release>:${NANOVDB_INCLUDE_RELEASE}>)
add_dependencies(cuda_to_ptx openvdb_external_dbg openvdb_external_rel)

set(SOURCE_FILES 
	main.cu
	raytracer.cu
	src/utility/performance_monitoring.cpp
	src/utility/debugging.cu
	src/camera/camera.cu
	src/scene/scene.cu
	src/scene/sceneobject.cu
	src/scene/environmentmap.cu
	src/shapes/shape.cu
	src/shapes/sphere.cu
	src/shapes/circle.cu
	src/shapes/cuboid.cu
	src/shapes/rectangle.cu
	src/material/lambertian_brdf.cu
	src/material/blinn_phong_brdf.cu
	src/material/material.cu
	src/sampling/sampler.cu
	src/sampling/distribution_1d.cu
	src/sampling/distribution_2d.cu
	#src/integrators/direct_lighting_integrator.cu
	src/medium/homogeneous_medium.cu
	src/medium/henyey_greenstein_phase_function.cu
	src/medium/medium.cu
	src/medium/heterogenous_medium.cu
	src/medium/nvdb_medium.cu
	src/medium/sggx_phase_function.cu
	src/medium/phase_function.cu
	src/texture/texture.cu
	src/backend/rt_backend.cu
	src/mesh/mesh.cu
	#src/texture/texture_manager.cu
	src/backend/config_loader.cu
	src/backend/asset_manager.cu
	src/backend/build_optix_accel.cu
	src/medium/medium_instance.cu
	src/filtering/openvdb_backend.cpp
	src/scene/sceneobject_mask.cpp
	pipeline.cu
	filter.cu)
set(HEADER_FILES
	raytracer.hpp
	include/utility/performance_monitoring.hpp
	include/utility/qualifiers.hpp
	include/utility/functions.hpp
	include/utility/debugging.hpp
	include/camera/camera.hpp
	include/camera/pixel_sampler.hpp
	include/scene/scene.hpp
	include/scene/sceneobject.hpp
	include/scene/interaction.hpp
	include/scene/environmentmap.hpp
	include/scene/device_scene.hpp
	include/scene/device_sceneobject.hpp
	include/scene/types.hpp
	include/scene/device_scene_impl.hpp
	include/shapes/shape.hpp
	include/shapes/sphere.hpp
	include/shapes/circle.hpp
	include/shapes/cuboid.hpp
	include/shapes/rectangle.hpp
	include/intersect/ray.hpp
	include/intersect/hit_information.hpp
	include/material/material.hpp
	include/material/lambertian_brdf.hpp
	include/material/blinn_phong_brdf.hpp
	include/material/oren_nayar_brdf.hpp
	include/material/brdf_functions.hpp
	include/material/microfacet_brdf.hpp
	include/material/fresnel.hpp
	include/material/microfacet_distribution.hpp
	include/sampling/sampler.hpp
	include/sampling/distribution_1d.hpp
	include/sampling/mis.hpp
	include/sampling/distribution_2d.hpp
	#include/integrators/direct_lighting_integrator.hpp
	include/integrators/path_integrator.hpp
	include/integrators/objects.hpp
	include/integrators/path_integrator_impl.hpp
	include/medium/homogeneous_medium.hpp
	include/medium/henyey_greenstein_phase_function.hpp
	include/medium/medium.hpp
	include/medium/heterogenous_medium.hpp
	include/medium/nvdb_medium.hpp
	include/medium/sggx_phase_function.hpp
	include/medium/phase_function.hpp
	include/medium/medium_impl.hpp
	include/medium/phase_function_impl.hpp
	include/medium/nvdb_medium_impl.hpp
	include/texture/texture.hpp
	include/backend/rt_backend.hpp
	include/backend/types.hpp
	include/mesh/mesh.hpp
	#include/texture/texture_manager.hpp
	include/backend/config_loader.hpp
	config.json
	include/backend/asset_manager.hpp
	include/backend/build_optix_accel.hpp
	include/intersect/aabb.hpp
	include/medium/medium_instance.hpp
	include/medium/medium_instance_impl.hpp
	include/filtering/openvdb_backend.hpp
	include/filtering/openvdb_data.hpp
	include/scene/sceneobject_mask.hpp
	include/filtering/mesh_filter.hpp
	pipeline.hpp
	filter.hpp
	include/filtering/launch_params.hpp
	include/filtering/filtered_data.hpp)




add_executable(raytracer ${SOURCE_FILES} ${HEADER_FILES})
set_property(TARGET raytracer PROPERTY CUDA_SEPARABLE_COMPILATION ON)
#set_target_properties(raytracer PROPERTIES CUDA_SEPERABLE_COMPILATION ON)
target_include_directories(raytracer PRIVATE include)
target_include_directories(raytracer PRIVATE "../3rdparty/glm")
target_include_directories(raytracer PRIVATE "../3rdparty")
target_include_directories(raytracer PRIVATE "../3rdparty/vcpkg/installed/x64-windows/include")
target_include_directories(raytracer PRIVATE "../3rdparty/assimp/include")

target_include_directories(raytracer PRIVATE $<$<CONFIG:Debug>:${NANOVDB_INCLUDE_DEBUG}>
											 $<$<CONFIG:Release>:${NANOVDB_INCLUDE_RELEASE}>)
#target_include_directories(raytracer PRIVATE ${NANOVDB_INCLUDE_RELEASE})

target_link_libraries(raytracer PRIVATE glm)
target_link_libraries(raytracer PRIVATE visualisation)
target_link_libraries(raytracer PRIVATE assimp)
add_dependencies(raytracer cuda_to_ptx)
#target_link_libraries(raytracer PRIVATE cuda_to_ptx)
#message("[DEBUG] Adding link library openvdb")
target_link_libraries(raytracer PRIVATE debug lib_openvdb_external_dbg optimized lib_openvdb_external_rel)
#target_link_libraries(raytracer PRIVATE lib_openvdb_external_rel)
#target_link_libraries(raytracer PRIVATE openvdb)
target_link_libraries(raytracer PRIVATE TBB::tbb) # Required from openvdb

source_group(TREE ${CMAKE_SOURCE_DIR} FILES ${SOURCE_FILES} ${HEADER_FILES} ${OPTIX_SOURCE_FILES})

#message("[DEBUG] Setting target compile options for raytracer")
target_compile_options(raytracer PRIVATE "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:DEBUG>>:-G>")
target_compile_options(raytracer PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_52,code=[sm_52,lto_52]>")
target_compile_features(raytracer PRIVATE cxx_std_17)
target_compile_definitions(raytracer PRIVATE NANOVDB_USE_INTRINSICS=1 BOOST_PP_VARIADICS=0 _ENABLE_EXTENDED_ALIGNED_STORAGE)
target_compile_definitions(raytracer PRIVATE "$<$<CONFIG:DEBUG>:DEBUG>" "$<$<CONFIG:RELEASE>:RELEASE>")

if (MSVC)
target_compile_options(raytracer PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/bigobj>")
endif()

if (UNIX)
set_target_properties(raytracer PROPERTIES CUDA_ARCHITECTURES "75")
elseif(WIN32)
set_target_properties(raytracer PROPERTIES VS_DEBUGGER_COMMAND_ARGUMENTS "../../raytracer/config.json")
endif()

if (LTO)
target_link_options(raytracer PRIVATE "$<$<CONFIG:RELEASE>:$<DEVICE_LINK:-dlto>>")
endif()

if (GUI)
target_compile_definitions(raytracer PRIVATE GUI_PLATFORM)
endif()

add_custom_command(TARGET raytracer POST_BUILD        
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:assimp>
        $<TARGET_FILE_DIR:raytracer>)

add_custom_command(TARGET raytracer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:glad>
        $<TARGET_FILE_DIR:raytracer>)

add_custom_command(TARGET raytracer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:glfw>
        $<TARGET_FILE_DIR:raytracer>)

get_target_property(OPENVDB_BIN_DEBUG lib_openvdb_external_dbg IMPORTED_LOCATION)
get_target_property(OPENVDB_BIN_RELEASE lib_openvdb_external_rel IMPORTED_LOCATION)

add_custom_command(TARGET raytracer POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_if_different
		$<$<CONFIG:DEBUG>:${OPENVDB_BIN_DEBUG}>
		$<$<CONFIG:RELEASE>:${OPENVDB_BIN_RELEASE}>
		$<TARGET_FILE_DIR:raytracer>)

add_custom_command(TARGET raytracer POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<$<CONFIG:DEBUG>:${TBB_BIN_DEBUG}>
		$<$<CONFIG:RELEASE>:${TBB_BIN_RELEASE}>
        $<TARGET_FILE_DIR:raytracer>)

