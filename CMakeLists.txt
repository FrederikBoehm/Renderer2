cmake_minimum_required(VERSION 3.18)

project(Renderer2 LANGUAGES CXX)

set(SUBPROJECT "raytracer" CACHE STRING "Subproject: raytracer, vdb_converter")

if(SUBPROJECT STREQUAL "raytracer")
	message(STATUS "Generating subproject 'raytracer'")
	include(CMakeLists_raytracer.txt)
elseif(SUBPROJECT STREQUAL "vdb_converter")
	message(STATUS "Generating subproject 'vdb_converter'")
	include(CMakeLists_vdb_converter.txt)
else()
	message(FATAL_ERROR "Subproject '${SUBPROJECT}' not supported")
endif()
