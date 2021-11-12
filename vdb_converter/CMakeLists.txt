
add_executable(vdb_converter main.cpp)

target_compile_features(vdb_converter PRIVATE cxx_std_17)

if (MSVC)
target_compile_options(vdb_converter PRIVATE "/bigobj")
endif()

target_link_libraries(vdb_converter PRIVATE openvdb)
target_include_directories(vdb_converter PRIVATE "${CMAKE_SOURCE_DIR}/3rdparty/openvdb/nanovdb" "${CMAKE_SOURCE_DIR}/3rdparty/openvdb/openvdb")
