set(SOURCE_FILES main.cu)
set(HEADER_FILES)

add_executable(raytracer ${SOURCE_FILES} ${HEADER_FILES})

target_compile_features(raytracer PRIVATE cxx_std_14)

set_target_properties(raytracer PROPERTIES CUDA_SEPERABLE_COMPILATION ON)
