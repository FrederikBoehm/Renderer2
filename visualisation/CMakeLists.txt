project(visualisation LANGUAGES CXX)

set(SOURCE_FILES)
set(HEADER_FILES visualisation.hpp)

add_library(visualisation STATIC ${SOURCE_FILES} ${HEADER_FILES})

set_target_properties(visualisation PROPERTIES LINKER_LANGUAGE CXX)