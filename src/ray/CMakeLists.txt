file(GLOB_RECURSE SOURCES "*.cpp")

add_library(muscle SHARED
        ${SOURCES}
)
target_link_libraries(muscle PRIVATE
        core
        util2
        ${TORCH_LIBRARIES}
)
target_include_directories(muscle PUBLIC
        ${TORCH_INCLUDE_DIRS}
)
target_compile_definitions(muscle PRIVATE MUSCLE_POSTFIX=38)
set_target_properties(muscle PROPERTIES
        PREFIX ""
        OUTPUT_NAME "muscle"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python"
)
