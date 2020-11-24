find_package(DART REQUIRED COMPONENTS collision-bullet external-odelcpsolver)
find_package(yaml-cpp REQUIRED)
file(GLOB srcs "*.h" "*.cpp")

add_library(core SHARED ${srcs})
target_link_libraries(core PUBLIC
        dart
        dart-collision-bullet
        dart-external-odelcpsolver
        util2
        yaml-cpp
        tinyxml2::tinyxml2
)
target_include_directories(core PUBLIC
        ${DART_INCLUDE_DIRS}
)
