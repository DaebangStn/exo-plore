add_compile_options(-fvisibility=hidden)

find_package(DART REQUIRED COMPONENTS gui collision-bullet)
find_package(indicators REQUIRED)
find_package(OpenGL REQUIRED COMPONENTS OpenGL)
find_package(GLUT REQUIRED)

link_libraries(${TORCH_LIBRARIES})

set(GUI_LIBS GL GLU glut glad glfw imgui dart-gui)
set(GUI_INCLUDE_DIRS
        ${PROJECT_SOURCE_DIR}/libs/imgui
        ${PROJECT_SOURCE_DIR}/libs/implot
)

set(COMMON_SRC
        GLFunctions.cpp
        ShapeRenderer.cpp
        GLFWApp.cpp
)

add_executable(render
        ${COMMON_SRC}
        render.cpp
)
target_link_libraries(render
        ${GUI_LIBS}
        dart
        dart-collision-bullet
        dart-gui
        pthread
        core
        util2
)
target_include_directories(render PRIVATE
        ${GUI_INCLUDE_DIRS}
        ${DART_INCLUDE_DIRS}
        ${TORCH_INCLUDE_DIRS}
)

add_executable(export
        ${COMMON_SRC}
        export.cpp
)
target_link_libraries(export
        ${GUI_LIBS}
        dart
        dart-collision-bullet
        dart-gui
        pthread
        core
        util2
        indicators::indicators
)
target_include_directories(export PRIVATE
        ${GUI_INCLUDE_DIRS}
        ${INDICATORS_INCLUDE_DIRS}
        ${TORCH_INCLUDE_DIRS}
)
