find_package(glfw3 REQUIRED)
find_package(glad REQUIRED)

# Build imgui / implot library
add_definitions(-DIMGUI_DISABLE_OBSOLETE_FUNCTIONS)
include_directories(imgui)
add_subdirectory(ImGuiFileDialog)
add_library(imgui STATIC
        imgui/imgui.cpp imgui/imgui_draw.cpp imgui/imgui_widgets.cpp imgui/imgui_demo.cpp imgui/imgui_tables.cpp
        imgui/backends/imgui_impl_glfw.cpp imgui/backends/imgui_impl_opengl3.cpp
        implot/implot.cpp implot/implot_demo.cpp implot/implot_items.cpp)
target_include_directories(imgui PUBLIC implot ImGuiFileDialog)
target_link_libraries(imgui PUBLIC glfw glad ImGuiFileDialog)
