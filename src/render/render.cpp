#include "render/GLFWApp.h"
#include "util/struct.h"


int main(int argc, char** argv) 
{
    const RenderArg args = render_arguments(argc, argv);
    MASS::GLFWApp app(args);
    app.startLoop();
    return 0;
}