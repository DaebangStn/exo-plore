#define MODULE_NAME muscle

#include <pybind11/pybind11.h>

namespace py = pybind11;
void bind_PyRecord(py::module& m);
void bind_PyRollout(py::module& m);
void bind_RayEnvManager(py::module& m);
void bind_MuscleTuple(py::module& m);

PYBIND11_MODULE(MODULE_NAME, m) {
    // Call the functions defined in the other .cpp files
    bind_PyRecord(m);
    bind_PyRollout(m);
    bind_MuscleTuple(m);
    bind_RayEnvManager(m);
}