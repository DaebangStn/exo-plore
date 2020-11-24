#include "pybind11/pybind11.h"
#include <Python.h>
#include <iostream>


void print_interpreter_info() {
    if (!Py_IsInitialized()) {
        std::cerr << "Python interpreter is not initialized." << std::endl;
        return;
    }

    pybind11::module sys = pybind11::module::import("sys");
    pybind11::print("[Python] sys.path:", sys.attr("path"));
    pybind11::print("[Python] interpreter path:", sys.attr("executable"));
}

void print_py_stacktrace(){
    if (!Py_IsInitialized()) {
        std::cerr << "Python interpreter is not initialized." << std::endl;
        return;
    }
    pybind11::module traceback = pybind11::module::import("traceback");
    pybind11::object print_stack = traceback.attr("print_stack");
    print_stack();
}
