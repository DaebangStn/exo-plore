#include "core/Record.h"
#include "ray/PyRecord.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace std;


py::array_t<double> PyRecord::get_data() const {
    const vector<ssize_t> shape = {static_cast<ssize_t>(_data.cols()), static_cast<ssize_t>(nrow)};
    const vector<ssize_t> strides = {
        static_cast<ssize_t>(_data.outerStride() * sizeof(double)),
        static_cast<ssize_t>(sizeof(double))
    };
    return {shape,strides,_data.data(),
        py::cast(this)  // base object
    };
}


void bind_PyRecord(pybind11::module& m)
{
    py::class_<PyRecord>(m, "Record")
        .def(py::init<vector<string>&>())
        .def_property_readonly("data", &PyRecord::get_data)
        .def_property_readonly("fields", &PyRecord::get_fields)
        .def_property_readonly("field_to_colidx", &PyRecord::get_field_to_colidx)
        .def_property_readonly("ncol", &PyRecord::get_ncol)
        .def_property_readonly("nrow", &PyRecord::get_nrow)
    ;
}
