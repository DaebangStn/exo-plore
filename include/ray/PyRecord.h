#ifndef PYRECORD_H
#define PYRECORD_H

#include "core/Record.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>


class PyRecord : public Record {
public:
    using Record::Record;  // Inherit constructor
    explicit PyRecord(const Record& record) : Record(record) {}

    pybind11::array_t<double> get_data() const;
    const std::vector<std::string>& get_fields() const { return field_names; }
    const std::unordered_map<std::string, int>& get_field_to_colidx() const { return field_to_colidx; }
    unsigned int get_ncol() const { return ncol; }
    unsigned int get_nrow() const { return nrow; }
};
#endif //PYRECORD_H
