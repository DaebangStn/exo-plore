#include "ray/Rollout.h"
#include "ray/PyRecord.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace std;

class PyRollout : public Rollout {
public:
    using Rollout::Rollout;  // Inherit constructor
    vector<PyRecord> RunParam(const unordered_map<string, float> &param)
    {
        const auto records = Rollout::RunParam(param);
        vector<PyRecord> py_records;
        for (const auto record : records)
        {
            py_records.emplace_back(*record);
        }
        return py_records;
    }
};

void bind_PyRollout(pybind11::module& m)
{
    py::class_<PyRollout>(m, "Rollout")
        .def(py::init<string&, bool>(), py::arg("ts_path"), py::arg("force_use_device") = false,
             R"pbdoc(
                Rollout(ts_path: string, force_use_device: bool = False) -> None
                Construct the Rollout object with the given torchscript path.

                Parameters:
                    ts_path (string): The torchscript directory path
                    force_use_device (bool): The flag to force using the device
             )pbdoc")
        .def("load_config", &PyRollout::LoadRecordConfig,
             py::arg("record_config"), py::arg("cycle") = -1,
             R"pbdoc(
                load_config(record_config: string, cycle: int = -1) -> None
                Load the record configuration file with the optional cycle.

                Parameters:
                    record_config (string): The record configuration file path
                    cycle (int): The cycle number for each sample cycle
             )pbdoc")
        .def("run", &PyRollout::RunParam, py::arg("param"),
             R"pbdoc(
                run(param: Dict[str, float]) -> Record
                Run the single rollout with the given parameter.

                Parameters:
                    param (Dict[str, float]): The parameter for the rollout

                Returns:
                    Record: The record of the rollout
             )pbdoc")
        .def("get_fields", &PyRollout::GetRecordFields,
             R"pbdoc(
                get_fields() -> List[str]
                Get the fields of the record.

                Returns:
                    List[str]: The fields of the record"
             )pbdoc")
        .def("set_shortening_multiplier", &PyRollout::SetShorteningMultiplier, py::arg("multiplier"),
             R"pbdoc(
                set_shortening_multiplier(multiplier: float) -> None
                Set the shortening multiplier for all muscles.

                Parameters:
                    multiplier (float): The shortening rate multiplier
             )pbdoc")
    ;
}
