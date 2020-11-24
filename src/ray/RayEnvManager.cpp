#include "core/Environment.h"
#include "core/Utils.h"
#include "pybind11/eigen.h"
#include "pybind11/stl.h"

namespace py = pybind11;
using namespace std;
class RayEnvManager : public MASS::Environment
{
public:
    explicit RayEnvManager(const string& metadata_path) : Environment(false) {
        dart::math::seedRand();
        InitFromYaml(metadata_path);
    }

    explicit RayEnvManager(const string& metadata_path, int seed) : Environment(false) {
        srand(seed);
        InitFromYaml(metadata_path);
    }

    void SetParam(const unordered_map<string, float>& param, bool allow_duplicated_param = false) {
        Environment::SetParam(param, allow_duplicated_param);
        for (const auto& [key, value] : param) this->param[key] = value;
    }

    bool IsTerminated() {
        const int failure_code = IsEndOfEpisode();
        if (failure_code == 0) return false;
        cout << "[FAIL] episode failed(code:" << failure_code << ")" << endl;
        cout << "Params: ";
        for (const auto& [key, value] : param) cout << key << ":" << value << ", ";
        cout << endl;
        return true;
    }

    void Step() {
        Environment::Step(nullptr);
    }

    void StepsAtOnce()
    {
        int num = GetNumSteps();
        for(int j=0;j<num;j++) Step();
    }
private:
    unordered_map<string, float> param;
};

void bind_MuscleTuple(py::module& m)
{
    py::class_<MASS::MuscleTuple>(m, "MuscleTuple")
        .def(py::init<>())
        .def_readonly("JtA", &MASS::MuscleTuple::JtA)
        .def_readonly("JtA_reduced", &MASS::MuscleTuple::JtA_reduced)
        .def_readonly("tau_active", &MASS::MuscleTuple::tau_active)
        ;
}


void bind_RayEnvManager(py::module& m)
{
    py::class_<RayEnvManager>(m, "RayEnvManager")
        .def(py::init<string&>(), py::arg("metadata_path"),
             R"pbdoc(
                Construct the RayEnvManager object with the given metadata.

                Parameters:
                    metadata_path (str): The metadata of the environment
            )pbdoc")
        .def(py::init<string&, int>(), py::arg("metadata_path"), py::arg("seed"),
             R"pbdoc(
                Construct the RayEnvManager object with the given metadata.

                Parameters:
                    metadata_path (str): The metadata of the environment
                    seed (int): The seed of the environment
            )pbdoc")
        .def("GetMuscleNames", &RayEnvManager::GetMuscleNames,
                R"pbdoc(
                    Get the muscle names.

                    Returns:
                        List[str]: The muscle names
                )pbdoc")
        .def("GetUseMuscle", &RayEnvManager::GetUseMuscle,
                R"pbdoc(
                    Get the muscle names.

                    Returns:
                        bool: The flag to check if the muscle is used
                )pbdoc")
        .def("GetMuscleIndices", &RayEnvManager::GetMuscleIndices,
             py::arg("names"),
             R"pbdoc(
                Get the muscle indices with the muscle names.

                Parameters:
                    names (List[str]): The muscle names to get the indices

                Returns:
                    Dict[str, int]: The muscle indices with the muscle name
            )pbdoc")
        .def("GetNumState", &RayEnvManager::GetNumState,
             R"pbdoc(
                Get the number of the state.

                Returns:
                    int: The number of the state
            )pbdoc")
        .def("GetNumAction", &RayEnvManager::GetNumAction,
             R"pbdoc(
                Get the number of the action.

                Returns:
                    int: The number of the action
            )pbdoc")
        .def("GetNumSteps", &RayEnvManager::GetNumSteps)
        .def("GetNumMuscles", &RayEnvManager::GetNumMuscles,
             R"pbdoc(
                Get the number of the muscles.

                Returns:
                    int: The number of the muscles
            )pbdoc")
        .def("GetPdDof", &RayEnvManager::GetPdDof,
             R"pbdoc(
                Get the number of the degrees of freedom except root joint.

                Returns:
                    int: The number of the degrees of freedom
            )pbdoc")
        .def("GetMcnDof", &RayEnvManager::GetMcnDof,
             R"pbdoc(
                Get the number of the degrees of desired torque for the muscle.

                Returns:
                    int: The number of the degrees of freedom
            )pbdoc")
        .def("GetRelatedDof", &RayEnvManager::GetRelatedDof,
             R"pbdoc(
                Get the number of the total muscle related degrees of freedom.

                Returns:
                    int: The number of the total muscle related degrees of freedom
            )pbdoc")
        .def("Step", &RayEnvManager::Step)
        .def("StepsAtOnce", &RayEnvManager::StepsAtOnce)
        .def("Reset", &RayEnvManager::Reset)
        .def("IsEndOfEpisode", &RayEnvManager::IsEndOfEpisode,
             R"pbdoc(
                Check if the episode is ended.

                Returns:
                    int: The flag to check if the episode is ended
            )pbdoc")
        .def("GetReward", [](RayEnvManager& self) {
            return self.GetReward(nullptr);
        }, R"pbdoc(
            Get the reward of the environment.

            Returns:
                float: The reward of the environment
        )pbdoc")
        .def("GetRewardMap", &RayEnvManager::GetRewardMap)
        .def("SetActivationLevels", &RayEnvManager::SetActivationLevels)
        .def("GetMuscleTuple", &RayEnvManager::GetMuscleTuple,
             R"pbdoc(
                Get the muscle tuple.

                Returns:
                    MuscleTuple: The muscle tuple
            )pbdoc")
        .def("SetAction", static_cast<void (RayEnvManager::*)(const Eigen::VectorXd&)>(&RayEnvManager::SetAction),
             py::arg("action"),
             R"pbdoc(
                Set the action of the environment.

                Parameters:
                    action (np.ndarray): The action to set
            )pbdoc")
        .def("GetState", &RayEnvManager::GetState,
             R"pbdoc(
                Get the state of the environment.

                Returns:
                    np.ndarray: The state of the environment
            )pbdoc")
        .def("GetStateExo", &RayEnvManager::GetStateExo,
             R"pbdoc(
                Get the exoskeleton state of the environment.

                Returns:
                    np.ndarray: The exoskeleton state of the environment
            )pbdoc")
        .def("SetParam", &RayEnvManager::SetParam,
             py::arg("param"), py::arg("allow_duplicated_param") = false,
             R"pbdoc(
                Set the dictionary of the parameter state.

                Parameters:
                    param (Dict[str, float]): The dictionary of the parameter state
                    allow_duplicated_param (bool): The flag to allow duplicated parameter (like Waddling and Footdrop both are set)
            )pbdoc")
        .def("SetDeviceZeroState", &RayEnvManager::SetDeviceZeroState,
             py::arg("zero_state"),
             R"pbdoc(
                Set the zero state of the device.

                Parameters:
                    zero_state (bool): The flag to set the zero state
            )pbdoc")
        .def("SetDeviceWeightEnabled", &RayEnvManager::SetDeviceWeightEnabled,
             py::arg("enabled"),
             R"pbdoc(
                Set the weight enabled of the device.

                Parameters:
                    enabled (bool): The flag to set the weight enabled
            )pbdoc")
        .def("SetDeviceWeightForced", &RayEnvManager::SetDeviceWeightForced,
             py::arg("forced"),
             R"pbdoc(
                Set the weight forced of the device.

                Parameters:
                    forced (bool): The flag to set the weight forced
            )pbdoc")
        .def("IsTerminated", &RayEnvManager::IsTerminated,
             R"pbdoc(
                Check if the episode is terminated.

                Returns:
                    bool: The flag to check if the episode is terminated
            )pbdoc")
    ;
}