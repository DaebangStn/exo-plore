#include "ray/Rollout.h"
#include "util/path.h"
#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>


using namespace std;
using namespace Eigen;


Rollout::Rollout(const filesystem::path &torchscript, bool force_use_device, bool do_modulate):
    mEnv(false, force_use_device), mDoModulate(do_modulate)
{
    const auto ckpt_path = (torchscript.is_absolute()) ? torchscript : path_rel_to_abs(torchscript);
    const auto metadata_path = ckpt_path / "metadata.yaml";
    if(filesystem::exists(metadata_path)) mEnv.InitFromYaml(metadata_path.string());
    else{
        cerr << "Error: metadata file not found" << endl;
        exit(1);
    }

    // Load NN
    mNN.useMuscle = mEnv.GetUseMuscle();
    mNN.sim = torch::jit::load(ckpt_path / "sim_nn.pt");
    mNN.sim.to(torch::kCUDA);
    if (mNN.useMuscle) mNN.muscle = torch::jit::load(ckpt_path / "muscle_nn.pt");
}


Rollout::~Rollout() = default;

void Rollout::LoadRecordConfig(const string& record_config, const int cycle)
{
    const auto config_path = path_rel_to_abs(record_config);
    if (!filesystem::exists(config_path)) {
        std::cerr << "Record config file not found: " << config_path << std::endl;
        return;
    }
    try {
        YAML::Node config = YAML::LoadFile(config_path);

        // Ensure the 'record' node exists and is a map
        if (!config["sample"] || !config["sample"].IsMap()) {
            std::cerr << "Invalid or missing 'sample' section in config file: " << config_path << std::endl;
            return;
        }

        YAML::Node sample = config["sample"];
        if (sample["cycle"]) mCycle = sample["cycle"].as<int>();
        if (sample["modulate"]) mDoModulate = sample["modulate"].as<bool>();
        if (sample["zero_state"]) mZeroState = sample["zero_state"].as<bool>();
    }
    catch (const YAML::BadFile& e) {
        std::cerr << "Failed to open record config file: " << config_path << " - " << e.what() << std::endl;
    }
    catch (const YAML::ParserException& e) {
        std::cerr << "Failed to parse record config file: " << e.what() << std::endl;
    }
    catch (const YAML::Exception& e) {
        std::cerr << "Error processing record config file: " << e.what() << std::endl;
    }

    if (!record_config.empty()) mEnv.LoadRecordConfig(record_config);
    if (cycle != -1) cout << "[INFO] Force cycle to " << cycle << endl;
    cout << "[INFO] Modulation=" << (mDoModulate ? "enabled" : "disabled") << " | Zero state=" << (mZeroState ? "enabled" : "disabled") << endl;
}

vector<Record*> Rollout::RunParam(const unordered_map<string, float> &param)
{
    vector<Record*> records;
    const auto record_fields = GetRecordFields();
    mEnv.SetParam(param);
    mEnv.GetDevice()->SetZeroState(mZeroState);
#ifdef LOG_VERBOSE
    cout << endl << "[VERBOSE] Sampling started with metadata:" << endl;
    cout << "K: " << mEnv.GetDeviceK() << " | Delay: " << mEnv.GetDeviceDelay() << endl;
#endif
    auto record = new Record(record_fields);
    records.push_back(record);
    mEnv.Reset();
    while(mEnv.GetCycleCount() <= mCycle)
    {
        step(record);
        if(checkFailure(param))
        {
            delete records.back();
            records.pop_back();
            break;
        }
    }
    return records;
}

void Rollout::step(Record *pRecord)
{
    torch::NoGradGuard no_grad;
    const auto action = mNN.sim.forward({eigen_mat_to_torch_f(mEnv.GetState()), mDoModulate}).toTensor();
    mEnv.SetAction(torch_to_eigen_vec(action));
    for(int i=0;i<mEnv.GetNumSteps();i+=1)
    {
        if (mNN.useMuscle){
            const auto [JtA, JtA_reduced, tau_active, JtP] = mEnv.GetMuscleTuple();
            const auto mt = eigen_vec_to_torch(JtA_reduced);
            const auto dt = eigen_vec_to_torch(tau_active);
            const auto activation = mNN.muscle.forward({mt, dt}).toTensor();
            mEnv.SetActivationLevels(torch_to_eigen_vec(activation));
        }
        mEnv.Step(pRecord);
    }
    // Critical for simulation output
    mEnv.GetReward();
}

bool Rollout::checkFailure(const unordered_map<string, float> &params, const string &msg)
{
    const int failure_code = mEnv.IsEndOfEpisode();
    if(failure_code == 0) return false;

    std::cout << std::endl << "[FAIL] sample failed(code:" << failure_code << ")" << msg;
    std::cout << "      Params: ";
    for (const auto& param : params) std::cout << param.first << ": " << param.second << ", ";
    std::cout << std::endl;
    return true;
}