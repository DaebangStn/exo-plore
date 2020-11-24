#ifndef EXOSKELETON_ROLLOUT_H
#define EXOSKELETON_ROLLOUT_H

#include "core/Environment.h"
#include "core/Record.h"
#include "util/struct.h"
#include <filesystem>
#include <map>
#include <torch/torch.h>
#include <torch/script.h>


class Rollout {
public:
    explicit Rollout(const std::filesystem::path &torchscript, bool force_use_device = false, bool do_modulate = true);
    ~Rollout();
    std::vector<Record*> RunParam(const std::unordered_map<std::string, float> &param);
    void LoadRecordConfig(const std::string &record_config, int cycle = -1);
    std::vector<std::string> GetRecordFields() { return mEnv.GetRecordFields(); }
    void SetShorteningMultiplier(double multiplier) { mEnv.SetShorteningMultiplier(multiplier); }

private:
    void step(Record* pRecord);
    bool checkFailure(const std::unordered_map<std::string, float> &param, const std::string &msg = "");

    MASS::Environment mEnv;
    NN mNN;
    std::optional<torch::jit::Method> forward_with_prev_out_render_jit;
    int mCycle = 1;
    bool mDoModulate = true;
    bool mZeroState = false;
};


#endif //EXOSKELETON_ROLLOUT_H
