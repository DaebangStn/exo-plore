#include "core/Environment.h"
#include "render/GLFWApp.h"
#include "render/GLFunctions.h"
#define STB_IMAGE_IMPLEMENTATION
#include "render/stb_image.h"

#include "glad/glad.h"
#include <GL/glu.h>
#include <GLFW/glfw3.h>
#include "imgui.h"
#include "implot.h"
#include "implot_internal.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include "ImGuiFileDialog.h"
#include "indicators/block_progress_bar.hpp"
#include "indicators/cursor_control.hpp"
#include "dart/external/lodepng/lodepng.h"
#include <vector>
#include <cstring>
#include <filesystem>
#include <ctime>
#include <iomanip>
#include <cmath>

// Helper function for implot (not in older versions)
namespace {
bool GetLastItemHidden() {
    ImPlotContext& gp = *GImPlot;
    if (gp.PreviousItem) return !gp.PreviousItem->Show;
    return false;
}
}

using namespace MASS;
using namespace std;
using namespace dart::dynamics;


GLFWApp::GLFWApp(const RenderArg &args): mArgs(args), 
        mCycleMinMax("angle_HipR", "dev_assist_raw"), 
        mCycleAccumDistance("ma2", "a2", "ma3", "a3", "ma2_leg", "ma2_torso", "ma15",
            "ma2_quadriceps", "ma2_hamstrings", "ma2_he", "ma2_hf", "ma2_ke", "ma2_kf", "ma2_ae", "ma2_af", "ma2_fl",
            // "moment_muscle_hipR", "moment_muscle_kneeR", "moment_muscle_ankleR",
            "bhar", "bhar_activation", "bhar_maintenance", "bhar_shortening", "bhar_mechanical_work",
            "umberger", "umberger_activation", "umberger_maintenance", "umberger_shortening", "umberger_mechanical_work",
            "mine", "houd"
        ),
        mCyclePositiveAccumDistance(
            "power_muscle", "power_muscle_hipR", "power_muscle_kneeR", "power_muscle_ankleR",
            "power_tau", "power_tau_hipR", "power_tau_kneeR", "power_tau_ankleR"
            ),
        mCycleAccumRMS("dev_assist_raw"),
        mCycleAccumStep("rew_gait", "rew_meta", "rew_total", "rew_dev", "rew_loco", "rew_sway", "rew_head", "rew_footstep", "rew_velocity", "footstep", "power_neg", "power_pos"),
        mGraphCycleData("ma2", "a2", "ma3", "a3", "ma2_leg", "ma2_torso", "ma2_weighted", "ma15",
            "angle_HipR_min", "angle_HipR_max", "angle_HipR_range", "dev_assist_raw_min", "dev_assist_raw_max", "dev_assist_raw_range", "footstep",
            "ma2_quadriceps", "ma2_hamstrings", "ma2_he", "ma2_hf", "ma2_ke", "ma2_kf", "ma2_ae", "ma2_af", "ma2_fl",
            "bhar", "bhar_activation", "bhar_maintenance", "bhar_shortening", "bhar_mechanical_work",
            "umberger", "umberger_activation", "umberger_maintenance", "umberger_shortening", "umberger_mechanical_work",
            "mine", "houd",
            "power_muscle", "power_muscle_hipR", "power_muscle_kneeR", "power_muscle_ankleR",
            "power_tau", "power_tau_hipR", "power_tau_kneeR", "power_tau_ankleR",
            // "moment_muscle_hipR", "moment_muscle_kneeR", "moment_muscle_ankleR",
            "rew_gait_step", "rew_meta_step", "rew_total_step", "rew_dev_step", "rew_loco_step", "rew_sway_step", "rew_head_step", "rew_footstep_step", "rew_velocity_step", 
            "power_neg", "power_pos", "dev_assist_raw"
            ),
        mGraphData("contact_phaseR", "contact_GRF_R", "phase_displacement", "phase_lTime", 
            "angle_Obliquity", "angle_Rotation", "angle_HipR", "angle_HipAbR", "angle_HipIRR", "angle_KneeR", "angle_AnkleR", "angle_Tilt", "head_ypos",
            "footstep", "grf_x", "grf_y", "grf_z",
            // "energy_tau", 
            "energy_muscle", "energy_avg",
            "velocity_HipR", "max_joint_velocity", "max_body_velocity", "max_joint_acceleration", "max_body_acceleration",
            "dev_velocity", "dev_angleR", "dev_angleR_raw", "dev_assist_raw", "dev_assist_bw", "dev_power", "power_dev_bw",
            "power_pos", "power_neg", "exo_gain",
            
            "actHe_glt_max", "actHe_sem_bra", "actHe_sem_ten", 
            "actHf_illc", "actHf_psoas", "actHf_rec_fem", 
            "actKe_vas_lat", "actKe_vas_med", 
            "actAe_gas_med", "actAe_gas_lat", "actAe_sol", "actAe_tibp",
            "actAf_tiba", 

            "ma2", "a2", "ma3", "a3", "ma15",
            "ma2_leg", "ma2_torso", "ma2_quadriceps", "ma2_hamstrings", "ma2_he", "ma2_hf", "ma2_ke", "ma2_kf", "ma2_ae", "ma2_af", "ma2_fl",
            "bhar", "bhar_activation", "bhar_maintenance", "bhar_shortening", "bhar_mechanical_work",
            "umberger", "umberger_activation", "umberger_maintenance", "umberger_shortening", "umberger_mechanical_work",
            "mine", "houd",

            "moment_tau_hipR", "moment_tau_hipRx", "moment_tau_kneeR", "moment_tau_ankleR", 
            "moment_muscle_hipR", "moment_muscle_hipRy_ie", "moment_muscle_hipRz_aa", 
            "moment_muscle_hipRx_ef", "moment_muscle_kneeR", "moment_muscle_ankleRx",
            "power_tau", "power_tau_hipR", "power_tau_kneeR", "power_tau_ankleR", "power_muscle", "power_muscle_hipR", "power_muscle_hipRx_ef", "power_muscle_kneeR", "power_muscle_ankleR",
            // "smooth_dt", 

            "force_glt_max", "force_glt_med", "force_quadriceps", "force_hamstrings", "force_tib_a", "force_tib_p", "force_sol_gas", "force_total", 
            
            "rew_loco", "rew_gait", "rew_meta", "rew_meta_act", "rew_meta_torque", "rew_dev", "rew_total", "rew_head", 
            "rew_sway", "rew_avg_vel", "rew_footstep", "rew_velocity", "rew_imit", "rew_imit_pos", "rew_imit_vel",

            "sway_Foot_R", "sway_Foot_RXZ", "sway_Torso_X",
            "head_rot", "head_linacc", "head_rotacc", "head_relvel"),
            mStoredGraphData()
{
    mTrackball.setTrackball(Eigen::Vector2d(mWidth * 0.5, mHeight * 0.5), mWidth * 0.5);
    mTrackball.setQuaternion(Eigen::Quaterniond::Identity());

    _initGLFW();
    _initImGui();
    _initEnv();
}

GLFWApp::~GLFWApp() 
{
    // Clean up video recording
    _stopVideoRecording();
    
    ImPlot::DestroyContext();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(mWindow);
    glfwTerminate();
}

void GLFWApp::_initEnv()
{
    mEnv = new MASS::Environment(true, mArgs.force_device);
    filesystem::path pathRoot = mArgs.torchscript_dir;
    const auto sim_nn_path = pathRoot / "sim_nn.pt";
    const auto muscle_nn_path = pathRoot / "muscle_nn.pt";
    filesystem::path metadata_path = pathRoot / "metadata.yaml";
    if (filesystem::exists(metadata_path)) {
        YAML::Node metadata = YAML::LoadFile(metadata_path.string());
        mParamCfg = metadata["learning"]["param"];
    }else{
        cerr << "Error: metadata file not found" << endl;
        exit(1);
    }
    mEnv->SetGroundSize(1000.0);
    mEnv->InitFromYaml(metadata_path.string());

    cout << "[GLFWApp] useDevice=" << mEnv->GetUseDevice()
         << " useMuscle=" << mEnv->GetUseMuscle() << endl;
    if (mEnv->GetUseDevice()) {
        cout << "[GLFWApp] Device skeleton mass: "
             << mEnv->GetDevice()->GetSkeleton()->getMass() << " kg" << endl;
    }

    mNN.useMuscle = mEnv->GetUseMuscle();
    mNN.sim = torch::jit::load(sim_nn_path);
    mNN.sim.to(torch::kCUDA);
    if (mNN.useMuscle) mNN.muscle = torch::jit::load(muscle_nn_path);

    mNumDof = mEnv->GetCharacter()->GetSkeleton()->getNumDofs();
    mDesiredTorquePrev = Eigen::VectorXd::Zero(mNumDof);
    mMuscleTorque = Eigen::VectorXd::Zero(mNumDof);
    for (const auto& muscle : mEnv->GetCharacter()->GetMuscles()) {
        const auto& name = muscle->GetName();
        mMuscleRenderMap[name] = true;
        mMuscleActivationMap[name] = true;
    }

    Reset();

    BodyNode* groundNode = mEnv->GetGround()->getBodyNode(0);
    mGroundHeight = groundNode->getTransform().translation()[1];
    groundNode->eachShapeNodeWith<dart::dynamics::VisualAspect>([&](const ShapeNode* sn) {
        Eigen::Vector3d size = dynamic_cast<const BoxShape*>(sn->getShape().get())->getSize();
        mGroundHeight += size[1] * 0.5;
    });

    // Initial environment parameters for renderer
    auto env_params = mEnv->GetParam();
    // env_params["reward_device"] = 1;
    // delete all key starts with muscle_force
    for (auto it = env_params.begin(); it != env_params.end(); ) {
        if (it->first.find("muscle_force") != string::npos || it->first.find("muscle_length") != string::npos) {
            it = env_params.erase(it);
        } else ++it;
    }
    mEnv->SetParam(env_params);
    
    int muscleParamSize = mEnv->GetMuscleLengthParams().size() + mEnv->GetMuscleForceParams().size();
    mSelectedParameter = new bool[muscleParamSize]();
    int numMLP = mEnv->GetMuscleLengthParams().size();
    int numMFP = mEnv->GetMuscleForceParams().size();
    for(int i=0; i < numMLP+numMFP; i++) mSelectedParameter[i] = false;
}

void GLFWApp::startLoop() 
{
    const double renderFPS = 60.0;
    double simPrevTime = glfwGetTime();
    double renderPrevTime = glfwGetTime();
    double fpsLastTime = simPrevTime;
    int fpsFrameCount = 0;

    while (!glfwWindowShouldClose(mWindow))
    {
        if (glfwGetTime() - renderPrevTime >= 1.0 / renderFPS) {
            renderWindow();
            renderPrevTime = glfwGetTime();
        }

        if (glfwGetTime() - simPrevTime >= 1.0 / mFramerate) {
            _stepPd();
            fpsFrameCount++;
            simPrevTime = glfwGetTime();
        }
        calculateAndDisplayFPS(fpsLastTime, fpsFrameCount);
    }
}

void GLFWApp::Reset()
{
	mEnv->Reset();
    mMuscleTorque.setZero();
    mRolloutStatus.reset();
    mXmin = 0.0; mXmax = 0.0;
    snprintf(mSearchKey, sizeof(mSearchKey), "%s", "");
}

void GLFWApp::_stepPd(Record* pRecord)
{
    for(int i=0;i<mEnv->GetNumSteps(); i+=1) {
        if (!mRolloutStatus.pause) _step(pRecord);
    }
}

void GLFWApp::_stepPdOnce()
{
    for(int i=0;i<mEnv->GetNumSteps(); i+=1) _step();
}

void GLFWApp::_stepRollout(){
    if (!mEnv->IsGaitCycleComplete()) return;
    
    mRolloutStatus.step();
    if (mRolloutStatus.store) _storeGraphDataKeys(mSearchKey);
    if (mRolloutStatus.pause) {
        if (mRolloutStatus.paramExists()) {
            const auto params = mRolloutStatus.popParams();
            if (mRolloutStatus.store) snprintf(mSearchKey, sizeof(mSearchKey), "%s", mRolloutStatus.storeKey.c_str());
            mEnv->GetDevice()->SetFakeAssist(mRolloutStatus.fake_assist);
            mEnv->SetParam(params);
            cout << ">>> Rollout Status: " << endl;
            for (const auto& [key, value]: params) cout << key << ": " << value << ", ";
            cout << endl;
        }else if (!mRolloutStatus.isAlarmed) {
            mRolloutStatus.isAlarmed = true;
            string cmd = "notify-send \"" + mRolloutStatus.name + " done!\"";
            system(cmd.c_str());
        }
    }
}

void GLFWApp::_testReset(int num_reset){
    cout << "[GLFWApp::_testReset] Resetting " << num_reset << " times" << endl;
    indicators::BlockProgressBar bar{indicators::option::BarWidth{70}, indicators::option::ShowRemainingTime{true}};
    bool rollout_status = mRolloutStatus.pause;
    mRolloutStatus.pause = false;
    for(int i=0; i<num_reset; i++){
        mEnv->Reset();
        for (int j=0; j<20; j++) {
            _stepPd();
        }
        bar.tick();
    }
    mRolloutStatus.pause = rollout_status;
}

void GLFWApp::_step(Record* pRecord){
    torch::NoGradGuard no_grad;
    switch(mControlMode.getMode()){
        case ControlMode::Inference:
            _stepNetwork();
            mEnv->Step(pRecord, &mGraphData, true, true);
            _updateGaitCycleData();
            _summarizeGaitCycleData();
            _stepRollout();        
            if(mEnv->IsControlStep()) mEnv->GetReward(&mGraphData);            
            break;
        case ControlMode::FixUpper:
        case ControlMode::FixWhole:
        default:
            mEnv->Step(nullptr, &mGraphData, false, true);
            break;
    }
}

void GLFWApp::_stepNetwork(){
    if(mEnv->IsControlStep()){
        const auto action = mNN.sim.forward({eigen_mat_to_torch_f(mEnv->GetState()), mRolloutStatus.modulate}).toTensor();
        mEnv->SetAction(torch_to_eigen_vec(action), &mGraphData);
    }
    if (mNN.useMuscle){
        const auto [JtA, JtA_reduced, tau_active, JtP] = mEnv->GetMuscleTuple();
        const auto mt = eigen_vec_to_torch(JtA_reduced);
        const auto dt = eigen_vec_to_torch(tau_active);
        if (mt.isnan().any().item<bool>()) std::cerr << "[GLFWApp::_stepNetwork] Warning: mt contains NaN!" << std::endl;
        if (dt.isnan().any().item<bool>()) std::cerr << "[GLFWApp::_stepNetwork] Warning: dt contains NaN!" << std::endl;
        const auto activation = mNN.muscle.forward({mt, dt}).toTensor();
        Eigen::VectorXd masked_activation = torch_to_eigen_vec(activation);
        
        // Apply muscle activation mask
        const auto& muscles = mEnv->GetCharacter()->GetMuscles();
        for (int i = 0; i < muscles.size() && i < masked_activation.size(); i++) {
            auto it = mMuscleActivationMap.find(muscles[i]->GetName());
            if (it != mMuscleActivationMap.end() && !it->second) {
                masked_activation[i] = 0.0;
            }
        }
        
        mEnv->SetActivationLevels(masked_activation);
    }
}

void GLFWApp::_updateGaitCycleData(){
    if (mAccumData) {
        auto keys = mCycleAccumDistance.keys();
        for (const auto& key : keys) mCycleAccumDistance.accumulate_abs(key, mGraphData.get_last(key));
        keys = mCyclePositiveAccumDistance.keys();
        for (const auto& key : keys) mCyclePositiveAccumDistance.accumulate(key, mGraphData.get_last(key));
        keys = mCycleAccumStep.keys();
        for (const auto& key : keys) mCycleAccumStep.push(key, mGraphData.get_last(key));
        keys = mCycleAccumRMS.keys();
        for (const auto& key : keys) mCycleAccumRMS.accumulate(key, mGraphData.get_last(key));
        keys = mCycleMinMax.keys();
        for (const auto& key : keys) mCycleMinMax.push(key, mGraphData.get_last(key));
    }
}

void GLFWApp::_summarizeGaitCycleData(){
    if (!mEnv->IsGaitCycleComplete()) return;

    if (!mAccumData) {
        mAccumData = true;
        return;
    }

    if (mNumCycRemainFixWeightMA2 > 0) {
        mNumCycRemainFixWeightMA2--;
        if (mNumCycRemainFixWeightMA2 == 0) {
            mFixWeightEnergy = true;
            cout << "[GUI] Fix Energy Weight" << endl;
        }
    }

    if (mDisplayMetric) {
        std::cout << std::left << std::setw(16) << "Metric" << std::right << std::setw(8) << "Value" << std::endl;
        std::cout << std::string(24, '-') << std::endl;
        cout << ">>> Per Distance" << endl;
    }

    auto per_distance = mCycleAccumDistance.divideBy(mEnv->GetCycleDist());

    // compute reward per cycle
    for (const auto& [key, value] : per_distance) {
        if (key.find("rew") != std::string::npos) mGraphCycleData.push(key + "_cycle", value / 400);
        else mGraphCycleData.push(key, value);
        if (mDisplayMetric) std::cout << std::left << std::setw(16) << (key) << 
            std::right << std::setw(8) << std::fixed << std::setprecision(1) << value << std::endl;
    }

    if (mGraphCycleData.key_exists("ma2_weighted") && mEnv->GetUseMuscle()) {
        if (!mFixWeightEnergy) {
            if (per_distance["ma2_he"] < 1e-4 || per_distance["ma2_ke"] < 1e-4 || per_distance["ma2_ae"] < 1e-4 || per_distance["ma2_fl"] < 1e-4) {
                cout << "Unbalanced MA2. Skip computing weighted sum: " << "HE: " << per_distance["ma2_he"] << ", KE: " << per_distance["ma2_ke"] << ", AE: " << per_distance["ma2_ae"] << ", FL: " << per_distance["ma2_fl"] << endl;
            } else {
                mWeightMA2["he"] = mEnergyRatio["Hip"] / per_distance["ma2_he"];
                mWeightMA2["ke"] = mEnergyRatio["Knee"] / per_distance["ma2_ke"];
                mWeightMA2["ae"] = mEnergyRatio["Ankle"] / per_distance["ma2_ae"];
                mWeightMA2["fl"] = mEnergyRatio["Flexions"] / per_distance["ma2_fl"];
                mWeightMA2["total"] = mWeightMA2["he"] + mWeightMA2["ke"] + mWeightMA2["ae"] + mWeightMA2["fl"];
            }
        }
        if (mWeightMA2.size() > 0) {
            const double ma2_weighted = 8 * (per_distance["ma2_he"] * mWeightMA2["he"] + per_distance["ma2_ke"] * mWeightMA2["ke"] + per_distance["ma2_ae"] * mWeightMA2["ae"] + per_distance["ma2_fl"] * mWeightMA2["fl"]) / mWeightMA2["total"];
            mGraphCycleData.push("ma2_weighted", ma2_weighted);
        }
    }

    if (mDisplayMetric) cout << endl << ">>> Per Distance Positive" << endl;
    auto per_distance_positive = mCyclePositiveAccumDistance.divideBy(mEnv->GetCycleDist());
    for (const auto& [key, value] : per_distance_positive) {
        mGraphCycleData.push(key, value);
        if (mDisplayMetric) std::cout << std::left << std::setw(16) << (key) << 
            std::right << std::setw(8) << std::fixed << std::setprecision(1) << value << std::endl;
    }

    if (mDisplayMetric) cout << endl << ">>> Per Step" << endl;
    const auto& per_step = mCycleAccumStep.average();
    for (const auto& [key, value] : per_step) {
        if (key.find("rew") != std::string::npos) mGraphCycleData.push(key + "_step", value);
        else mGraphCycleData.push(key, value);
        if (mDisplayMetric) std::cout << std::left << std::setw(16) << (key) << 
            std::right << std::setw(8) << std::fixed << std::setprecision(1) << value << std::endl;
    }

    if (mDisplayMetric) cout << endl << ">>> Per Step RMS" << endl;
    const auto& keys1 = mCycleAccumRMS.keys();
    for (const auto& key : keys1) {
        const auto& rms = mCycleAccumRMS[key];
        mGraphCycleData.push(key, rms);
        if (mDisplayMetric) std::cout << std::left << std::setw(16) << (key) << 
            std::right << std::setw(8) << std::fixed << std::setprecision(1) << rms << std::endl;
    }

    if (mDisplayMetric) cout << endl << ">>> Min Max" << endl;
    const auto& keys2 = mCycleMinMax.keys(); 
    for (const auto& key : keys2) {
        const auto [min, max] = mCycleMinMax[key];
        mGraphCycleData.push(key + "_min", min); mGraphCycleData.push(key + "_max", max); mGraphCycleData.push(key + "_range", max - min); 
        if (mDisplayMetric) std::cout << std::left << std::setw(16) << (key) << 
            std::right << std::setw(8) << std::fixed << std::setprecision(1) << min << " " << max << std::endl;
    }

    if (mDisplayMetric) {cout << std::endl; cout.unsetf(std::ios::fixed);}

    _resetAccumData();
}

void GLFWApp::TickControlMode(){
    mControlMode.nextMode();
    switch(mControlMode.getMode()){
        case ControlMode::FixUpper:
            mEnv->GetCharacter()->SetAnchor();
            mEnv->GravityOff();
            cout << "[GUI] Control Mode: Fix Upper" << endl;
            break;
        case ControlMode::FixWhole:
            mEnv->GetCharacter()->SetAnchor(false);
            mEnv->GravityOff();
            cout << "[GUI] Control Mode: Fix Whole" << endl;
            break;
        case ControlMode::Inference:
            mEnv->GetCharacter()->ResetAnchor();
            mEnv->GravityOn();
            cout << "[GUI] Control Mode: Inference" << endl;
            break;
        default:
            break;
    }
}

vector<Record*> GLFWApp::exportData(const unordered_map<string, float>& params, int cycle, bool render, bool do_modulate)
{
    vector<Record*> records;
    const auto record_fields = mEnv->GetRecordFields();
    mEnv->SetParam(params);
    auto record = new Record(record_fields);
    records.push_back(record);
    mEnv->Reset();
    mRolloutStatus.modulate = do_modulate;
    render ? (std::cout << std::endl << "======Run====== " << std::endl, 0) : 0;
    // Actual run
    while(mEnv->GetCycleCount() <= cycle)
    {
        _stepPd(record);
        render ? (renderWindow(), 0) : 0;
        if(checkFailed())
        {
            delete records.back();
            records.pop_back();
            break;
        }
    }
    render ? (std::cout << "======Done====== " << std::endl, 0) : 0;
    return records;
}


bool GLFWApp::checkFailed(const std::string& msg) const
{
    const int failure_code = mEnv->IsEndOfEpisode();
    if (failure_code == 0)
    {
        return false;
    }
    std::cout << std::endl << "[FAIL] sample failed(code:" << failure_code << ") : "
    << msg << std::endl;
    std::cout << "Params: ";
    const auto params = mEnv->GetParam();
    for (const auto& param : params)
    {
        std::cout << param.first << ": " << param.second << ", ";
    }
    std::cout << std::endl;
    return true;
}



// ======================================== GUI (glfw, imgui, implot) ========================================

void GLFWApp::_initGLFW()
{
    _loadWindowSetting();

    glfwInit();
    
    glfwWindowHint(GLFW_SAMPLES, 0);
    glfwWindowHint(GLFW_RED_BITS, 8);             // 8-bit red channel
    glfwWindowHint(GLFW_GREEN_BITS, 8);           // 8-bit green channel  
    glfwWindowHint(GLFW_BLUE_BITS, 8);            // 8-bit blue channel
    glfwWindowHint(GLFW_ALPHA_BITS, 8);           // 8-bit alpha channel
    glfwWindowHint(GLFW_DEPTH_BITS, 24);          // 24-bit depth buffer
    glfwWindowHint(GLFW_STENCIL_BITS, 8);         // 8-bit stencil buffer
    glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_TRUE); // Double buffering
    
    glfwWindowHintString(GLFW_X11_CLASS_NAME, "MuscleSim");
    glfwWindowHint(GLFW_FOCUSED, GLFW_FALSE);
    mWindow = glfwCreateWindow(mWidth, mHeight, "MuscleSim", nullptr, nullptr);
    glfwSetWindowPos(mWindow, mXpos, mYpos);
    glfwSetWindowTitle(mWindow, mArgs.torchscript_dir.c_str());
    int icon_width, icon_height, icon_channels;
    unsigned char* icon_image = stbi_load("data/res/thumbnail.png", &icon_width, &icon_height, &icon_channels, 4);
    if (icon_image) {
        GLFWimage icons[1];
        icons[0].width = icon_width;
        icons[0].height = icon_height;
        icons[0].pixels = icon_image;
        glfwSetWindowIcon(mWindow, 1, icons);
        stbi_image_free(icon_image);
    } else {
        cout << "Failed to load icon image: " << stbi_failure_reason() << endl;
    }

    if (mWindow == NULL) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    glfwMakeContextCurrent(mWindow);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_MULTISAMPLE);

    // Check and log MSAA samples
    GLint samples;
    glGetIntegerv(GL_SAMPLES, &samples);
    std::cout << "[Video Quality] MSAA samples: " << samples << " (GL_MULTISAMPLE disabled)" << std::endl;
    
    glViewport(0, 0, mWidth, mHeight);
    glfwSetWindowUserPointer(mWindow, this);

    auto framebufferSizeCallback = [](GLFWwindow* window, int mWidth, int mHeight) {
        GLFWApp* app = static_cast<GLFWApp*>(glfwGetWindowUserPointer(window));
        app->mWidth = mWidth;
        app->mHeight = mHeight;
        glViewport(0, 0, mWidth, mHeight);
    };
    glfwSetFramebufferSizeCallback(mWindow, framebufferSizeCallback);
    
    auto keyCallback = [](GLFWwindow* window, int key, int scancode, int action, int mods) {
        auto& io = ImGui::GetIO();
        if (!io.WantCaptureKeyboard) {
            GLFWApp* app = static_cast<GLFWApp*>(glfwGetWindowUserPointer(window));
            app->keyboardPress(key, scancode, action, mods);
        }
    };
    glfwSetKeyCallback(mWindow, keyCallback);
    
    auto cursorPosCallback = [](GLFWwindow* window, double xpos, double ypos) {
        auto& io = ImGui::GetIO();
        if (!io.WantCaptureMouse) {
            GLFWApp* app = static_cast<GLFWApp*>(glfwGetWindowUserPointer(window));
            app->mouseMove(xpos, ypos);
        }
    };
    glfwSetCursorPosCallback(mWindow, cursorPosCallback);
    
    auto mouseButtonCallback = [](GLFWwindow* window, int button, int action, int mods) {
        auto& io = ImGui::GetIO();
        if (!io.WantCaptureMouse) {
            GLFWApp* app = static_cast<GLFWApp*>(glfwGetWindowUserPointer(window));
            app->mousePress(button, action, mods);
        }
    };
    glfwSetMouseButtonCallback(mWindow, mouseButtonCallback);
    
    auto scrollCallback = [](GLFWwindow* window, double xoffset, double yoffset) {
        auto& io = ImGui::GetIO();
        if (!io.WantCaptureMouse) {
            GLFWApp* app = static_cast<GLFWApp*>(glfwGetWindowUserPointer(window));
            app->mouseScroll(xoffset, yoffset);
        }
    };
    glfwSetScrollCallback(mWindow, scrollCallback);
}


void GLFWApp::_initImGui()
{
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(mWindow, true);
    ImGui_ImplOpenGL3_Init("#version 150");
    ImPlot::CreateContext();

}

GLFWmonitor* get_current_monitor(GLFWwindow* window) {
    int windowX, windowY, windowWidth, windowHeight;
    glfwGetWindowPos(window, &windowX, &windowY);
    glfwGetWindowSize(window, &windowWidth, &windowHeight);

    int bestArea = 0;
    GLFWmonitor* bestMonitor = nullptr;

    int monitorCount;
    GLFWmonitor** monitors = glfwGetMonitors(&monitorCount);

    for (int i = 0; i < monitorCount; ++i) {
        int monitorX, monitorY;
        glfwGetMonitorPos(monitors[i], &monitorX, &monitorY);
        const GLFWvidmode* mode = glfwGetVideoMode(monitors[i]);

        int overlapX = std::max(0, std::min(windowX + windowWidth, monitorX + mode->width) - std::max(windowX, monitorX));
        int overlapY = std::max(0, std::min(windowY + windowHeight, monitorY + mode->height) - std::max(windowY, monitorY));
        int area = overlapX * overlapY;

        if (area > bestArea) {
            bestArea = area;
            bestMonitor = monitors[i];
        }
    }

    return bestMonitor;
}

void GLFWApp::calculateAndDisplayFPS(double& lastTime, int& frameCount) {
    double currentTime = glfwGetTime();
    double delta = currentTime - lastTime;

    if (delta >= 1.0) { // Update every second
        double fps = double(frameCount) / delta;
        mMeasuredFps = fps;
        // Reset for next calculation
        frameCount = 0;
        lastTime = currentTime;
    }
}

void GLFWApp::renderWindow()
{
    glfwPollEvents();
    DrawSimFrame();
    _recordVideoFrame();
    _drawGui();
    glfwSwapBuffers(mWindow);
}

void GLFWApp::mouseMove(double xpos, double ypos) {
    double deltaX = xpos - mMouseX;
    double deltaY = ypos - mMouseY;
    mMouseX = xpos;
    mMouseY = ypos;

    if (mRotate)
    {
        if (deltaX != 0 || deltaY != 0) {
            mTrackball.updateBall(xpos, mHeight - ypos);
        }
    }

    if (mTranslate)
    {
        Eigen::Matrix3d rot;
        rot = mTrackball.getRotationMatrix();
        mTrans += (1 / mZoom) * rot.transpose() * Eigen::Vector3d(deltaX, -deltaY, 0.0);
    }

    if (mZooming)
    {
        mZoom = std::max(0.01, mZoom + deltaY * 0.01);
    }
}

void GLFWApp::mousePress(int button, int action, int mods) 
{
    if (action == GLFW_PRESS) {
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            mRotate = true;
            mTrackball.startBall(mMouseX, mHeight - mMouseY);
        }
        else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
            mTranslate = true;
        }
    }
    else if (action == GLFW_RELEASE) {
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            mRotate = false;
        }
        else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
            mTranslate = false;
        }
    }
}

void GLFWApp::_resetAccumData()
{
    mCycleAccumDistance.reset(); mCyclePositiveAccumDistance.reset(); mCycleAccumStep.reset();
    mCycleMinMax.reset(); mCycleAccumRMS.reset();
}

void GLFWApp::_resetPlotData()
{
    mGraphData.reset(); mGraphCycleData.reset();
    mAccumData = false;
    _resetAccumData();
}

void GLFWApp::_tickDrawMode()
{
    mDrawMode.nextMode();
    switch(mDrawMode.getMode())
    {
        case DrawMode::Mesh:
        case DrawMode::Primitive:
            glfwSwapInterval(1);
            break;
        case DrawMode::NoSimUpdate:
            glfwSwapInterval(0);
            break;
        default:
            break;
    }
}

void GLFWApp::keyboardPress(int key, int scancode, int action, int mods) 
{
    if(action == GLFW_PRESS) 
    {
        switch (key) 
        {
            case GLFW_KEY_SPACE: {
                mRolloutStatus.pause = !mRolloutStatus.pause; 
                mRolloutStatus.cycle = -1;
                break;
            }
            case GLFW_KEY_R: 
                if (mods & GLFW_MOD_SHIFT) _resetPlotData();
                else Reset();
                break;
            case GLFW_KEY_C: 
                if (mods & GLFW_MOD_SHIFT) mDrawCollision = !mDrawCollision;
                else TickControlMode();
                break;
            case GLFW_KEY_S: 
                if (mods & GLFW_MOD_SHIFT) _step();
                else _stepPdOnce();
                break;
            case GLFW_KEY_T: mDrawTitle = !mDrawTitle; break;
            case GLFW_KEY_D: _tickDrawMode(); break;
            case GLFW_KEY_F: mFocusMode.nextMode(); break;
            case GLFW_KEY_A: mIncludeAction++; mIncludeAction%=3; break;
            case GLFW_KEY_W: mEnv->GetDevice()->ToggleZeroState(); break;
            case GLFW_KEY_Z: 
            {
                mCameraMoving = 0;
                mTrackball.setQuaternion(Eigen::Quaterniond::Identity());
                mZoom = 0.25;
                break;
            }
            case GLFW_KEY_Y:  // sagittal view
            {
                mCameraMoving = 0;
                Eigen::Quaterniond q = Eigen::Quaterniond(Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d::UnitY()));
                mTrackball.setQuaternion(q);
                mZoom = 0.3;
                break;
            }
            case GLFW_KEY_KP_0: 
            {
                SkeletonPtr skel = mEnv->GetCharacter()->GetSkeleton();
                Utils::setSkelPos(skel,mEnv->GetCharacter()->GetMirrorPosition(skel->getPositions()));                
                break;
            }
            case GLFW_KEY_7: 
            case GLFW_KEY_KP_7: 
                mCameraMoving = 1; 
                break;
            case GLFW_KEY_5: 
            case GLFW_KEY_KP_5: 
                mCameraMoving = 0; 
                mTrackball.setQuaternion(Eigen::Quaterniond::Identity()); 
                break;
            case GLFW_KEY_9: 
            case GLFW_KEY_KP_9:
                mCameraMoving = -1; 
                break;
            case GLFW_KEY_4:
            case GLFW_KEY_KP_4:
            {
                mCameraMoving = 0;
                Eigen::Quaterniond r = Eigen::Quaterniond(Eigen::AngleAxisd(0.01 * M_PI, Eigen::Vector3d::UnitY()))  * mTrackball.getCurrQuat();
                mTrackball.setQuaternion(r);
                break;
            }
            case GLFW_KEY_6:
            case GLFW_KEY_KP_6:
            {
                mCameraMoving = 0;
                Eigen::Quaterniond r = Eigen::Quaterniond(Eigen::AngleAxisd(-0.01 * M_PI, Eigen::Vector3d::UnitY()))  * mTrackball.getCurrQuat();
                mTrackball.setQuaternion(r);
                break;
            }
            case GLFW_KEY_8:
            case GLFW_KEY_KP_8:
            {
                mCameraMoving = 0;
                Eigen::Quaterniond r = Eigen::Quaterniond(Eigen::AngleAxisd(0.01 * M_PI, Eigen::Vector3d::UnitX()))  * mTrackball.getCurrQuat();
                mTrackball.setQuaternion(r);
                break;
            }
            case GLFW_KEY_1:
            case GLFW_KEY_KP_1:
            {
                mCameraMoving = 0;
                mTrackball.setQuaternion(Eigen::Quaterniond(0.917, 0, 0.4, 0));
                break;
            }
            case GLFW_KEY_2:
            case GLFW_KEY_KP_2:
            {
                mCameraMoving = 0;
                Eigen::Quaterniond r = Eigen::Quaterniond(Eigen::AngleAxisd(-0.01 * M_PI, Eigen::Vector3d::UnitX()))  * mTrackball.getCurrQuat();
                mTrackball.setQuaternion(r);
                break;
            }
            case GLFW_KEY_I:
            {
                std::cout << "Skeleton Information" << std::endl;
                mDrawNodeCOM = true;
                mDrawJoint = true;
                break;
            }
            case GLFW_KEY_G:
                mGroundMode.nextMode();
                break;
            case GLFW_KEY_V:
                if (mods & GLFW_MOD_SHIFT) {
                    // Stop video recording
                    if (mVideoRecording) {
                        _stopVideoRecording();
                    }
                } else {
                    // Start video recording
                    if (!mVideoRecording) {
                        std::string timestamp = std::to_string(std::time(nullptr));
                        std::string filename = "video_" + timestamp + ".mp4";
                        strncpy(mVideoFilename, filename.c_str(), sizeof(mVideoFilename));
                        _startVideoRecording(mVideoFilename, mVideoFPS);
                    }
                }
                break;
            case GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose(mWindow, true);
                break;
            default:
                break;
        }
    }
}

void GLFWApp::_loadWindowSetting()
{
    const auto config_path = "render.yaml";
    if (filesystem::exists(config_path)) {
        cout << "Loading render config from " << config_path << endl;
        YAML::Node config = YAML::LoadFile(config_path);
        if (config["geometry"] && config["geometry"].IsMap()) {
            YAML::Node geometry = config["geometry"];
            if (geometry["window"] && geometry["window"].IsMap()) {
                YAML::Node window = geometry["window"];
                mWidth = window["width"] ? window["width"].as<int>() : mWidth;
                mHeight = window["height"] ? window["height"].as<int>() : mHeight;
                mXpos = window["xpos"] ? window["xpos"].as<int>() : mXpos;
                mYpos = window["ypos"] ? window["ypos"].as<int>() : mYpos;
            }
            mControlWidth = geometry["control"] ? geometry["control"].as<int>() : mControlWidth;
            mPlotWidth = geometry["plot"] ? geometry["plot"].as<int>() : mPlotWidth;
        }
        if (config["render"] && config["render"].IsMap()) {
            YAML::Node render = config["render"];
            if (render["muscle"] && render["muscle"].IsScalar()) {
                mDrawMuscleTotal = render["muscle"].as<bool>();
            }
            if (render["device"] && render["device"].IsScalar()) {
                mDrawDevice = render["device"].as<bool>();
            }
            if (render["plot"] && render["plot"].IsMap()) {
                YAML::Node plot = render["plot"];
                mPlotTitle = plot["title"] ? plot["title"].as<bool>() : mPlotTitle;
                mPlotAverage = plot["average"] ? plot["average"].as<bool>() : mPlotAverage;
                mPlotDiff = plot["diff"] ? plot["diff"].as<bool>() : mPlotDiff;
                mXmin = plot["x_min"] ? plot["x_min"].as<double>() : mXmin;
                mPlotAvgTxtInterval = plot["avg_window"] ? plot["avg_window"].as<int>() : mPlotAvgTxtInterval;
            }
            if (render["rollout"] && render["rollout"].IsMap()) {
                YAML::Node rollout = render["rollout"];
                mRolloutCount = rollout["count"] ? rollout["count"].as<int>() : mRolloutCount;
            }
            if (render["capture"] && render["capture"].IsMap()) {
                YAML::Node capture = render["capture"];
                if (capture["maxtime"] && capture["maxtime"].IsScalar()) {
                    mVideoMaxTime = capture["maxtime"].as<double>();
                }    
                if (capture["x0"]) mCaptureX0 = capture["x0"].as<int>();
                if (capture["y0"]) mCaptureY0 = capture["y0"].as<int>();
                if (capture["x1"]) mCaptureX1 = capture["x1"].as<int>();
                if (capture["y1"]) mCaptureY1 = capture["y1"].as<int>();

                // Optional support for arrays: top_left: [x0,y0], bottom_right: [x1,y1]
                if (capture["top_left"] && capture["top_left"].IsSequence() && capture["top_left"].size() >= 2) {
                    mCaptureX0 = capture["top_left"][0].as<int>();
                    mCaptureY0 = capture["top_left"][1].as<int>();
                }
                if (capture["bottom_right"] && capture["bottom_right"].IsSequence() && capture["bottom_right"].size() >= 2) {
                    mCaptureX1 = capture["bottom_right"][0].as<int>();
                    mCaptureY1 = capture["bottom_right"][1].as<int>();
                }
            }
        }
    }
}

void GLFWApp::_loadRolloutSetting(const std::string& path)
{
    const auto config_path = filesystem::absolute(path);
    if (!filesystem::exists(config_path)) {
        std::cout << "[Warning] There is no " << path << " at " << std::filesystem::current_path() << std::endl;
        return;
    }

    try {
        // First read the raw file contents
        std::ifstream file(config_path);
        std::stringstream buffer;
        buffer << file.rdbuf();
        mRolloutStatus.fileContents = buffer.str();
        file.close();
        
        // Then parse as YAML
        YAML::Node config = YAML::LoadFile(config_path.string());
        mRolloutStatus.settingPath = path;

        if (!config["params"] || !config["params"].IsSequence()) {
            std::cout << "[Warning] There is no params section in " << path << std::endl;
            return;
        }
        auto params_vec = config["params"].as<std::vector<YAML::Node>>();
        std::deque<YAML::Node> params(params_vec.begin(), params_vec.end());
        cout << "[GUI] Loading " << params.size() << " params" << std::endl;
        mRolloutStatus.params = params;
        if (config["name"] && config["name"].IsScalar()) mRolloutStatus.name = config["name"].as<std::string>();

        mRolloutStatus.cycle = 0;
        mRolloutStatus.pause = false;
        mRolloutStatus.isAlarmed = false;

        mStoredGraphData.clear(); // clear stored graph data

        if (config["plot"] && config["plot"].IsSequence()) {
            const auto plot = config["plot"];
            mXmin = plot[0].as<double>(); mXmax = plot[1].as<double>();
        }
    }
    catch (const YAML::BadFile& e) {
        std::cerr << "Failed to open record config: " << config_path << " - " << e.what() << std::endl;
    }
    catch (const YAML::ParserException& e) {
        std::cerr << "Failed to parse record config: " << e.what() << std::endl;
    }
    catch (const YAML::Exception& e) {
        std::cerr << "Error processing record config: " << e.what() << std::endl;
    }
}

void GLFWApp::SetFocusing()
{
    if(mFocusMode.getMode() == FocusMode::Root)
    {
        mTrans = -mEnv->GetWorld()->getSkeleton("Human")->getRootBodyNode()->getCOM();
        if (!mEnv->GetUseTerrain()) mTrans[1] = -1.0;
        mTrans *= 1000;
    }else if(mFocusMode.getMode() == FocusMode::Foot){
        mTrans = -mEnv->GetWorld()->getSkeleton("Human")->getRootBodyNode()->getCOM();
        mTrans[1] = -0.01;
        mTrans *= 1000;
        Eigen::Quaterniond q = Eigen::Quaterniond(Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d::UnitY()));
        mTrackball.setQuaternion(q);
        mZoom = 0.5;
    }

	Eigen::Quaterniond origin_r = mTrackball.getCurrQuat();
	if (mCameraMoving == 1 && Eigen::AngleAxisd(origin_r).angle() < 0.5 * M_PI)
	{
		Eigen::Quaterniond r = Eigen::Quaterniond(Eigen::AngleAxisd(mCameraMoving * 0.01 * M_PI, Eigen::Vector3d::UnitY())) * origin_r;
		mTrackball.setQuaternion(r);
	}
	else if (mCameraMoving == -1 && Eigen::AngleAxisd(origin_r).axis()[1] > 0)
	{
		Eigen::Quaterniond r = Eigen::Quaterniond(Eigen::AngleAxisd(mCameraMoving * 0.01 * M_PI, Eigen::Vector3d::UnitY())) * origin_r;
		mTrackball.setQuaternion(r);
	}
}

void GLFWApp::_drawGui() 
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    _drawController();
    _drawSimData();
    _drawPhase();
    if (mDrawTitle) DrawTitle();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void GLFWApp::_drawController()
{
    double displayH = mHeight - 20;
    double displayPosX = 10;
    double displayPosY = 10;

    auto params = mEnv->GetParam();
    const auto paramCopy = params;

    ImGui::SetNextWindowSize(ImVec2(mControlWidth, displayH), ImGuiCond_Once);
    ImGui::SetNextWindowPos(ImVec2(displayPosX, displayPosY), ImGuiCond_Once);
    ImGui::Begin("Controller");

    if(ImGui::CollapsingHeader("Rollout", ImGuiTreeNodeFlags_DefaultOpen)){


        // contact debouncer setting
        bool use_debouncer = mEnv->GetContact()->getUseDebouncer();
        if(ImGui::Checkbox("Debouncer", &use_debouncer)) mEnv->GetContact()->setUseDebouncer(use_debouncer);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(40);
        float debouncer_alpha = (float)mEnv->GetContact()->getDebouncerAlpha();
        if(ImGui::DragFloat("alp", &debouncer_alpha, 0.001, 0.0, 1.0)) mEnv->GetContact()->setDebouncerAlpha((double)debouncer_alpha);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(40);
        float grf_phase_change_ratio = (float)mEnv->GetGRFPhaseChangeRatio();
        if(ImGui::DragFloat("GRF Ratio", &grf_phase_change_ratio, 0.001, 0.0, 1.0)) mEnv->SetGRFPhaseChangeRatio((double)grf_phase_change_ratio);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(40);
        float step_min_ratio = (float)mEnv->GetStepMinRatio();
        if(ImGui::DragFloat("Step Ratio", &step_min_ratio, 0.001, 0.0, 1.0)) mEnv->SetStepMinRatio((double)step_min_ratio);

        ImGui::Checkbox("Mod", &mRolloutStatus.modulate);
        ImGui::SameLine();
        ImGui::Checkbox("Fix MA2 Weight", &mFixWeightEnergy);
        ImGui::SetNextItemWidth(75);
        ImGui::InputInt("Cycle", &mRolloutCount, 10);
        ImGui::SameLine();
        if(ImGui::Button("Run")) {
            // mDrawMode = DrawMode::NoSimUpdate;
            mRolloutStatus.cycle = mRolloutCount;
            mRolloutStatus.pause = false;
            mRolloutStatus.settingPath = "";
            glfwSwapInterval(0);
        }
        ImGui::SameLine();

        if(ImGui::Button("File")){
            IGFD::FileDialogConfig config;
            config.path = "./data/paramsGui";
            ImGuiFileDialog::Instance()->OpenDialog(
                "LoadRolloutFile", 
                "Choose Rollout Config File", 
                ".yaml,.yml", 
                config
            );
        }
        // Display file dialog
        if (ImGuiFileDialog::Instance()->Display("LoadRolloutFile")) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                std::string filePathName = ImGuiFileDialog::Instance()->GetFilePathName();
                _loadRolloutSetting(filePathName);
                mDrawMode = DrawMode::NoSimUpdate;
                glfwSwapInterval(0);
            }
            ImGuiFileDialog::Instance()->Close();
        }

        // ImGui::SameLine();
        // if(ImGui::Button("delay")) _loadRolloutSetting("data/paramsGui/delay.yaml");
        // ImGui::SameLine();
        // if(ImGui::Button("delay1")) _loadRolloutSetting("data/paramsGui/delay1.yaml");
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(80);
    ImGui::InputText("Memo", mRolloutStatus.memo, IM_ARRAYSIZE(mRolloutStatus.memo));

    if(ImGui::CollapsingHeader("Plot", ImGuiTreeNodeFlags_DefaultOpen)){
        ImGui::SetNextItemWidth(30);
        ImGui::InputDouble("X min", &mXmin); ImGui::SameLine();
        ImGui::SetNextItemWidth(30);
        ImGui::InputDouble("X max", &mXmax); ImGui::SameLine();
        ImGui::SetNextItemWidth(30);
        if (ImGui::Button("HS")) {_setXminToHeelStrike();}; ImGui::SameLine();
        // if (ImGui::Button("1.1")) {mXmine = -1.1; mXmax = 0.0;}; ImGui::SameLine();
        if (ImGui::Button("20")) {mXmin = -20.0; mXmax = 0.0;}; ImGui::SameLine();
        if (ImGui::Button("Reset")) {mXmin = 0.0; mXmax = 0.0;};

        mPlotHideLegend = ImGui::Button("Hide"); ImGui::SameLine();
        ImGui::Checkbox("Title", &mPlotTitle); ImGui::SameLine();
        ImGui::Checkbox("Avg", &mPlotAverage); ImGui::SameLine();
        ImGui::Checkbox("Diff", &mPlotDiff); ImGui::SameLine();
        if(ImGui::Checkbox("Jit", &mPlotJitter)) mPlotSmoothAlpha = 0.1f; 
        ImGui::SameLine();
        ImGui::SetNextItemWidth(40);
        ImGui::InputFloat("Alpha", &mPlotSmoothAlpha);

        ImGui::SetNextItemWidth(100);
        ImGui::InputText("Key", mSearchKey, IM_ARRAYSIZE(mSearchKey)); ImGui::SameLine();
        
        if (ImGui::Button("Print")) _printGraphDataKeys(mSearchKey); ImGui::SameLine();
        if (ImGui::Button("PrintC")) _printContactPhase(); ImGui::SameLine();
        if (ImGui::Button("Store")) _storeGraphDataKeys(mSearchKey, true); ImGui::SameLine();
        if (ImGui::Button("Show")) _showGraphDataKeys(mSearchKey, true); ImGui::SameLine();
        if (ImGui::Button("Clear")) mStoredGraphData.clear(); ImGui::SameLine();
        ImGui::SetNextItemWidth(15);
        ImGui::InputInt("Ofs", &mStoreColorOffset, 0, 10);
    }

    if (mEnv->GetUseTerrain()) {
        if(ImGui::CollapsingHeader("Terrain", ImGuiTreeNodeFlags_DefaultOpen)){
            ImGui::TextUnformatted("Param"); ImGui::SameLine();
            
            auto terrain_manager = mEnv->GetTerrainManager();
            auto config = terrain_manager->getConfig();
            bool config_changed = false;

            // Terrain mode combo box
            const char* terrain_modes[] = { "Flat", "Stair" };
            int terrain_mode = static_cast<int>(config.mode);
            ImGui::SetNextItemWidth(100);
            if (ImGui::Combo("Mode", &terrain_mode, terrain_modes, IM_ARRAYSIZE(terrain_modes))) {
                config.mode = static_cast<TerrainMode>(terrain_mode);
                config_changed = true;
            }

            // Width parameter
            ImGui::SetNextItemWidth(100);
            if (ImGui::InputDouble("Width", &config.width, 0.01, 0.1, "%.3f")) {
                config_changed = true;
            }
            ImGui::SameLine();
            // Margin parameter
            ImGui::SetNextItemWidth(100);
            if (ImGui::InputDouble("Margin", &config.margin, 0.01, 0.1, "%.3f")) {
                config_changed = true;
            }

            // Stair mHeight (only shown when in STAIR mode)
            if (config.mode == TerrainMode::STAIR) {
                ImGui::SetNextItemWidth(100);
                if (ImGui::InputDouble("Stair Height", &config.stairHeight, 0.01, 0.1, "%.3f")) {
                    config_changed = true;
                }
            }

            if (config_changed) {
                terrain_manager->setConfig(config);
                // You might need to reset or update the terrain after changing config
                terrain_manager->reset(mEnv->GetCharacter()->GetSkeleton()->getRootBodyNode()->getCOM());
            }
        }
    }

    if(ImGui::CollapsingHeader("Energy Reward", ImGuiTreeNodeFlags_DefaultOpen)){
        Energy* energy = mEnv->GetEnergy();
        const char* skel_jnt_types[] = { "Torque", "Full Muscle", "Lower Muscle" };
        int skel_jnt_type = static_cast<int>(energy->GetJntType());
        ImGui::SetNextItemWidth(100);
        if(ImGui::Combo("Jnt Type", &skel_jnt_type, skel_jnt_types, IM_ARRAYSIZE(skel_jnt_types))) energy->SetJntType(static_cast<JntType>(skel_jnt_type));
        ImGui::SameLine();
        ImGui::SetNextItemWidth(100);
        const char* energy_reward_curves[] = { "EXP", "LINEAR1", "LINEAR2", "LINEAR3" };
        int energy_reward_curve = (int)energy->GetEnergyRewardCurve();
        if(ImGui::Combo("Curve", &energy_reward_curve, energy_reward_curves, IM_ARRAYSIZE(energy_reward_curves))) energy->SetEnergyRewardCurve(static_cast<EnergyRewardCurve>(energy_reward_curve));

        if (energy->GetJntType() >= JntType::FullMuscle) {
            const char* muscle_mass_types[] = { "MT0", "M0", "M_Handsfield14" };
            int muscle_mass_type = mEnv->GetCharacter()->GetMuscleMassType();
            ImGui::SetNextItemWidth(100);
            if (ImGui::Combo("Mass Type", &muscle_mass_type, muscle_mass_types, IM_ARRAYSIZE(muscle_mass_types))) mEnv->GetCharacter()->SetMuscleMassType(muscle_mass_type);

            ImGui::SameLine();
        }

        const char* energy_modes[] = { 
            "A2", "MA2", "A3", 
            "MA3", "A", "MA",
            "M2A", "M2A2", "M2A3",
            "M05A", "M05A2", "M05A3",
            "A15", "M05A15", "MA15", "M2A15", 
            "BHAR04",
            "A125", "M05A125", "MA125", "M2A125",
            // "MA2COT", "MA2COT2",  "A2ANK", "MA2ANK",
            "POWER", "TAU", 
        };
        int energy_mode = (int)energy->GetEnergyMode();
        ImGui::SetNextItemWidth(50);
        if (ImGui::Combo("Mode", &energy_mode, energy_modes, IM_ARRAYSIZE(energy_modes))) energy->SetEnergyMode((EnergyMode)energy_mode);
        
        ImGui::SetNextItemWidth(50);
        float act_rew_coeff = (float)energy->GetActRewCoeff();
        if(ImGui::DragFloat("Act", &act_rew_coeff, 0.0, 100.0)) energy->SetActRewCoeff((double)act_rew_coeff);
        ImGui::SameLine();

        ImGui::SetNextItemWidth(50);
        float torque_rew_coeff = (float)energy->GetTorqueRewCoeff();
        if(ImGui::DragFloat("Tau", &torque_rew_coeff, 0.0, 0.01)) energy->SetTorqueRewCoeff((double)torque_rew_coeff);
        ImGui::SameLine();

        ImGui::SetNextItemWidth(50);
        float target_coeff = (float)energy->GetTargetCoeff();
        if(ImGui::DragFloat("Target", &target_coeff, 0.0, 100.0)) energy->SetTargetCoeff((double)target_coeff);

        if(mEnv->GetUseMuscle()) {
            ImGui::SameLine();
            ImGui::SetNextItemWidth(50);
            float shortening_multiplier = (float)mEnv->GetShorteningMultiplier();
            if(ImGui::DragFloat("Short.Mult (Bhar)", &shortening_multiplier, 0.1, 0.0, 100.0)) mEnv->SetShorteningMultiplier((double)shortening_multiplier);
        }

        ImGui::SetNextItemWidth(50);
        double ankle_stiffness = mEnv->GetAnkleStiffness();
        if(ImGui::InputDouble("Ankle (K)", &ankle_stiffness, 0.0, 100.0)) mEnv->SetAnkleStiffness((double)ankle_stiffness);

        ImGui::SameLine();
        ImGui::SetNextItemWidth(50);
        double action_alpha = mEnv->GetActionAlpha();
        if(ImGui::InputDouble("Action", &action_alpha, 0.0, 1.0)) mEnv->SetActionAlpha((double)action_alpha);
    }

    double override_k = -1, override_delay = -1;
    const double current_k = params["k"], current_delay = params["delay"];
    if(mEnv->GetUseDevice() && ImGui::CollapsingHeader("Exo", ImGuiTreeNodeFlags_DefaultOpen)){
        
        // Define k values and delays
        const std::vector<float> k_values = {0.2667f, 0.5f};  // K1 and K2
        const std::vector<float> delays = {0.05f, 0.15f, 0.25f, 0.35f};
        
        // Calculate cell mWidth and mHeight
        float cell_width = 25.0f;
        float cell_height = ImGui::GetTextLineHeight() + 4.0f;
        
        // Create table header
        ImGui::BeginTable("Params", delays.size() + 1, ImGuiTableFlags_Borders);
        
        // Empty top-left cell
        ImGui::TableNextColumn();
        ImGui::TextUnformatted("K/Delay");
        
        // Delay value headers
        for (const auto& delay : delays) {
            ImGui::TableNextColumn();
            ImGui::Text("%.2f", delay);
        }
        ImGui::TableNextRow();
        
        // Create rows for each k value
        for (size_t i = 0; i < k_values.size(); i++) {
            ImGui::TableNextColumn();
            char k_label[16];
            snprintf(k_label, sizeof(k_label), "%.2f", k_values[i]);
            if (ImGui::Button(k_label, ImVec2(cell_width * 1.4, cell_height))) {
                override_k = k_values[i];
                override_delay = 0.0f;
            }
            
            // Create cells with buttons for each delay value
            for (size_t j = 0; j < delays.size(); j++) {
                ImGui::TableNextColumn();
                std::string button_label = "##" + std::to_string(i) + std::to_string(j);
                if (ImGui::Button(button_label.c_str(), ImVec2(cell_width, cell_height))) {
                    override_k = k_values[i];
                    override_delay = delays[j];
                }
                
                // Display current k and delay values if this cell is selected
                if (Utils::close(current_k, k_values[i]) && Utils::close(current_delay, delays[j])) {
                    ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 255, 0, 255));
                    ImGui::SameLine();
                    ImGui::Text("*");
                    ImGui::PopStyleColor();
                }
            }
        }
        ImGui::EndTable();
        // Add individual buttons for K0 and K8
        ImGui::TextUnformatted("Specials:"); ImGui::SameLine();
        if (ImGui::Button("K0", ImVec2(cell_width, cell_height))) {
            override_k = 0.0f;
        }
        ImGui::SameLine();
        if (ImGui::Button("K10", ImVec2(cell_width, cell_height))) {
            override_k = 1.0f;
        }
        ImGui::SameLine();
        if (ImGui::Button("d0", ImVec2(cell_width, cell_height))) {
            override_delay = 0.0f;
        }
        ImGui::SameLine();
        if (ImGui::Button("d3", ImVec2(cell_width, cell_height))) {
            override_delay = 0.25f;
        }
        ImGui::Checkbox("Exo1", &mExoTune1);
        ImGui::SameLine();
        ImGui::Checkbox("Exo2", &mExoTune2);
        ImGui::SameLine();
        static float tune_coeff = 10000.0;
        ImGui::SetNextItemWidth(50);
        if (ImGui::InputFloat("Speed", &tune_coeff))

        if (mExoTune2) mExoTune1 = false;
        double tau_target = -1.0;
        static double tau_current = -1.0;
        if (!mRolloutStatus.pause) tau_current = mEnv->GetDevice()->GetRMSMomentR();
        if (mExoTune1 || mExoTune2) {
            override_delay = 0.2f;
            if (mExoTune1) {
                mExoTune2 = false;
                tau_target = 5.5;
            }

            if (mExoTune2) {
                mExoTune1 = false;
                tau_target = 6.5;
            }
            if (!mRolloutStatus.pause) override_k = current_k + (tau_target - tau_current) / tune_coeff;
        }
        float k = current_k;
        ImGui::SetNextItemWidth(50);
        if (ImGui::InputFloat("K", &k)) params["k"] = k;
        ImGui::SameLine();
        float delay = current_delay;
        ImGui::SetNextItemWidth(50);
        if (ImGui::InputFloat("Delay", &delay)) params["delay"] = delay;

        ImGui::Text("K=%.4f, Delay=%.2f, RMS=%.2f/%.2f", current_k, current_delay, tau_current, tau_target);

        bool weight_enabled = mEnv->GetDevice()->IsWeightEnabled();
        if(ImGui::Checkbox("Enable Weight", &weight_enabled)) mEnv->SetDeviceWeightEnabled(weight_enabled); 
        ImGui::SameLine();
        bool weight_forced = mEnv->GetDevice()->IsWeightForced();
        if(ImGui::Checkbox("Force Weight", &weight_forced)) mEnv->SetDeviceWeightForced(weight_forced); 
        ImGui::SameLine();
        bool zero_state = mEnv->GetDevice()->IsZeroState();
        if(ImGui::Checkbox("Zero State", &zero_state)) mEnv->SetDeviceZeroState(zero_state);
        bool virtual_coupling = mEnv->GetDevice()->IsVirtualCoupling();
        if(ImGui::Checkbox("Virtual Coupling", &virtual_coupling)) mEnv->GetDevice()->SetVirtualCoupling(virtual_coupling); 
        ImGui::SameLine();
        bool fake_assist = mEnv->GetDevice()->IsFakeAssist();
        if(ImGui::Checkbox("Fake Assist", &fake_assist)) mEnv->GetDevice()->SetFakeAssist(fake_assist);

        const char* device_types[] = { "Hip", "HipSlope`", "Dev", "Super" };
        int device_type = mEnv->GetDevice()->GetAngleType();
        ImGui::SetNextItemWidth(50);
        if (ImGui::Combo("Type", &device_type, device_types, IM_ARRAYSIZE(device_types))) mEnv->GetDevice()->SetAngleType(device_type);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(50);
        float weight_scaler = mEnv->GetDevice()->GetWeightScaler();
        if (ImGui::DragFloat("Weight", &weight_scaler, 0.05, 1.0)) mEnv->GetDevice()->SetWeightScaler(weight_scaler);
    }    
    
    double override_stride = -1, override_phase = -1;
    if(ImGui::CollapsingHeader("Gait", ImGuiTreeNodeFlags_DefaultOpen)){
        ImGui::TextUnformatted("Param"); ImGui::SameLine();
        if (ImGui::Button("low")) {override_stride = 0.4; override_phase = 0.4; } ImGui::SameLine();
        if (ImGui::Button("low-mid")) {override_stride = 0.75; override_phase = 0.75; } ImGui::SameLine();
        if (ImGui::Button("mid")) {override_stride = 1.0; override_phase = 1.0; } ImGui::SameLine();
        if (ImGui::Button("mid-high")) {override_stride = 1.25; override_phase = 1.25; } ImGui::SameLine();
        if (ImGui::Button("high")) {override_stride = 1.4; override_phase = 1.4; } 
        float stride = params["stride"];
        ImGui::SetNextItemWidth(50);
        if (ImGui::InputFloat("Stride", &stride)) params["stride"] = stride;
        ImGui::SameLine();
        float phase = params["phase"];
        ImGui::SetNextItemWidth(50);
        if (ImGui::InputFloat("Phase", &phase)) params["phase"] = phase;
    }

    if(ImGui::CollapsingHeader("Parameter"
    // , ImGuiTreeNodeFlags_DefaultOpen
    ))
    {
        for(const auto& [name, value] : params)
        {
            if(Utils::startsWith(name, "muscle_")) continue;
            ImGui::SetNextItemWidth(ImGui::CalcTextSize("000000").x);
            ImGui::TextUnformatted(name.c_str());
            ImGui::SameLine();
            bool controllerDrawn = false;
            ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
            try {
                if(!mParamCfg.IsNull() && mParamCfg[name] && mParamCfg[name]["sample"].IsDefined()) {
                    controllerDrawn = true;
                    const auto sample_cfg = mParamCfg[name]["sample"];
                    float min_val, max_val;
                    try{
                        if (sample_cfg["value"].IsDefined()) {
                            const auto range = sample_cfg["value"].as<vector<float>>();
                            min_val = *min_element(range.begin(), range.end());
                            max_val = *max_element(range.begin(), range.end());
                        }
                    }catch (const YAML::Exception& e) {
                        min_val = 0.0f; max_val = 1.0f;
                    }
                    ImGui::DragFloat(("##" + name).c_str(), &params[name], 0.001f, min_val, max_val, "%.3f");

                    if (mCorrelateParams && name == "phase"){
                        const float portion = (params[name] - min_val) / (max_val - min_val + 1e-6f);
                        const auto stride_range = mParamCfg["stride"]["sample"]["value"].as<vector<float>>();
                        const auto stride_min = *min_element(stride_range.begin(), stride_range.end());
                        const auto stride_max = *max_element(stride_range.begin(), stride_range.end());
                        params["stride"] = stride_min + portion * (stride_max - stride_min);
                    }else if(mCorrelateParams && name == "stride"){
                        const float portion = (params[name] - min_val) / (max_val - min_val + 1e-6f);
                        const auto phase_range = mParamCfg["phase"]["sample"]["value"].as<vector<float>>();
                        const auto phase_min = *min_element(phase_range.begin(), phase_range.end());
                        const auto phase_max = *max_element(phase_range.begin(), phase_range.end());
                        params["phase"] = phase_min + portion * (phase_max - phase_min);
                    }
                }
            }catch (const YAML::Exception& e) {
                std::cerr << "[GUI] Error processing metadata: " << e.what() << std::endl;
            }
            if (!controllerDrawn) ImGui::DragFloat(("##" + name).c_str(), &params[name], 0.001f, 0, 2, "%.3f");
        }
    }
    if(ImGui::CollapsingHeader("Muscle Parameter", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::Button("Foot")) { params["muscle_force_Footdrop"] = 0.0f; } ImGui::SameLine();
        if (ImGui::Button("Wad")) { params["muscle_force_Waddling"] = 0.0f; } ImGui::SameLine();
        if (ImGui::Button("Calc")) { params["muscle_force_Calcaneal"] = 0.0f; } ImGui::SameLine();
        if (ImGui::Button("Equi")) { params["muscle_length_Equinus"] = 0.73f; } ImGui::SameLine();
        if (ImGui::Button("Hyp")) { params["muscle_length_Hyperlordosis"] = 0.6f; }
        for (const auto &[name, value]: params) {
            if(!Utils::startsWith(name, "muscle_")) continue;
            string displayName = name.substr(strlen("muscle_"));
            float min_val = 0.0f, max_val = 1.0f; // Default range
            if (!mParamCfg.IsNull() && mParamCfg[name] && mParamCfg[name]["sample"].IsDefined()) {
                const auto sample_cfg = mParamCfg[name]["sample"];
                const auto range = sample_cfg["value"].as<std::vector<float>>();
                min_val = *std::min_element(range.begin(), range.end());
                max_val = *std::max_element(range.begin(), range.end());
            }
            ImGui::TextUnformatted(displayName.c_str());
            ImGui::SameLine();
            ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
            ImGui::DragFloat(("##" + name).c_str(), &params[name], 0.001f, min_val, max_val, "%.3f");
        }
    }

    if (override_k >= 0) params["k"] = override_k;
    if (override_delay >= 0) params["delay"] = override_delay;
    if (override_stride >= 0) params["stride"] = override_stride;
    if (override_phase >= 0) params["phase"] = override_phase;

    for (auto it = params.begin(); it != params.end();) {
        if (auto paramCopyIt = paramCopy.find(it->first); paramCopyIt != paramCopy.end() && it->second == paramCopyIt->second) it = params.erase(it);
        else ++it;
    }

    if (!params.empty()) mEnv->SetParam(params);

    if(ImGui::CollapsingHeader("Joint Angle"))
    {
        for(auto jn : mEnv->GetCharacter()->GetSkeleton()->getJoints())
        {
            if(ImGui::TreeNode(jn->getName().c_str()))
            {
                Eigen::VectorXf p = jn->getPositions().cast<float>();
                int dof = jn->getNumDofs();                
                for(int i = 0; i < dof; i++)
                {
                    if(dof == 6 && i<3) ImGui::SliderFloat(std::to_string(i).c_str(), &p[i], (float)(-M_PI), (float)(M_PI), "%.3f");
                    else {
                        float lLimit = jn->getPositionLowerLimit(i), uLimit = jn->getPositionUpperLimit(i);
                        lLimit = lLimit < -M_PI ? -M_PI : lLimit; uLimit = uLimit > M_PI ? M_PI : uLimit;
                        ImGui::SliderFloat(std::to_string(i).c_str(), &p[i], lLimit, uLimit, "%.3f");
                    }
                }
                ImGui::TreePop();
                jn->setPositions(p.cast<double>());
            }
        }

        if(mEnv->GetUseDevice())
        {
            auto dev_skel = mEnv->GetDevice()->GetSkeleton();
            for(auto jn : dev_skel->getJoints())
            {
                if(jn->getNumDofs() < 1) continue;
                if(ImGui::TreeNode(jn->getName().c_str()))
                {
                    Eigen::VectorXf p = jn->getPositions().cast<float>();
                    for(int i = 0; i < (jn->getNumDofs()>3?3:jn->getNumDofs()); i++) ImGui::SliderFloat(std::to_string(i).c_str(), &p[i], (float)(jn->getPositionLowerLimit(i) < -M_PI?-M_PI:jn->getPositionLowerLimit(i)), (float)(jn->getPositionUpperLimit(i)>M_PI?M_PI:jn->getPositionUpperLimit(i)), "%.3f");
                    ImGui::TreePop();
                    jn->setPositions(p.cast<double>());
                }
            }
        }
    }

    if(ImGui::CollapsingHeader("Muscle Render"))
    {
        ImGui::SliderFloat("Force", &mDrawMuscleRange, 0.0f, 1600.0f, "%.1f");
        ImGui::Checkbox("Upper", &mDrawMuscleRangePart);
        ImGui::Checkbox("Mode", &mDrawMuscleMode);
        ImGui::Checkbox("Anchor", &mDrawMuscleAnchorMode);

        bool prev = mDrawMuscleTotal;
        ImGui::Checkbox("Total", &mDrawMuscleTotal);
        if(mDrawMuscleTotal!=prev){
            if(mDrawMuscleTotal){
                for(auto m : mEnv->GetCharacter()->GetMuscles())
                {   
                    std::string name = m->GetName();
                    mMuscleRenderMap[name] = true;
                }        
            }
            else{
                for(auto m : mEnv->GetCharacter()->GetMuscles())
                {   
                    std::string name = m->GetName();
                    mMuscleRenderMap[name] = false;
                }
            }
        }
        else{
            for(auto m : mEnv->GetCharacter()->GetMuscles())
            {
                std::string name = m->GetName();
                ImGui::Checkbox(name.c_str(), &mMuscleRenderMap[name]);    
            }
        }              
    }

    if(ImGui::CollapsingHeader("Muscle Activation"))
    {
        bool prev = mActivateMuscleTotal;
        ImGui::Checkbox("Total", &mActivateMuscleTotal);
        if(mActivateMuscleTotal!=prev){
            if(mActivateMuscleTotal){
                for(auto m : mEnv->GetCharacter()->GetMuscles())
                {   
                    std::string name = m->GetName();
                    mMuscleActivationMap[name] = true;
                }        
            }
            else{
                for(auto m : mEnv->GetCharacter()->GetMuscles())
                {   
                    std::string name = m->GetName();
                    mMuscleActivationMap[name] = false;
                }
            }
        }
        else{
            for(auto m : mEnv->GetCharacter()->GetMuscles())
            {
                std::string name = m->GetName();
                ImGui::Checkbox(name.c_str(), &mMuscleActivationMap[name]);    
            }
        }              
    }

    if(ImGui::CollapsingHeader("Rendering Option"))
    {
        ImGui::Checkbox("Log Gait Cycle Metrics", &mDisplayMetric);
        ImGui::DragInt("FrameRate", &mFramerate, 1, 1, 60);

        ImGui::TextUnformatted("Draw Settings");
        ImGui::Checkbox("Foot Step", &mDrawFootStep);
        ImGui::Checkbox("Ground", &mDrawGround);
        ImGui::Checkbox("Phase State", &mDrawPhaseState);
        ImGui::Checkbox("Collision", &mDrawCollision);
        ImGui::Checkbox("Reference", &mDrawReference); 
        ImGui::Checkbox("Character ", &mDrawCharacter);
        ImGui::Checkbox("Muscle ", &mDrawMuscle);
        ImGui::Checkbox("Device ", &mDrawDevice);        
        ImGui::Checkbox("Device Torque ", &mDrawDeviceTorque);
        ImGui::Checkbox("Device Torque from Thigh ", &mDrawDeviceTorqueFromThigh);
        ImGui::Checkbox("Muscle Torque ", &mDrawMuscleTorque);
        ImGui::Checkbox("Muscle Passive ", &mDrawMusclePassive);
        ImGui::Checkbox("Hip Torque ", &mDrawHipTorque);

        ImGui::Spacing();

        ImGui::TextUnformatted("Plot Settings");
        ImGui::Checkbox("Show Smooth", &mPlotShowSmooth);
        ImGui::DragFloat("Avg Font Scale", &mPlotAvgFontScale, 0.1f, 1.0f, 1.5f, "%.2f");
        ImGui::DragFloat("Avg Text Offset", &mPlotAvgTxtOffset, 1.0f, -100.0f, 100.0f, "%.1f");
        ImGui::DragInt("Avg Decimal", &mPlotAvgDecimal, 1, 0, 3);
        ImGui::InputDouble("Avg Compute Interval", &mPlotAvgTxtInterval); ImGui::SameLine();
    }

    if (ImGui::CollapsingHeader("Capture", ImGuiTreeNodeFlags_DefaultOpen))
    {
        if (ImGui::Button("Capture##capture"))
        {
            // clamp and capture
            // Inputs are in center-origin (0,0 at window center, +Y up)
            int x0 = (int)(mWidth * 0.5) + mCaptureX0;
            int y0 = mCaptureY0; // convert to top-left origin
            int x1 = (int)(mWidth * 0.5) + mCaptureX1;
            int y1 = mCaptureY1;
            x0 = std::min(std::max(0, x0), (int)mWidth);
            y0 = std::min(std::max(0, y0), (int)mHeight);
            x1 = std::min(std::max(0, x1), (int)mWidth);
            y1 = std::min(std::max(0, y1), (int)mHeight);
            if (_captureRegionPNG(mCaptureFilename, x0, y0, x1, y1))
                cout << "[Capture] saved: " << mCaptureFilename << std::endl;
            else
                cerr << "[Capture] failed: " << mCaptureFilename << std::endl;
        }
        ImGui::SameLine();
        ImGui::Checkbox("Region", &mCaptureShowRect);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(200);
        ImGui::InputText("File##capture", mCaptureFilename, IM_ARRAYSIZE(mCaptureFilename));

        const int sizeStep = 5; 
        ImGui::SetNextItemWidth(100);
        ImGui::InputInt("x0", &mCaptureX0, sizeStep, sizeStep); ImGui::SameLine();
        ImGui::SetNextItemWidth(100);
        ImGui::InputInt("y0", &mCaptureY0, sizeStep, sizeStep);
        ImGui::SetNextItemWidth(100);
        ImGui::InputInt("x1", &mCaptureX1, sizeStep, sizeStep); ImGui::SameLine();
        ImGui::SetNextItemWidth(100);
        ImGui::InputInt("y1", &mCaptureY1, sizeStep, sizeStep);

        ImGui::TextUnformatted(("Size: " + std::to_string(mCaptureX1 - mCaptureX0) + " x " + std::to_string(mCaptureY1 - mCaptureY0)).c_str());
    }

    if (ImGui::CollapsingHeader("Video Recording", ImGuiTreeNodeFlags_DefaultOpen))
    {
        // Record/Stop buttons
        if (mVideoRecording) {
            if (ImGui::Button("Stop Recording##video")) {
                _stopVideoRecording();
            }
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(1.0f, 0.2f, 0.2f, 1.0f), "REC");
            
            // Show recording info
            ImGui::Text("Time: %.1fs / %.0fs", mVideoElapsedTime, mVideoMaxTime);
            ImGui::Text("Frames: %d (skip: %d)", mVideoFrameCounter, mVideoFrameSkip);
            
            // Progress bar
            if (mVideoMaxTime > 0) {
                float progress = mVideoElapsedTime / mVideoMaxTime;
                ImGui::ProgressBar(progress, ImVec2(-1.0f, 0.0f), "");
            }
        } else {
            if (ImGui::Button("Start Recording##video")) {
                std::string timestamp = std::to_string(std::time(nullptr));
                std::string filename = "video_" + timestamp + ".mp4";
                strncpy(mVideoFilename, filename.c_str(), sizeof(mVideoFilename));
                _startVideoRecording(mVideoFilename, mVideoFPS);
            }
        }
        
        ImGui::Separator();
        
        // Video settings
        ImGui::SetNextItemWidth(200);
        ImGui::InputText("Filename##video", mVideoFilename, IM_ARRAYSIZE(mVideoFilename));
        
        ImGui::SetNextItemWidth(100);
        ImGui::InputInt("FPS##video", &mVideoFPS, 1, 100);
        mVideoFPS = std::clamp(mVideoFPS, 1, 100);
        
        ImGui::SetNextItemWidth(100);
        ImGui::InputDouble("Max Time (s)##video", &mVideoMaxTime, 5.0, 30.0, "%.0f");
        mVideoMaxTime = std::max(0.0, mVideoMaxTime);
        
        ImGui::Checkbox("Stop rollout when done##video", &mVideoStopRollout);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Automatically pause simulation when video recording reaches max time");
        }
        
        // Recording info when not recording
        if (!mVideoRecording) {
            double simHz = mEnv->GetControlHz();
            int estimatedSkip = std::max(1, (int)round(simHz / mVideoFPS));
            ImGui::Text("Sim Hz: %.0f, Skip: %d frames", simHz, estimatedSkip);
            
            // Show video dimensions based on capture region
            int x0 = (int)(mWidth * 0.5) + mCaptureX0;
            int y0 = mCaptureY0;
            int x1 = (int)(mWidth * 0.5) + mCaptureX1;
            int y1 = mCaptureY1;
            x0 = std::min(std::max(0, x0), (int)mWidth);
            y0 = std::min(std::max(0, y0), (int)mHeight);
            x1 = std::min(std::max(0, x1), (int)mWidth);
            y1 = std::min(std::max(0, y1), (int)mHeight);
            int videoWidth = std::max(0, x1 - x0);
            int videoHeight = std::max(0, y1 - y0);
            videoWidth = (videoWidth / 2) * 2;  // Ensure even
            videoHeight = (videoHeight / 2) * 2;
            ImGui::Text("Video size: %dx%d (from capture region)", videoWidth, videoHeight);
            
            if (mRolloutStatus.pause) {
                ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Simulation paused - no recording");
            }
        }
    }

    if(ImGui::CollapsingHeader("Camera##Control")){
        // Translation controls
        ImGui::TextUnformatted("Translation");
        ImGui::SetNextItemWidth(80);
        float trans_x = (float)mTrans.x();
        if(ImGui::DragFloat("X", &trans_x, 0.01f, -10.0f, 10.0f)) mTrans.x() = (double)trans_x;
        ImGui::SameLine();
        ImGui::SetNextItemWidth(80);
        float trans_y = (float)mTrans.y();
        if(ImGui::DragFloat("Y", &trans_y, 0.01f, -10.0f, 10.0f)) mTrans.y() = (double)trans_y;
        ImGui::SameLine();
        ImGui::SetNextItemWidth(80);
        float trans_z = (float)mTrans.z();
        if(ImGui::DragFloat("Z", &trans_z, 0.01f, -10.0f, 10.0f)) mTrans.z() = (double)trans_z;

        // Zoom control
        ImGui::SetNextItemWidth(100);
        if(ImGui::DragFloat("Zoom", &mZoom, 0.001f, 0.01f, 2.0f)) {
            // Zoom value is directly modified
        }

        // Trackball quaternion controls
        ImGui::TextUnformatted("Rotation (Quaternion)");
        Eigen::Quaterniond current_quat = mTrackball.getCurrQuat();
        float quat_x = (float)current_quat.x();
        float quat_y = (float)current_quat.y();
        float quat_z = (float)current_quat.z();
        float quat_w = (float)current_quat.w();
        
        bool quat_changed = false;
        ImGui::SetNextItemWidth(80);
        if(ImGui::DragFloat("Qx", &quat_x, 0.01f, -1.0f, 1.0f)) quat_changed = true;
        ImGui::SameLine();
        ImGui::SetNextItemWidth(80);
        if(ImGui::DragFloat("Qy", &quat_y, 0.01f, -1.0f, 1.0f)) quat_changed = true;
        ImGui::SameLine();
        ImGui::SetNextItemWidth(80);
        if(ImGui::DragFloat("Qz", &quat_z, 0.01f, -1.0f, 1.0f)) quat_changed = true;
        ImGui::SameLine();
        ImGui::SetNextItemWidth(80);
        if(ImGui::DragFloat("Qw", &quat_w, 0.01f, -1.0f, 1.0f)) quat_changed = true;
        
        if(quat_changed) {
            Eigen::Quaterniond new_quat((double)quat_w, (double)quat_x, (double)quat_y, (double)quat_z);
            new_quat.normalize(); // Ensure quaternion is normalized
            mTrackball.setQuaternion(new_quat);
        }

        // Reset button
        if(ImGui::Button("Reset Camera")) {
            mTrans = Eigen::Vector3d(0.0, 0.0, 0.0);
            mZoom = 0.25f;
            mTrackball.setQuaternion(Eigen::Quaterniond::Identity());
            // mTrackball.setQuaternion(Eigen::Quaterniond(0.917, 0, 0.4, 0));
        }
    }
    
    ImGui::End();
}

void GLFWApp::_drawSimData()
{
    double displayH = mHeight - 20;
    double displayPosX = mWidth - mPlotWidth - 10;
    double displayPosY = 10;

    ImGui::SetNextWindowSize(ImVec2(mPlotWidth, displayH), ImGuiCond_Once);
    ImGui::SetNextWindowPos(ImVec2(displayPosX, displayPosY), ImGuiCond_Once);
    ImGui::Begin("Data");

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImVec2 pos = ImGui::GetCursorScreenPos();
    float radius = 5.0f;

    ImU32 color;
    if (mRolloutStatus.pause) color = IM_COL32(255, 0, 0, 255);  // Red
    else if (mEnv->IsEndOfEpisode()==1) color = IM_COL32(255, 255, 0, 255); // Yellow, terminated           
    else color = IM_COL32(0, 255, 0, 255); // Green

    draw_list->AddCircleFilled(ImVec2(pos.x + radius, pos.y + radius), radius, color);
    // Add spacing for the circle to avoid overlap
    ImGui::Dummy(ImVec2(radius * 2, radius * 2)); ImGui::SameLine();
    ImGui::TextUnformatted("Status"); 

    if(ImGui::CollapsingHeader("Overview"
//                               ,ImGuiTreeNodeFlags_DefaultOpen
                               )) {
        ImGui::Indent();
        _drawSimulationInformation();
        ImGui::Unindent();
    }

    if(ImGui::CollapsingHeader("Character")) {
        ImGui::Indent();
        _drawCharacterInformation();
        ImGui::Unindent();
    }
        
    if(ImGui::CollapsingHeader("Metadata")){
        ImGui::Indent();
        _drawMetadata();
        ImGui::Unindent();
    }    
    
    if(ImGui::CollapsingHeader("Rollout")){
        ImGui::Indent();
        _drawRolloutParameters();
        ImGui::Unindent();
    }
        
    if(ImGui::CollapsingHeader("Reward"))
    {
        ImGui::Indent();
        _drawReward();
        ImGui::Unindent();
    }

    if(ImGui::CollapsingHeader("State"
                                //   , ImGuiTreeNodeFlags_DefaultOpen
                               ))
    {   
        ImGui::Indent();
        _drawState();
        ImGui::Unindent();
    }

    if(ImGui::CollapsingHeader("Action"
//                                   , ImGuiTreeNodeFlags_DefaultOpen
                               ))
    {   
        ImGui::Indent();
        _drawAction();
        ImGui::Unindent();
    }

    if(ImGui::CollapsingHeader("Metabolic"
                                   , ImGuiTreeNodeFlags_DefaultOpen
                               ))
    {
        ImGui::Indent();
        DrawUIDisplay_Metabolic();
        ImGui::Unindent();
    }

    if(ImGui::CollapsingHeader("Device"
                            //    , ImGuiTreeNodeFlags_DefaultOpen
                               ))
    {
        ImGui::Indent();
        _drawDevice();
        ImGui::Unindent();
    }

    if(ImGui::CollapsingHeader("Per Cycle"
                            //   , ImGuiTreeNodeFlags_DefaultOpen
                               ))
    {
        ImGui::Indent();
        _drawPerCycle();
        ImGui::Unindent();
    }

    if(ImGui::CollapsingHeader("Gait Analysis"
                            //   , ImGuiTreeNodeFlags_DefaultOpen
                               ))
    {
        ImGui::Indent();
        _drawGaitAnalysis();
        ImGui::Unindent();
    }   
    
    // if(ImGui::CollapsingHeader("Max"
                            //   , ImGuiTreeNodeFlags_DefaultOpen
                            //    ))
    // {
        // ImGui::Indent();
        // _drawMax();
        // ImGui::Unindent();
    // }

    ImGui::End();
}

// void GLFWApp::_plotMA2Part(){
//     if (ImGui::CollapsingHeader("MA2 (Part)")) {
//         if (ImPlot::BeginPlot("MA2 (Part)")) {
//             ImPlot::SetupLegend(ImPlotLocation_NorthWest, ImPlotLegendFlags_Horizontal);
//             ImPlot::SetupAxis(ImAxis_X1, "Data Index");
//             ImPlot::SetupAxis(ImAxis_Y1, "");
//             ImPlot::SetupAxis(ImAxis_Y2, "", ImPlotAxisFlags_Opposite);
//             vector<string> meta_key1 = {"ma2_glt_max", "ma2_glt_med", "ma2_quadriceps", "ma2_hamstrings", "ma2_tib_a", "ma2_tib_p"};
//             vector<string> meta_key2 = {"ma2_sol_gas"};
//             std::ignore = _setupGraphAxes(mGraphCycleData, meta_key1, -1, ImAxis_Y1, 0, 20, false);
//             std::ignore = _setupGraphAxes(mGraphCycleData, meta_key2, -1, ImAxis_Y2, 40, 80, false);
//             _pl1otHeadlessGraphData(mGraphCycleData, meta_key1, ImAxis_Y1, false);
//             _pl1otHeadlessGraphData(mGraphCycleData, meta_key2, ImAxis_Y2, false);
//             ImPlot::EndPlot();
//         }
//     }
// }

void GLFWApp::_storeGraphDataKeys(const char* search_key, bool verbose){
    bool found = false;
    for (const auto& key : mGraphData.keys()) {
        if (key.find(search_key) != std::string::npos) {
            mStoredGraphData.deepcopy(key, mGraphData);
            if (verbose) cout << "[Store] key: " << key << endl;
            found = true;
        }
    }
    if (!found) cout << "[Store] No keys found matching: " << search_key << endl;
}

void GLFWApp::_showGraphDataKeys(const char* search_key, bool verbose){
    bool found = false;
    for (const auto& key : mGraphData.keys()) {
        if (key.find(search_key) != std::string::npos) {
            mShowGraphData.push_back(key);
            if (verbose) cout << "[Show] key: " << key << endl;
            found = true;
        }
    }
    if (!found) cout << "[Show] No keys found matching: " << search_key << endl;
}

void GLFWApp::_printGraphDataKeys(const char* search_key, bool verbose){
    bool found = false;
    for (const auto& key : mGraphData.keys()) {
        if (key.find(search_key) != std::string::npos) {
            const auto& buffer = mGraphData[key];
            std::string csv_data;
            int count = 0;
            // Generate (x, y) data where x is time index and y is value
            float ts = mEnv->GetWorld()->getTimeStep();
            for (size_t i = 0; i < buffer.size(); i++) {
                float xval = -(float)(buffer.size() - 1 - i) * ts;
                // if (xval < mXmin || xval > mXmax) cout << "xval: " << xval << " is out of range " << mXmin << " to " << mXmax << endl;
                if (xval < mXmin || xval > mXmax) continue;
                csv_data += std::to_string(xval) + "," + std::to_string(buffer[i]) + "\n";
                count++;
            }
            
            // Copy to clipboard using GLFW
            glfwSetClipboardString(mWindow, csv_data.c_str());
            
            cout << "[Print] key: " << key << " (" << count << " points copied to clipboard)" << endl;
            if (verbose) cout << csv_data << endl;
            found = true;
            break; // Use only the first matching key as requested
        }
    }
    if (!found) cout << "[Print] No keys found matching: " << search_key << endl;
}

void GLFWApp::_printContactPhase(bool verbose){
    if (mGraphData.key_exists("contact_phaseR")) {
        const boost::circular_buffer<double> &contact_phase_buffer = mGraphData["contact_phaseR"];
        std::string csv_data;
        
        // Ensure there are at least two points to compare for transitions
        if (contact_phase_buffer.size() >= 2) {
            bool prev_phase = static_cast<bool>(contact_phase_buffer[0]);
            int transition_count = 0;
            
            for (int i = 1; i < contact_phase_buffer.size(); ++i) {
                bool current_phase = static_cast<bool>(contact_phase_buffer[i]);
                
                // Check for phase transition
                if (prev_phase != current_phase) {
                    // Calculate time based on buffer index and time step
                    const double transition_time = (-(static_cast<int>(contact_phase_buffer.size())) + i) * mEnv->GetWorld()->getTimeStep();
                    
                    // Add transition time to CSV data
                    csv_data += std::to_string(transition_time) + ",\n";
                    transition_count++;
                }
                prev_phase = current_phase;
            }
            
            // Copy transition times to clipboard
            glfwSetClipboardString(mWindow, csv_data.c_str());
            cout << csv_data << endl;
            
            cout << "[PrintC] contact_phaseR: " << transition_count << " phase transitions copied to clipboard" << endl;
        } else {
            cout << "[PrintC] contact_phaseR: Not enough data points for transition detection" << endl;
        }
    } else {
        cout << "[PrintC] contact_phaseR key not found in graph data" << endl;
    }
}

void GLFWApp::_setXminToHeelStrike(){
    if (mGraphData.key_exists("contact_phaseR")) {
        const boost::circular_buffer<double> &contact_phase_buffer = mGraphData["contact_phaseR"];
        
        // Ensure there are at least two points to compare for transitions
        if (contact_phase_buffer.size() >= 2) {
            bool prev_phase = static_cast<bool>(contact_phase_buffer[0]);
            double heel_strike_time = 0.0;
            bool found_heel_strike = false;
            
            // Search for the most recent heel strike (swing to stance transition)
            for (int i = 1; i < contact_phase_buffer.size(); ++i) {
                bool current_phase = static_cast<bool>(contact_phase_buffer[i]);
                
                // Check for heel strike: transition from swing (false) to stance (true)
                if (!prev_phase && current_phase) {
                    // Calculate time based on buffer index and time step
                    float heel_strike_time_candidate = (-(static_cast<int>(contact_phase_buffer.size())) + i) * mEnv->GetWorld()->getTimeStep();
                    if (heel_strike_time_candidate < -0.3) {
                        heel_strike_time = heel_strike_time_candidate;
                        found_heel_strike = true;
                    } // Don't break - we want the most recent (last) heel strike
                }
                prev_phase = current_phase;
            }
            
            if (found_heel_strike) {
                mXmin = heel_strike_time;
                mXmax = 0.0;  // Keep current time as max
            } else {
                cout << "[HeelStrike] No heel strike found in current data" << endl;
            }
        } else {
            cout << "[HeelStrike] Not enough data points for heel strike detection" << endl;
        }
    } else {
        cout << "[HeelStrike] contact_phaseR key not found in graph data" << endl;
    }
}

bool GLFWApp::_captureRegionPNG(const char* filename, int x0, int y0, int x1, int y1){
    // Interpret inputs as top-left (x0,y0) and bottom-right (x1,y1) in window coordinates (origin: top-left)
    int x_min = std::min(x0, x1);
    int x_max = std::max(x0, x1);
    int y_top = std::min(y0, y1);
    int y_bottom = std::max(y0, y1);

    // Clamp to window bounds
    x_min = std::max(0, std::min(x_min, (int)mWidth));
    x_max = std::max(0, std::min(x_max, (int)mWidth));
    y_top = std::max(0, std::min(y_top, (int)mHeight));
    y_bottom = std::max(0, std::min(y_bottom, (int)mHeight));

    int w = std::max(0, x_max - x_min);
    int h = std::max(0, y_bottom - y_top);
    if (w <= 0 || h <= 0) return false;

    // Convert to OpenGL lower-left origin
    int gl_x = x_min;
    int gl_y = (int)mHeight - (y_top + h);
    gl_y = std::max(0, gl_y);

    std::vector<unsigned char> pixels(w * h * 4);
    std::vector<unsigned char> flipped(w * h * 4);

    glFinish();
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(gl_x, gl_y, w, h, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());

    // Flip vertically for PNG top-left origin
    for (int row = 0; row < h; ++row) {
        const unsigned char* src = pixels.data() + (h - 1 - row) * w * 4;
        unsigned char* dst = flipped.data() + row * w * 4;
        std::memcpy(dst, src, w * 4);
    }

    // check if filename has postfix (.png)
    std::string filename_with_postfix = std::string(filename);
    if (std::string(filename).find(".png") == std::string::npos) {
        filename_with_postfix += ".png";
    }

    unsigned int result = lodepng::encode(filename_with_postfix, flipped, w, h);
    if (result != 0) {
        std::cerr << "[Capture] lodepng error " << result << std::endl;
        return false;
    }
    return true;
}

bool GLFWApp::_startVideoRecording(const std::string& filename, int fps) {
    if (mFFmpegPipe) {
        _stopVideoRecording();
    }
    
    // Create video directory if it doesn't exist
    std::filesystem::create_directories("video");
    
    // Calculate capture region dimensions (same as _captureRegionPNG)
    int x0 = (int)(mWidth * 0.5) + mCaptureX0;
    int y0 = mCaptureY0;
    int x1 = (int)(mWidth * 0.5) + mCaptureX1;
    int y1 = mCaptureY1;
    
    // Clamp to window bounds
    x0 = std::min(std::max(0, x0), (int)mWidth);
    y0 = std::min(std::max(0, y0), (int)mHeight);
    x1 = std::min(std::max(0, x1), (int)mWidth);
    y1 = std::min(std::max(0, y1), (int)mHeight);
    
    int width = std::max(0, x1 - x0);
    int height = std::max(0, y1 - y0);
    
    // Ensure even dimensions for x264
    width = (width / 2) * 2;
    height = (height / 2) * 2;
    
    // Validate capture region
    if (width <= 0 || height <= 0) {
        std::cerr << "[Video] Invalid capture region: " << width << "x" << height 
                  << ". Please adjust capture region in GUI." << std::endl;
        return false;
    }
    
    // Build ffmpeg command for direct pipe input
    std::string full_path = "video/" + filename;
    std::string cmd = "ffmpeg -y -f rawvideo -vcodec rawvideo -pix_fmt rgb24";
    cmd += " -s " + std::to_string(width) + "x" + std::to_string(height);
    cmd += " -r " + std::to_string(fps);
    cmd += " -i - -c:v libx264 -pix_fmt yuv420p";
    cmd += " -crf 23 -preset medium";
    cmd += " -movflags +faststart"; // Enable fast start for web playback
    cmd += " \"" + full_path + "\" 2>/dev/null"; // Suppress ffmpeg logs
    
    mFFmpegPipe = popen(cmd.c_str(), "w");
    if (!mFFmpegPipe) {
        std::cerr << "[Video] Failed to open ffmpeg pipe for: " << full_path << std::endl;
        return false;
    }
    
    // Initialize recording state
    mVideoRecording = true;
    mVideoElapsedTime = 0.0;
    mVideoFrameCounter = 0;
    mLastVideoTime = 0.0; // Reset simulation time tracking
    
    // Cache the control step time (computed once)
    double simTimeStep = mEnv->GetWorld()->getTimeStep();
    int numSimSteps = mEnv->GetNumSteps(); // simulation steps per control step
    mCachedControlStepTime = simTimeStep * numSimSteps;
    
    // Calculate frame skip based on simulation vs video framerate
    double simHz = mEnv->GetControlHz();
    mVideoFrameSkip = std::max(1, (int)round(simHz / fps));
    
    std::cout << "[Video] Started recording: " << full_path 
              << " (fps=" << fps << ", skip=" << mVideoFrameSkip 
              << ", step_time=" << mCachedControlStepTime << "s)" << std::endl;
    return true;
}

void GLFWApp::_stopVideoRecording() {
    if (mFFmpegPipe) {
        pclose(mFFmpegPipe);
        mFFmpegPipe = nullptr;
        mVideoRecording = false;
        
        std::cout << "[Video] Recording stopped. Duration: " 
                  << std::fixed << std::setprecision(1) << mVideoElapsedTime 
                  << "s, Frames: " << mVideoFrameCounter << std::endl;
    }
}

void GLFWApp::_recordVideoFrame() {
    if (!mVideoRecording || !mFFmpegPipe) return;
    
    // Don't record if simulation is paused
    if (mRolloutStatus.pause) return;
    
    // Check if we should record this frame based on frame skip
    if (!_shouldRecordFrame()) return;
    
    // Check maximum recording time
    if (mVideoMaxTime > 0 && mVideoElapsedTime >= mVideoMaxTime) {
        if (mVideoStopRollout) {
            std::cout << "[Video] Maximum recording time reached (" 
                      << mVideoMaxTime << "s). Stopping recording and rollout." << std::endl;
            _stopVideoRecording();
            
            // Stop the rollout/simulation
            mRolloutStatus.pause = true;
            mRolloutStatus.cycle = 0; // End rollout
        } else {
            std::cout << "[Video] Maximum recording time reached (" 
                      << mVideoMaxTime << "s). Stopping recording only." << std::endl;
            _stopVideoRecording();
        }
        return;
    }
    
    // Calculate capture region (same as _captureRegionPNG logic)
    int x0 = (int)(mWidth * 0.5) + mCaptureX0;
    int y0 = mCaptureY0;
    int x1 = (int)(mWidth * 0.5) + mCaptureX1;
    int y1 = mCaptureY1;
    
    // Clamp to window bounds
    x0 = std::min(std::max(0, x0), (int)mWidth);
    y0 = std::min(std::max(0, y0), (int)mHeight);
    x1 = std::min(std::max(0, x1), (int)mWidth);
    y1 = std::min(std::max(0, y1), (int)mHeight);
    
    int width = std::max(0, x1 - x0);
    int height = std::max(0, y1 - y0);
    
    // Ensure even dimensions
    width = (width / 2) * 2;
    height = (height / 2) * 2;
    
    if (width <= 0 || height <= 0) return;
    
    // Convert to OpenGL lower-left origin
    int gl_x = x0;
    int gl_y = (int)mHeight - (y0 + height);
    gl_y = std::max(0, gl_y);
    
    // Capture region framebuffer as RGB24
    std::vector<unsigned char> pixels(width * height * 3);
    std::vector<unsigned char> flipped(width * height * 3);
    
    glFinish();
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(gl_x, gl_y, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
    
    // Flip vertically (OpenGL is bottom-up, video is top-down)
    int stride = width * 3;
    for (int y = 0; y < height; ++y) {
        const unsigned char* src = pixels.data() + (height - 1 - y) * stride;
        unsigned char* dst = flipped.data() + y * stride;
        std::memcpy(dst, src, stride);
    }
    
    // Write frame to ffmpeg pipe
    size_t written = fwrite(flipped.data(), 1, flipped.size(), mFFmpegPipe);
    if (written != flipped.size()) {
        std::cerr << "[Video] Failed to write frame to ffmpeg pipe" << std::endl;
        _stopVideoRecording();
        return;
    }
    fflush(mFFmpegPipe);
    
    mVideoFrameCounter++;
    
    // Update elapsed time using cached control step time
    mVideoElapsedTime += mCachedControlStepTime;
    mLastVideoTime += mCachedControlStepTime;
}

bool GLFWApp::_shouldRecordFrame() {
    // Always record if frame skip is 1
    if (mVideoFrameSkip <= 1) return true;
    
    // Skip frames based on frame skip ratio
    static int frameSkipCounter = 0;
    frameSkipCounter++;
    
    if (frameSkipCounter >= mVideoFrameSkip) {
        frameSkipCounter = 0;
        return true;
    }
    
    return false;
}

void GLFWApp::_drawPerCycle()
{
    mOpendGraphTab = true;
    if (mEnv->GetUseMuscle()) {
        // _plotGraphData(mGraphCycleData, {"bhar", "mine", "houd", "ma2", "ma2_leg", "ma2_torso"}, "Meta Zoo", 200, 550, nullptr, false);
        _plotGraphData(mGraphCycleData, {
            // "ma2_leg", 
            // "ma2_weighted", 
            "ma15"}, "MA15", 20, 80, nullptr, false);
    mOpendGraphTab = false;
        _plotGraphData(mGraphCycleData, {"ma2_he", "ma2_hf", "ma2_ke", "ma2_kf", "ma2_ae", "ma2_af"}, "MA2 (Part)", 0, 10, nullptr, false);
        // _plotGraphData(mGraphCycleData, {"a2"}, "A2", 100, 300, nullptr, false);      
    }
    mOpendGraphTab = false;
    _plotGraphData(mGraphCycleData, {"power_tau", "power_tau_hipR", "power_tau_kneeR", "power_tau_ankleR", "power_muscle", "power_muscle_hipR", "power_muscle_kneeR", "power_muscle_ankleR"}, "PoT", 200, 550, nullptr, false);
    // mOpendGraphTab = true;
    _plotGraphData(mGraphCycleData, {"power_neg", "power_pos", "dev_assist_raw"}, "Assist (Power)", -5, 15, nullptr, false);
    mOpendGraphTab = false;
    _plotGraphData(mGraphCycleData, {"rew_total_step", "rew_meta_step", "rew_loco_step"
        }, "Reward (Step)", 2, 5, nullptr, false, false);
    // _plotGraphData(mGraphCycleData, {
        // "moment_muscle_hipR", 
        // "moment_muscle_hipRx_ef", "moment_muscle_hipRy_ie", "moment_muscle_hipRz_aa", 
        // "moment_muscle_kneeR", "moment_muscle_ankleR"}, "Moment", -1, 1, nullptr, false);
    // _plotGraphData(mGraphCycleData, {"footstep"}, "Footstep", 0, 1.0, nullptr, false);
    // _plotGraphData(mGraphCycleData, {"dev_assist_max", "dev_assist_range", "dev_assist_min"}, "Dev Assist", 0, 1.0, nullptr, false);
    _plotGraphData(mGraphCycleData, {"angle_HipR_min", "angle_HipR_max", "angle_HipR_range"}, "Angle (Hip)", -20, 70, nullptr, false);
}

void GLFWApp::DrawUIDisplay_Metabolic()
{
    mOpendGraphTab = false;
    _plotGraphData(mGraphData, {"bhar_activation", "bhar_maintenance", "bhar_shortening", "bhar_mechanical_work"}, "Metabolic Cost (Bhar)", 0.0, 1.0);
    _plotGraphData(mGraphData, {"umberger_activation", "umberger_maintenance", "umberger_shortening", "umberger_mechanical_work"}, "Metabolic Cost (Umberger)", 0.0, 1.0);
    mOpendGraphTab = false;
    if (ImGui::CollapsingHeader("Meta (Bhar)")) {
        float totla_bhar = mGraphCycleData["bhar_activation"].back() + mGraphCycleData["bhar_maintenance"].back() + mGraphCycleData["bhar_shortening"].back() + mGraphCycleData["bhar_mechanical_work"].back();
        float bhar_activation_ratio = mGraphCycleData["bhar_activation"].back() / totla_bhar * 100.0;
        float bhar_maintenance_ratio = mGraphCycleData["bhar_maintenance"].back() / totla_bhar * 100.0;
        float bhar_shortening_ratio = mGraphCycleData["bhar_shortening"].back() / totla_bhar * 100.0;
        float bhar_mechanical_work_ratio = mGraphCycleData["bhar_mechanical_work"].back() / totla_bhar * 100.0;
        static const char* labels[] = {"Activation", "Maintenance", "Shortening", "Mechanical Work"};
        float bhar_part[] = {bhar_activation_ratio, bhar_maintenance_ratio, bhar_shortening_ratio, bhar_mechanical_work_ratio};
        if (ImPlot::BeginPlot("Meta (Bhar)", ImVec2(300, 300), mPlotPie)) {
            ImPlot::SetupLegend(ImPlotLocation_NorthWest);
            ImPlot::PlotPieChart(labels, bhar_part, 4, 0.5, 0.5, 0.5, "%.3f");
            ImPlot::EndPlot();
        }
    }
    if (ImGui::CollapsingHeader("Meta (Umberger)")) {
        float totla_umberger = mGraphCycleData["umberger_activation"].back() + mGraphCycleData["umberger_maintenance"].back() + mGraphCycleData["umberger_shortening"].back() + mGraphCycleData["umberger_mechanical_work"].back();
        float umberger_activation_ratio = mGraphCycleData["umberger_activation"].back() / totla_umberger * 100.0;
        float umberger_maintenance_ratio = mGraphCycleData["umberger_maintenance"].back() / totla_umberger * 100.0;
        float umberger_shortening_ratio = mGraphCycleData["umberger_shortening"].back() / totla_umberger * 100.0;
        float umberger_mechanical_work_ratio = mGraphCycleData["umberger_mechanical_work"].back() / totla_umberger * 100.0;
        static const char* labels[] = {"Activation", "Maintenance", "Shortening", "Mechanical Work"};
        float umberger_part[] = {umberger_activation_ratio, umberger_maintenance_ratio, umberger_shortening_ratio, umberger_mechanical_work_ratio};
        if (ImPlot::BeginPlot("Meta (Umberger)", ImVec2(300, 300), mPlotPie)) {
            ImPlot::SetupLegend(ImPlotLocation_NorthWest);
            ImPlot::PlotPieChart(labels, umberger_part, 4, 0.5, 0.5, 0.5, "%.3f");
            ImPlot::EndPlot();
        }
    }

    if (ImGui::CollapsingHeader("Meta (Pie)")) {
        static bool show_title = false;
        float windowWidth = ImGui::GetWindowWidth();
        float checkboxWidth = 45.0f, copyButtonWidth = 30.0f; // Approximate mWidth for each checkbox
        float titleCheckboxPosX = windowWidth - checkboxWidth - 10; // 10px padding

        ImVec2 cursorPos = ImGui::GetCursorPos();

        ImGui::SetCursorPos(ImVec2(titleCheckboxPosX, cursorPos.y));
        ImGui::Checkbox("Titl##MA2Pie", &show_title);

        const char* title_str;
        string filename = filesystem::path(mArgs.torchscript_dir).filename().string();
        title_str = show_title ? filename.c_str() : "MA2 (Pie)";

        if (mEnv->GetUseMuscle() && mGraphCycleData.key_exists("ma2_he")) {
            const float sum_ma2 = mGraphCycleData["ma2_he"].back() + mGraphCycleData["ma2_ke"].back() + mGraphCycleData["ma2_ae"].back() + mGraphCycleData["ma2_fl"].back();
            const float ma2_he = mGraphCycleData["ma2_he"].back() / sum_ma2;
            const float ma2_ke = mGraphCycleData["ma2_ke"].back() / sum_ma2;
            const float ma2_ae = mGraphCycleData["ma2_ae"].back() / sum_ma2;
            const float ma2_fl = mGraphCycleData["ma2_fl"].back() / sum_ma2;
            static const char* labels[] = {"HE", "KE", "AE", "FL"};
            float ma2_part[] = {ma2_he, ma2_ke, ma2_ae, ma2_fl};
            if (ImPlot::BeginPlot(title_str, ImVec2(300, 300), mPlotPie)) {
                ImPlot::SetupLegend(ImPlotLocation_NorthWest);
                ImPlot::PlotPieChart(labels, ma2_part, 4, 0.5, 0.5, 0.5, "%.3f");
                ImPlot::EndPlot();
            }
        }
        // if (mEnv->GetUseMuscle() && mGraphCycleData.key_exists("power_muscle_hipR")) {
        //     const float power_muscle = mGraphCycleData["power_muscle_hipR"].back() + mGraphCycleData["power_muscle_kneeR"].back() + mGraphCycleData["power_muscle_ankleR"].back();
        //     const float power_muscle_hipR = mGraphCycleData["power_muscle_hipR"].back() / power_muscle;
        //     const float power_muscle_kneeR = mGraphCycleData["power_muscle_kneeR"].back() / power_muscle;
        //     const float power_muscle_ankleR = mGraphCycleData["power_muscle_ankleR"].back() / power_muscle;
        //     static const char* labels[] = {"Hip", "Knee", "Ankle"};
        //     float power_muscle_part[] = {power_muscle_hipR, power_muscle_kneeR, power_muscle_ankleR};
        //     if (ImPlot::BeginPlot("Power-muscle (Pie)", ImVec2(300,300), mPlotPie)) {
        //         ImPlot::SetupLegend(ImPlotLocation_NorthWest);
        //         ImPlot::PlotPieChart(labels, power_muscle_part, 3, 0.5, 0.5, 0.5, "%.3f");
        //         ImPlot::EndPlot();
        //     }
        // }
        // if (mGraphCycleData.key_exists("power_tau_hipR")) {
        //     const float power_tau = mGraphCycleData["power_tau_hipR"].back() + mGraphCycleData["power_tau_kneeR"].back() + mGraphCycleData["power_tau_ankleR"].back();
        //     const float power_tau_hipR = mGraphCycleData["power_tau_hipR"].back() / power_tau;
        //     const float power_tau_kneeR = mGraphCycleData["power_tau_kneeR"].back() / power_tau;
        //     const float power_tau_ankleR = mGraphCycleData["power_tau_ankleR"].back() / power_tau;
        //     static const char* labels[] = {"Hip", "Knee", "Ankle"};
        //     float power_tau_part[] = {power_tau_hipR, power_tau_kneeR, power_tau_ankleR};
        //     if (ImPlot::BeginPlot("Power-tau (Pie)", ImVec2(290,290), mPlotPie)) {
        //         ImPlot::SetupLegend(ImPlotLocatio_n_NorthWest);
        //         ImPlot::PlotPieChart(labels, power_tau_part, 3, 0.5, 0.5, 0.5, "%.3f");
        //         ImPlot::EndPlot();
        //     }
        // }
    }
}

void GLFWApp::_drawSimulationInformation()
{
    ImGui::Text("Sim FPS :  %.2f ", mMeasuredFps);
    ImGui::Text("Sim Step :  %d ", mEnv->GetSimStep());
    ImGui::Text(" Wall Time :  %.1f s", mEnv->GetWorld()->getTime());

    ImGui::NewLine();

    ImGui::Columns(4, "Runtime Metrics", true);
    ImGui::Separator();

    ImGui::Text("Metric"); ImGui::NextColumn();
    ImGui::Text("Value"); ImGui::NextColumn();
    ImGui::Text("Target"); ImGui::NextColumn();
    ImGui::Text("Ratio"); ImGui::NextColumn();
    ImGui::Separator();

    const double target_vel = mEnv->GetTargetVelocity();
    const double avg_vel = mEnv->GetAvgVelocity();
    const double vel_ratio = avg_vel / target_vel * 100.0;
    ImGui::Text("Velocity"); ImGui::NextColumn();
    ImGui::Text("%.3f m/s\n%.3f km/h", avg_vel, avg_vel*3.6); ImGui::NextColumn();
    ImGui::Text("%.3f m/s\n%.3f km/h", target_vel, target_vel*3.6); ImGui::NextColumn();
    ImGui::Text("%.1f %%", vel_ratio); ImGui::NextColumn();
    ImGui::Separator();

    const double target_step = mEnv->GetTargetStride() / 2;
    const double avg_step = (mEnv->GetStrideLengthR() + mEnv->GetStrideLengthL()) / 4.0;
    const double step_ratio = avg_step / target_step * 100.0;
    ImGui::Text("Step"); ImGui::NextColumn();
    ImGui::Text("%.3f m", avg_step); ImGui::NextColumn();
    ImGui::Text("%.3f m", target_step); ImGui::NextColumn();
    ImGui::Text("%.1f %%", step_ratio); ImGui::NextColumn();
    ImGui::Separator();

    const double target_cadence = mEnv->GetTargetCadence();
    const double avg_cadence = mEnv->GetCurrentCadence();
    const double cadence_ratio = avg_cadence / target_cadence * 100.0;
    ImGui::Text("Cadence"); ImGui::NextColumn();
    ImGui::Text("%.3f hz", avg_cadence); ImGui::NextColumn();
    ImGui::Text("%.3f hz", target_cadence); ImGui::NextColumn();
    ImGui::Text("%.1f %%", cadence_ratio); ImGui::NextColumn();
    ImGui::Separator();

    ImGui::Text("Cycle#"); ImGui::NextColumn();
    ImGui::Text("Dist"); ImGui::NextColumn();
    ImGui::Text("Time"); ImGui::NextColumn();
    ImGui::Text("Phase"); ImGui::NextColumn();
    ImGui::Separator();

    ImGui::Text("%d", mEnv->GetCycleCount()); ImGui::NextColumn();
    ImGui::Text("%.3f m", mEnv->GetCycleDist()); ImGui::NextColumn();
    ImGui::Text("%.3f s", mEnv->GetCycleTime()); ImGui::NextColumn();
    ImGui::Text("%.3f", mEnv->GetPhase()); ImGui::NextColumn();
    ImGui::Separator();
    ImGui::Columns(1);
}

void GLFWApp::_drawCharacterInformation()
{
    ImGui::Text("Character Height :  %.2f m", mEnv->GetCharacter()->GetHeight());
    ImGui::Text("Character Weight :  %.2f kg", mEnv->GetCharacter()->GetWeight());

    if(mEnv->GetUseDevice())
    {
        ImGui::Text("Hip Device Weight :  %.2f kg", mEnv->GetDevice()->GetSkeleton()->getMass());
    }
}

void GLFWApp::_drawMetadata()
{
    // Save original font scale
    float originalFontScale = ImGui::GetFont()->Scale;
    
    // Set new scale and push font
    ImGui::GetFont()->Scale = 1.25f;
    ImGui::PushFont(ImGui::GetFont());
    
    // Save and modify spacing
    ImGuiStyle& style = ImGui::GetStyle();
    float originalItemSpacingY = style.ItemSpacing.y;
    style.ItemSpacing.y *= 1.75f;

    // Draw text
    ImGui::TextUnformatted(mEnv->GetMetadata().c_str());

    // Restore spacing
    style.ItemSpacing.y = originalItemSpacingY;
    
    // Restore font scale before popping font
    ImGui::GetFont()->Scale = originalFontScale;
    ImGui::PopFont();
}

void GLFWApp::_drawRolloutParameters()
{
    ImGui::Text("Rollout remains %d cycles and params are %d", mRolloutStatus.cycle, mRolloutStatus.paramSize());
    if (mRolloutStatus.cycle >= 0) {
        ImGui::Text("Params path: %s", mRolloutStatus.settingPath.c_str());
        ImGui::Text("Params: ");

        // Save original font scale
        float originalFontScale = ImGui::GetFont()->Scale;
        
        // Set new scale and push font
        ImGui::GetFont()->Scale = 1.25f;
        ImGui::PushFont(ImGui::GetFont());
        
        // Save and modify spacing
        ImGuiStyle& style = ImGui::GetStyle();
        float originalItemSpacingY = style.ItemSpacing.y;
        style.ItemSpacing.y *= 1.75f;

        // Draw text
        ImGui::TextUnformatted(mRolloutStatus.fileContents.c_str());

        // Restore spacing
        style.ItemSpacing.y = originalItemSpacingY;
        
        // Restore font scale before popping font
        ImGui::GetFont()->Scale = originalFontScale;
        ImGui::PopFont();
    }
}

void GLFWApp::_drawReward()
{
    static const char* reward_types[] = { "overall", "gait", "energy", "imit" };
    static int reward_type = 0;
    ImGui::SetNextItemWidth(130);
    ImGui::Combo("Type", &reward_type, reward_types, IM_ARRAYSIZE(reward_types));
    vector<string> reward_keys;
    if (reward_type == 0) reward_keys = {"rew_total", "rew_gait", "rew_meta", "rew_dev", "rew_imit"};
    else if (reward_type == 1) reward_keys = {"rew_gait", "rew_loco", "rew_footstep", "rew_velocity", "rew_head", "rew_sway"};
    else if (reward_type == 2) reward_keys = {"rew_meta", "rew_meta_act", "rew_meta_torque"};
    else if (reward_type == 3) reward_keys = {"rew_imit", "rew_imit_pos", "rew_imit_vel"};

    _plotGraphData(mGraphData, reward_keys, "Reward", -0.05, 1.5, nullptr, false);
}

void GLFWApp::_drawState()
{
    if(ImGui::CollapsingHeader("network state")){
        const auto state = mEnv->GetState();
        int numState = state.rows();
        ImPlot::SetNextAxesLimits(-0.5, numState + 0.5, -5.0, 5.0);
        if (ImPlot::BeginPlot("State"))
        {
            if (mBarXCoords.find("State") == mBarXCoords.end()) {
                mBarXCoords["State"] = std::vector<double>(numState);
                for(int i = 0; i < numState; i++) mBarXCoords["State"][i] = i;
            }
            std::vector<double> y(numState);
            for(int i = 0; i < numState; i++) y[i] = state[i];
            ImPlot::PlotBars("", mBarXCoords["State"].data(), y.data(), numState, 1.0);
            ImPlot::EndPlot();
        }
    }    
    if(ImGui::CollapsingHeader("raw power")){
        const float bar_width = 0.25;
        const auto powerTau = mEnv->GetPowerTau();
        int numPower = powerTau.rows();
        ImPlot::SetNextAxesLimits(-0.5, numPower + 0.5, -5.0, 5.0);
        if (ImPlot::BeginPlot("Power", ImVec2(-1, 700)))
        {
            std::vector<double> y1(numPower);
            if (mBarXCoords.find("power_tau") == mBarXCoords.end()) {
                mBarXCoords["power_tau"] = std::vector<double>(numPower);
                for(int i = 0; i < numPower; i++) mBarXCoords["power_tau"][i] = i;
            }
            for(int i = 0; i < numPower; i++) y1[i] = powerTau[i];
            // Set color for tau bars (e.g., blue)
            ImPlot::PushStyleColor(ImPlotCol_Fill, ImVec4(0.0f, 0.4f, 0.8f, 0.7f));
            ImPlot::PlotBars("tau", mBarXCoords["power_tau"].data(), y1.data(), numPower, bar_width);
            ImPlot::PopStyleColor();
            
            if (mEnv->GetUseMuscle()) {
                std::vector<double> y2(numPower);
                const auto powerMuscle = mEnv->GetPowerMuscle();
                if (mBarXCoords.find("power_muscle") == mBarXCoords.end()) {
                    mBarXCoords["power_muscle"] = std::vector<double>(numPower);
                    for(int i = 0; i < numPower; i++) mBarXCoords["power_muscle"][i] = i + bar_width;
                }
                for(int i = 0; i < numPower; i++) y2[i] = powerMuscle[i];
                // Set color for muscle bars (e.g., red)
                ImPlot::PushStyleColor(ImPlotCol_Fill, ImVec4(0.8f, 0.2f, 0.2f, 0.7f));
                ImPlot::PlotBars("muscle", mBarXCoords["power_muscle"].data(), y2.data(), numPower, bar_width);
                ImPlot::PopStyleColor();
            }
            ImPlot::EndPlot();
        }
    }

    map<string, float> horizontal_lines_head = {
        {"Head Rot", mEnv->mDiffHeadRot * mEnv->mHeadMarginRatio},
        {"Head Linacc", mEnv->mDiffHeadLinacc * mEnv->mHeadMarginRatio},
        {"Head Rotacc", mEnv->mDiffHeadRotacc * mEnv->mHeadMarginRatio},
    };

    _plotGraphData(mGraphData, {"head_rot", "head_linacc", "head_rotacc", "head_relvel"}, "Head Acc", -0.01, 0.1, &horizontal_lines_head, false);
    _plotGraphData(mGraphData, {"head_ypos"}, "Head Pos", 1.6, 1.7);
    _plotGraphData(mGraphData, {
        "actHe_glt_max", "actHe_sem_bra", "actHe_sem_ten", 
        "actHf_illc", "actHf_psoas", "actHf_rec_fem", 
        "actKe_vas_lat", "actKe_vas_med", 
        "actAe_gas_med", "actAe_gas_lat", "actAe_sol", "actAe_tibp",
        "actAf_tiba",
         }, "Muscle", -0.01, 1.0);
    _plotGraphData(mGraphData, {"energy_avg", "energy_muscle"}, "Energy", 0.5, 2.0);
    map<string, float> horizontal_lines = {
        {"FootDiff", mEnv->mDiffFootX},
        // {"ArmDiff", mEnv->mDiffArmX},
        // {"TorsoXDiff", mEnv->mDiffTorsoX}, 
        // {"TorsoXYDiff", mEnv->mDiffTorsoXY}
    };
    _plotGraphData(mGraphData, {"sway_Foot_R", "sway_Foot_RXZ"}, "Sway", -0.01, 0.1, &horizontal_lines, false);
    map<string, float> horizontal_lines_phase = {
        {"Action Scale", mEnv->GetActionPhaseScale()},
        {"Action Scale (m)", -mEnv->GetActionPhaseScale()},
        {"Hard Limit", -1.0 / mEnv->GetControlHz()},
    };
    _plotGraphData(mGraphData, {"phase_displacement", "phase_lTime"}, "Phase", -0.005, 0.005, &horizontal_lines_phase, false);
}

void GLFWApp::_drawAction()
{
    const auto action = mEnv->GetAction();
    const auto numAction = action.rows();
    const auto action_scale = mEnv->GetActionScale();
    const auto numHidden = 512;

    const auto main_tensor = mNN.getTensorData("main_mu");
    const auto mod1_tensor = mNN.getTensorData("mod1_mu");
    const auto mod2_tensor = mNN.getTensorData("mod2_mu");
    const auto mod3_tensor = mNN.getTensorData("mod3_mu");
    const auto mod4_tensor = mNN.getTensorData("mod4_mu");
    const double* main_data = main_tensor.data_ptr<double>();
    const double* mod1_data = mod1_tensor.data_ptr<double>();
    const double* mod2_data = mod2_tensor.data_ptr<double>();
    const double* mod3_data = mod3_tensor.data_ptr<double>();
    const double* mod4_data = mod4_tensor.data_ptr<double>();
    const vector<double> y_main(main_data, main_data + numAction);
    const vector<double> y_mod1(mod1_data, mod1_data + numHidden);
    const vector<double> y_mod2(mod2_data, mod2_data + numHidden);
    const vector<double> y_mod3(mod3_data, mod3_data + numHidden);
    const vector<double> y_mod4(mod4_data, mod4_data + numAction);

    ImPlot::SetNextAxesLimits(-0.5, numAction + 0.5, -5.0, 5.0);
    if (ImPlot::BeginPlot("Action"))
    {
        float bar_width = 0.2;
        if (mBarXCoords.find("Action") == mBarXCoords.end() || mBarXCoords.find("ActionMain") == mBarXCoords.end() || mBarXCoords.find("ActionMod") == mBarXCoords.end()) {
            mBarXCoords["Action"] = std::vector<double>(numAction);
            mBarXCoords["ActionMain"] = std::vector<double>(numAction);
            mBarXCoords["ActionMod"] = std::vector<double>(numAction);
            for(int i = 0; i < numAction; i++) {
                mBarXCoords["Action"][i] = i - bar_width;
                mBarXCoords["ActionMain"][i] = i;
                mBarXCoords["ActionMod"][i] = i + bar_width;
            }
        }
        vector<double> y_action(numAction);
        for(int i = 0; i < numAction; i++) {
            y_action[i] = action[i] / action_scale;
        }
        ImPlot::PlotBars("Action", mBarXCoords["Action"].data(), y_action.data(), numAction, bar_width);
        ImPlot::PlotBars("Main", mBarXCoords["ActionMain"].data(), y_main.data(), numAction, bar_width);
        ImPlot::PlotBars("Modulation", mBarXCoords["ActionMod"].data(), y_mod4.data(), numAction, bar_width);
        ImPlot::EndPlot();
    }

    // ImPlot::SetNextAxesLimits(-0.5, numHidden + 0.5, -1.0, 1.0);
    // if (ImPlot::BeginPlot("Hidden")) {
    //     float bar_width = 0.2;
    //     if (mBarXCoords.find("HiddenL1") == mBarXCoords.end() || mBarXCoords.find("HiddenL2") == mBarXCoords.end() || mBarXCoords.find("HiddenL3") == mBarXCoords.end() || mBarXCoords.find("HiddenL4") == mBarXCoords.end()) {
    //         mBarXCoords["HiddenL1"] = std::vector<double>(numHidden);
    //         mBarXCoords["HiddenL2"] = std::vector<double>(numHidden);
    //         mBarXCoords["HiddenL3"] = std::vector<double>(numHidden);
    //         for(int i = 0; i < numHidden; i++) {
    //             mBarXCoords["HiddenL1"][i] = i - bar_width;
    //             mBarXCoords["HiddenL2"][i] = i;
    //             mBarXCoords["HiddenL3"][i] = i + bar_width;
    //         }
    //     }
    //     ImPlot::PlotBars("Hidden1", mBarXCoords["HiddenL1"].data(), y_mod1.data(), numHidden, bar_width);
    //     ImPlot::PlotBars("Hidden2", mBarXCoords["HiddenL2"].data(), y_mod2.data(), numHidden, bar_width);
    //     ImPlot::PlotBars("Hidden3", mBarXCoords["HiddenL3"].data(), y_mod3.data(), numHidden, bar_width);
    //     ImPlot::EndPlot();
    // }

    const auto torques_char = mEnv->GetDesiredTorque();
    const auto numTorques = torques_char.rows();
    ImPlot::SetNextAxesLimits(0.0, numTorques + 0.5, -60.0, 60.0);
    if (ImPlot::BeginPlot("SpdTorques"))
    {
        if (mBarXCoords.find("Torques") == mBarXCoords.end()) {
            mBarXCoords["Torques"] = std::vector<double>(numTorques);
            for(int i = 0; i < numTorques; i++) mBarXCoords["Torques"][i] = i;
        }
        vector<double> y_torques(numTorques);
        for(int i = 0; i < numTorques; i++) y_torques[i] = torques_char[i];
        ImPlot::PlotBars("", mBarXCoords["Torques"].data(), y_torques.data(), numTorques, 1.0);
        ImPlot::EndPlot();
    }
}

void GLFWApp::_drawMax()
{
    _plotGraphData(mGraphData, {"max_joint_velocity", "max_body_velocity"}, "Max Velocity", 0.0, 30.0);
    _plotGraphData(mGraphData, {"max_joint_acceleration", "max_body_acceleration"}, "Max Acceleration", 0.0, 300.0);
}

void GLFWApp::_drawGaitAnalysis()
{
    // moment
    if (mEnv->GetUseMuscle()) {
        _plotGraphData(mGraphData, 
            {
                // "moment_muscle_hipR", "moment_muscle_hipRy_ie", "moment_muscle_hipRz_aa",
            "moment_muscle_hipRx_ef", "moment_muscle_kneeR", "moment_muscle_ankleRx",
            // "moment_tau_hipR", "moment_tau_hipRx", "moment_tau_kneeR", "moment_tau_ankleR", 
            "dev_assist_bw"},
            "Moment (Partial)", -1.0, 1.0, nullptr, true);
    } else {
        _plotGraphData(mGraphData,
            {"moment_tau_kneeR", "moment_tau_ankleR", "moment_tau_hipR", "moment_tau_hipRx", "dev_assist_bw"},
            "Moment (Partial)", -1.5, 2.5, nullptr, true);
    }
    // power
    if (mEnv->GetUseMuscle()) {
        _plotGraphData(mGraphData, 
            {"power_muscle", "power_muscle_hipR", "power_muscle_hipRx_ef", "power_muscle_kneeR", "power_muscle_ankleR", "power_dev_bw"},
            "Power (Partial)", -0.5, 2, nullptr, true);
    } else {
        _plotGraphData(mGraphData, 
            {"power_tau", "power_tau_hipR", "power_tau_kneeR", "power_tau_ankleR"},
            "Power (Partial)", -0.5, 2, nullptr, true);
    }
    // angle
    mOpendGraphTab = true;
    _plotGraphData(mGraphData, {"angle_HipR", "angle_KneeR", "angle_AnkleR"}, "Major Angle", -40.0, 40.0, nullptr, true);
    mOpendGraphTab = false;
    _plotGraphData(mGraphData, {"angle_HipAbR", "angle_HipIRR", "angle_Obliquity", "angle_Rotation", "angle_Tilt", "sway_Torso_X"}, "Minor angle", -40.0, 40.0, nullptr, true);
    // GRF (Ground Reaction Force)
    _plotGraphData(mGraphData, {"grf_x", "grf_y", "grf_z", "contact_GRF_R"}, "GRF Components (BW)", -0.5, 2.0, nullptr, true);
    // swing, stance ratio
    if(ImGui::CollapsingHeader("Stance Ratio")) {
        static const char* labels[] = {"Stance", "Swing"};
        const float stance_ratio = mEnv->GetStanceRatioR() * 100;
        const float swing_ratio = 100 - stance_ratio;
        float ratio[] = {stance_ratio, swing_ratio};
        if(ImPlot::BeginPlot("Stance Ratio", ImVec2(300, 300), mPlotPie)) {
            ImPlot::SetupLegend(ImPlotLocation_NorthWest);
            ImPlot::PlotPieChart(labels, ratio, 2, 0.5, 0.5, 0.5, "%.1f");
            ImPlot::EndPlot();
        }
    }
}

void GLFWApp::_drawDevice()
{
        _plotGraphData(mGraphData, {"dev_angleR", "dev_angleR_raw"}, "Device Angle", -60.0, 20.0);
        _plotGraphData(mGraphData, {"dev_assist_raw"}, "Moment (Device)", -6.0, 6.0);
        _plotGraphData(mGraphData, {"dev_power" }, "Power (Device)", -4.0, 20.0);
        _plotGraphData(mGraphData, {"dev_velocity"}, "Velocity (Device)", -4.0, 4.0);
}

void GLFWApp::DrawUIDisplay_MuscleForce()
{
    _plotGraphData(mGraphData, {
        "force_glt_max", "force_glt_med", "force_quadriceps", "force_hamstrings", 
        "force_tib_a", "force_tib_p", "force_sol_gas",
        // "force_total",
    }, "Force", -20.0, 1200.0);
}

std::pair<double, double> GLFWApp::_setupGraphAxes(const CBufferData& graph_data, const std::vector<std::string>& keys, int default_x_len, 
    ImAxis y_axis, double y_min, double y_max, bool show_phase){
    const auto first_buffer = graph_data[keys[0]];
    double x_max = 0.1;
    double x_min = (default_x_len==-1) ? static_cast<double>(first_buffer.capacity()) : default_x_len;
    x_min *= -1;
    if (show_phase) x_min *= mEnv->GetWorld()->getTimeStep();

    if (!(Utils::close(mXmax, 0.0) && Utils::close(mXmin, 0.0))) ImPlot::SetupAxisLimits(ImAxis_X1, mXmin, mXmax, ImGuiCond_Always);
    else ImPlot::SetupAxisLimits(ImAxis_X1, x_min, x_max, ImGuiCond_Appearing);
    ImPlot::SetupAxisLimits(y_axis, y_min, y_max, ImGuiCond_Appearing);
    return std::make_pair(x_min, x_max);
}

void GLFWApp::_plotHorizontalBar(const std::map<std::string, float>* horizontal, double x_min, double x_max){
    if (horizontal != nullptr && !horizontal->empty()) {
        std::vector<double> x_vals = {x_min, x_max};
        for (const auto& [field, yValue] : *horizontal) {
            std::vector<double> y_vals = {yValue, yValue};
            ImPlot::PlotLine(field.c_str(), x_vals.data(), y_vals.data(), static_cast<int>(x_vals.size()));
        }
    }else{
        std::cerr << "[GUI] Key horizontal not found in mGraphData" << std::endl;
    }
}

void GLFWApp::_plotPhaseBar(const CBufferData& graph_data, double x_min, double x_max, double y_min, double y_max){
    if (graph_data.key_exists("contact_phaseR")) {
        const boost::circular_buffer<double> &contact_phase_buffer = graph_data["contact_phaseR"];

        // Ensure there are at least two points to compare
        if (contact_phase_buffer.size() >= 2) {
            bool prev_phase = static_cast<bool>(contact_phase_buffer[0]);
            ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 2.0f); // Thicker line
            for (int i = 1; i < contact_phase_buffer.size(); ++i) {
                bool current_phase = static_cast<bool>(contact_phase_buffer[i]);
                bool phase_change = prev_phase != current_phase;
                prev_phase = current_phase;
                const double x_val = (-(static_cast<int>(contact_phase_buffer.size())) + i) * mEnv->GetWorld()->getTimeStep();
                if (phase_change) {
                    const auto color = (current_phase == STANCE_PHASE) ? IM_COL32(127, 0, 0, 255) : IM_COL32(0, 0, 127, 255); // Red: Heel strike, Blue: Toe off
                    ImPlot::PushStyleColor(ImPlotCol_Line, color);
                    const vector<double> x_vals = {x_val, x_val};
                    const vector<double> y_vals = {y_min, y_max};
                    ImPlot::PlotLine("", x_vals.data(), y_vals.data(), static_cast<int>(x_vals.size()));
                    ImPlot::PopStyleColor();
                }
            }
            ImPlot::PopStyleVar();
        }
    }else{
        std::cerr << "[GUI] Key contact_phaseR not found in mGraphData" << std::endl;
    }
}

void GLFWApp::_plotHeadlessGraphData(const CBufferData& graph_data, const vector<string>& keys, ImAxis y_axis, 
    bool show_phase, bool plot_avg_copy, std::string postfix){
    int bufferSize = static_cast<int>(graph_data[keys[0]].size());
    vector<float> x(bufferSize);
    for (int i = 0; i < bufferSize; ++i) {
        x[i] = -i;
        if (show_phase) x[i] *= mEnv->GetWorld()->getTimeStep();
    }

    ImPlot::SetAxis(y_axis);

    int color_idx = postfix.size() > 0 ? mStoreColorOffset : 0;
    for (const auto &key : keys) {
        if (graph_data.key_exists(key)) {
            // Only proceed if key exists
            const boost::circular_buffer<double> &buffer = graph_data[key];

            // Create y values with the size of the buffer
            vector<float> smooth_y(buffer.size()), y(buffer.size());
            if (mPlotSmoothAlpha > 0.0f){
                int n = buffer.size();
                smooth_y[0] = buffer[n-1];
                for (int i = 1; i < n; ++i) {
                    smooth_y[i] = buffer[n-i-1] * mPlotSmoothAlpha + (1.0f - mPlotSmoothAlpha) * smooth_y[i-1];
                }
            }
            transform(buffer.rbegin(), buffer.rend(), y.begin(), [](float val) -> double { return static_cast<float>(val);});

            // Plot the line
            size_t underscore_pos = key.find('_');
            string selected_key = key;
            if (underscore_pos != std::string::npos) { // Check if underscore exists
                selected_key = key.substr(underscore_pos + 1); // Extract substring after underscore
            }
            selected_key = selected_key + postfix;
            ImPlot::PushStyleColor(ImPlotCol_Line, plot_color_list[color_idx]);
            if (mPlotHideLegend) ImPlot::HideNextItem(true, ImPlotCond_Always);
            auto it = std::find(mShowGraphData.begin(), mShowGraphData.end(), key);
            if (it != mShowGraphData.end()) ImPlot::HideNextItem(false, ImPlotCond_Always);

            const bool plot_smooth = mPlotSmoothAlpha > 0.0f && mPlotShowSmooth;
            if (plot_smooth) ImPlot::PlotLine(selected_key.c_str(), x.data(), smooth_y.data(), static_cast<int>(buffer.size()));
            else ImPlot::PlotLine(selected_key.c_str(), x.data(), y.data(), static_cast<int>(buffer.size()));

            // Add average value annotations if enabled
            if (mPlotAverage && !GetLastItemHidden()) {
                const int numSegments = bufferSize / mPlotAvgTxtInterval;
                if (numSegments > 0) { // Only add annotations if we have enough data
                    for (int segment = 0; segment < numSegments; segment += 1) {
                        int startIdx = segment * mPlotAvgTxtInterval;
                        int endIdx = (segment + 1) * mPlotAvgTxtInterval - 1;

                        // Cutoff marginal data
                        startIdx += 2;
                        endIdx -= 2;
                        
                        // Calculate average for this segment
                        double sum = 0.0;
                        int count = 0;
                        for (int i = startIdx; i <= endIdx && i < y.size(); i++) {
                            sum += y[i];
                            count++;
                        }
                        
                        if (count > 0) {
                            double avg = sum / count;
                            // Position for text (middle of segment on x-axis)
                            double x_pos = x[(endIdx + startIdx) / 2];
                            
                            // Format the average value with adjustable decimal places
                            std::ostringstream ss;
                            ss << std::fixed << std::setprecision(mPlotAvgDecimal) << avg;
                            std::string avgText = ss.str();
                            if (plot_avg_copy) cout << avgText << endl;
                            
                            // Plot the text with the same color as the line
                            // get font size
                            ImGui::SetWindowFontScale(mPlotAvgFontScale);
                            ImPlot::PlotText(avgText.c_str(), x_pos, avg, ImVec2(0, mPlotAvgTxtOffset));
                            ImGui::SetWindowFontScale(1.0f);
                        }
                    }
                }
            }

            // Add std value annotations if enabled
            if (mPlotJitter && !GetLastItemHidden()) {
                if (mPlotSmoothAlpha < 0.0f) {
                    ImPlot::PlotText("Invalid LPF Alpha", (mXmin + mXmax) / 2, 0, ImVec2(0, -mPlotAvgTxtOffset));
                } else {
                    double sum_hf_sq = 0.0;
                    double sum_lf_sq = 0.0;
                    const int n = buffer.size();
                    for (int i = 0; i < n; ++i) {
                        double hf_res = y[i] - smooth_y[i];
                        sum_hf_sq += hf_res * hf_res;
                        sum_lf_sq += smooth_y[i] * smooth_y[i];
                    }
                    double jitter = sum_hf_sq / sum_lf_sq;
                    // Set precision for std value using stringstream
                    std::ostringstream jitter_ss;
                    jitter_ss << "Jitter(" << std::fixed << std::setprecision(mPlotAvgDecimal + 1) << mPlotSmoothAlpha << "): " << std::setprecision(mPlotAvgDecimal + 2) << jitter;
                    ImPlot::PlotText(jitter_ss.str().c_str(), (mXmin + mXmax) / 2, 0, ImVec2(0, -mPlotAvgTxtOffset));
                }
            }

            if (mPlotDiff && !GetLastItemHidden()) {
                double sum_diff = 0.0;
                const int n = buffer.size() - 1;
                if (plot_smooth) {
                    for (int i = 0; i < n; ++i) sum_diff += abs(smooth_y[i] - smooth_y[i+1]);
                } else {
                    for (int i = 0; i < n; ++i) sum_diff += abs(y[i] - y[i+1]);
                }
                double diff = sum_diff / n;
                std::ostringstream diff_ss;
                diff_ss << "Diff: " << std::fixed << std::setprecision(mPlotAvgDecimal + 2) << diff;
                ImPlot::PlotText(diff_ss.str().c_str(), (mXmin + mXmax) / 2, 0, ImVec2(0, -mPlotAvgTxtOffset));
            }
            
            ImPlot::PopStyleColor();
            color_idx = (color_idx + 1) % mNumPlotColor;
        } else {
            // Log the missing key
            std::cerr << "Key " << key << " not found in mGraphData" << std::endl;
        }
    }
}

void GLFWApp::_plotGraphData(const CBufferData& graph_data, const std::vector<std::string>& keys, const std::string& title, 
    double y_min, double y_max, const map<string, float>* horizontal, bool show_phase, float mHeight, int default_x_len, 
    bool hide_legend){
    // Use static maps to track double-size state and show-average state for each plot title
    static std::map<std::string, bool> doublePlotSizeMap;
    
    // Initialize the map entries if they don't exist
    if (doublePlotSizeMap.find(title) == doublePlotSizeMap.end()) {
        doublePlotSizeMap[title] = false;
    }

    for(const auto& key : keys){
        if(!graph_data.key_exists(key)) {
            std::cerr << "[GUI] Key " << key << " not found in mGraphData" << std::endl;
            return;
        }
    }
    
    // Calculate position for the checkboxes
    float windowWidth = ImGui::GetWindowWidth();
    float checkboxWidth = 45.0f, copyButtonWidth = 30.0f, inputWidth = 40.0f; // Approximate mWidth for each checkbox
    float checkboxPosX = windowWidth - checkboxWidth - 10; // 10px padding
    float copyCheckboxPosX = checkboxPosX - copyButtonWidth - 5; // Position for copy checkbox
    // Save cursor position
    ImVec2 cursorPos = ImGui::GetCursorPos();
    
    // Position the 2x checkbox
    ImGui::SetCursorPos(ImVec2(checkboxPosX, cursorPos.y));
    ImGui::Checkbox(("2x##" + title).c_str(), &doublePlotSizeMap[title]);

    // Position the copy checkbox
    ImGui::SetCursorPos(ImVec2(copyCheckboxPosX, cursorPos.y));
    bool copyButtonPressed = ImGui::Button(("Copy##" + title).c_str());
    
    // Restore cursor position for the header
    ImGui::SetCursorPos(cursorPos);

    // Draw the collapsing header
    bool headerOpen = false;
    if (mOpendGraphTab) {
        headerOpen = ImGui::CollapsingHeader(title.c_str(), ImGuiTreeNodeFlags_DefaultOpen);
    } else {
        headerOpen = ImGui::CollapsingHeader(title.c_str());
    }
        
    if(headerOpen){
        if (graph_data[keys[0]].size() > 0) {
            // Apply double mHeight if checkbox is checked for this specific plot
            float plotHeight = doublePlotSizeMap[title] ? mHeight * 2.0f : mHeight;
            string filename = filesystem::path(mArgs.torchscript_dir).filename().string();
            string title_str = mPlotTitle ? filename : title;
            string postfix = "";
            if (mRolloutStatus.settingPath.size() > 0) {
                filesystem::path path(mRolloutStatus.settingPath);
                postfix = path.stem().string();
            }
            if (mRolloutStatus.memo[0] != 0) {
                if (postfix.size() > 0) postfix += " | ";
                postfix += string(mRolloutStatus.memo);
            }
            title_str += "\n" + postfix;
            if (ImPlot::BeginPlot(title_str.c_str(), ImVec2(-1, plotHeight))){
                ImPlot::SetupLegend(ImPlotLocation_NorthWest, ImPlotLegendFlags_Horizontal);
                auto [x_min, x_max] = _setupGraphAxes(graph_data, keys, default_x_len, ImAxis_Y1, y_min, y_max, show_phase);
                
                if (horizontal != nullptr) _plotHorizontalBar(horizontal, x_min, x_max);
                if (show_phase) _plotPhaseBar(graph_data, x_min, x_max, y_min, y_max);

                _plotHeadlessGraphData(graph_data, keys, ImAxis_Y1, show_phase, copyButtonPressed);
                
                // Convert deque to vector for the second call
                std::deque<std::string> storedKeysde = mStoredGraphData.keys();
                std::vector<std::string> storedKeys(storedKeysde.begin(), storedKeysde.end());
                if (storedKeys.size() > 0) _plotHeadlessGraphData(mStoredGraphData, storedKeys, ImAxis_Y1, show_phase, copyButtonPressed, "_str");
                mShowGraphData.clear();
                ImPlot::EndPlot();
            }
        }
    }
}


// ======================================= GL =======================================

void GLFWApp::initGL() 
{
    glClearColor(1.0, 1.0, 1.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);
    glShadeModel(GL_SMOOTH);
    glPolygonMode(GL_FRONT, GL_FILL);
    
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glDisable(GL_CULL_FACE);
}

void GLFWApp::initFog() 
{
	glEnable(GL_FOG);
	GLfloat fogColor[] = {0.95, 0.95, 0.95, 0.95};
  	glFogfv(GL_FOG_COLOR, fogColor);
  	glFogi(GL_FOG_MODE, GL_LINEAR);
    glFogf(GL_FOG_DENSITY, 0.5);
  	glFogf(GL_FOG_START, 0.0);
  	glFogf(GL_FOG_END, 12.0);
}

void GLFWApp::initLights() 
{
    static float ambient[] = {0.2, 0.2, 0.2, 1.0};
    static float diffuse[] = {0.6, 0.6, 0.6, 1.0};
    static float mat_shininess[] = {60.0};
    static float mat_specular[] = {0.2, 0.2, 0.2, 1.0};
    static float mat_ambient[] = {0.2, 0.2, 0.2, 1.0};
    static float mat_diffuse[] = {0.5, 0.28, 0.38, 1.0};
    static float lmodel_ambient[] = {0.2, 0.2, 0.2, 1.0};
    static float lmodel_twoside[] = {GL_FALSE};
    GLfloat position[] = {1.0, 0.0, 0.0, 0.0};
    GLfloat position1[] = {-1.0, 0.0, 0.0, 0.0};
    
    glEnable(GL_LIGHT0);
    glLightfv(GL_LIGHT0, GL_AMBIENT, ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse);
    glLightfv(GL_LIGHT0, GL_POSITION, position);
    
    glEnable(GL_LIGHT1);
    glLightfv(GL_LIGHT1, GL_AMBIENT, ambient);
    glLightfv(GL_LIGHT1, GL_DIFFUSE, diffuse);
    glLightfv(GL_LIGHT1, GL_POSITION, position1);
    
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
    glLightModelfv(GL_LIGHT_MODEL_TWO_SIDE, lmodel_twoside);
    
    glEnable(GL_LIGHTING);
    
    glEnable(GL_RESCALE_NORMAL);  

    // Color material
    glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, mat_shininess);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mat_specular);
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, mat_diffuse);
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, mat_ambient);
    
    glEnable(GL_COLOR_MATERIAL);    
}

void GLFWApp::SetMuscleColor()
{
	int idx = 0;
	auto m_l = mEnv->GetMuscleLengthParams();
	auto m_f = mEnv->GetMuscleForceParams();
	
    mIsSelectedMode = false;

    for(int i = 0; i < mEnv->GetMuscleLengthParamNum() + mEnv->GetMuscleForceParamNum(); i++)
	{
		if(i < mEnv->GetMuscleLengthParamNum())
			for(auto m_e : m_l[i].muscle)
				m_e->selected = mSelectedParameter[i];
		else if(i < mEnv->GetMuscleForceParamNum())
			for(auto m_e : m_f[i - mEnv->GetMuscleLengthParamNum()].muscle)
				m_e->selected = mSelectedParameter[i];
	
        mIsSelectedMode = mIsSelectedMode || mSelectedParameter[i];
    }	
}

//===========================Draw Sim Frame==================================

void GLFWApp::DrawSimFrame()
{
    if(mDrawMode.getMode() == DrawMode::NoSimUpdate) return;
    SetMuscleColor();

    initGL();
    initLights();
    initFog();
    SetFocusing();

    /* Preprocessing */
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glViewport(0, 0, mWidth, mHeight);
    gluPerspective(mPersp, mWidth / mHeight, 0.1, 10.0);
    gluLookAt(mEye[0], mEye[1], mEye[2], 0.0, 0.0, -1.0, mUp[0], mUp[1], mUp[2]);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    mTrackball.setCenter(Eigen::Vector2d(mWidth*0.5, mHeight*0.5));
	mTrackball.setRadius(std::min(mWidth, mHeight)/2.5);
    mTrackball.applyGLRotation();
    {
        glDisable(GL_LIGHTING);
        glLineWidth(2.0);
        if (mRotate || mTranslate || mZooming)
        {
            glColor3f(1.0f, 0.0f, 0.0f);
            glBegin(GL_LINES);
            glVertex3f(-0.1f, 0.0f, -0.0f);
            glVertex3f(0.15f, 0.0f, -0.0f);
            glEnd();

            glColor3f(0.0f, 1.0f, 0.0f);
            glBegin(GL_LINES);
            glVertex3f(0.0f, -0.1f, 0.0f);
            glVertex3f(0.0f, 0.15f, 0.0f);
            glEnd();

            glColor3f(0.0f, 0.0f, 1.0f);
            glBegin(GL_LINES);
            glVertex3f(0.0f, 0.0f, -0.1f);
            glVertex3f(0.0f, 0.0f, 0.15f);
            glEnd();
        }
        glEnable(GL_LIGHTING);
    }

    glScalef(mZoom, mZoom, mZoom);
    glTranslatef(mTrans[0] * 0.001, mTrans[1] * 0.001, mTrans[2] * 0.001);

    GLfloat matrix[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, matrix);
    Eigen::Matrix3d A;
    Eigen::Vector3d b;
    A << matrix[0], matrix[4], matrix[8],
            matrix[1], matrix[5], matrix[9],
            matrix[2], matrix[6], matrix[10];
    b << matrix[12], matrix[13], matrix[14];
    mViewMatrix.linear() = A;
    mViewMatrix.translation() = b;

    if(mDrawGround) DrawGround();
    if(mDrawMuscleTorque) DrawMuscleTorque();
    if(mDrawCollision) DrawCollision();
    if(mDrawJoint) DrawJoint();
    if(mDrawNodeCOM) DrawNodeCOM();
    if(mEnv->GetUseMuscle() && mDrawMuscle) DrawMuscles(mEnv->GetCharacter()->GetMuscles());
    if(mDrawCharacter) DrawSkeleton(mEnv->GetCharacter()->GetSkeleton());

    if(mDrawReference){
        Eigen::VectorXd p = mEnv->GetBVHPositions();
        Eigen::VectorXd v = mEnv->GetBVHVelocities();
        SkeletonPtr skel_ref = mEnv->GetReferenceSkeleton();
        Utils::setSkelPosAndVel(skel_ref, p, v);
        DrawSkeleton(skel_ref);
	}

    if(mDrawDevice) DrawDevice();
    if(mDrawFootStep) DrawFootStep();
    if(mDrawPhaseState) DrawPhaseState();
}

void GLFWApp::DrawTitle()
{
    const float title_scale = 1.5f;

    // Save original font scale
    float originalFontScale = ImGui::GetFont()->Scale;

    // Center the window and text
    string filename = filesystem::path(mArgs.torchscript_dir).filename().string();
    const char* title = filename.c_str();
    float windowWidth = ImGui::CalcTextSize(title).x * title_scale;

    ImGui::SetNextWindowPos(ImVec2((mWidth - windowWidth)* 0.5f, 300), ImGuiCond_Once);
    ImGui::SetNextWindowSize(ImVec2(windowWidth, 50), ImGuiCond_Always);
    
    ImGui::Begin("##torchscript_title", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar);
    ImGui::GetFont()->Scale = title_scale; // Make the title larger
    ImGui::PushFont(ImGui::GetFont());
    ImGui::TextUnformatted(title);
    ImGui::GetFont()->Scale = originalFontScale;
    ImGui::PopFont();

    ImGui::End();
}

void GLFWApp::DrawGround()
{
    if (mEnv->GetUseTerrain()) {
        // Draw terrain
        for (const auto& terrain : mEnv->GetTerrainManager()->getTerrainSkeletons()) {
            for (const auto& bn : terrain->getBodyNodes()) {
                DrawBodyNode(bn);
            }
        }
    } else {
        auto ground = mEnv->GetGround();
        auto groundNode = ground->getBodyNode(0);
        Eigen::Vector3d size;
        groundNode->eachShapeNodeWith<VisualAspect>([&size](ShapeNode* sn) {
            auto shape = sn->getShape();
            if(shape->is<BoxShape>()) {
                size = dynamic_cast<const BoxShape*>(shape.get())->getSize();
                return false;
            }
            return true;
        });

        const double mWidth = size[0];
        const double mHeight = size[1];
        const double depth = size[2];

        glDisable(GL_LIGHTING);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        
        if (mGroundMode.getMode() == GroundMode::Checker) {
            // Draw checkerboard pattern
            glBegin(GL_QUADS);
            
            Eigen::Vector3d color1,color2;
            color1 << 216.0/255.0, 211.0/255.0, 204.0/255.0;
            color2 << 216.0/255.0-0.1, 211.0/255.0-0.1, 204.0/255.0-0.1;    
            const double grid_size = 1.0;

            for(double x=-mWidth/2.0; x<mWidth/2.0; x+=grid_size)
            {
                for(double z=-depth/2.0; z<depth/2.0; z+=grid_size)
                {
                    bool isEven = (int(x/grid_size) + int(z/grid_size)) % 2 == 0;
                    if(isEven) glColor4f(color1[0], color1[1], color1[2] ,1.0);			
                    else glColor4f(color2[0], color2[1], color2[2] ,1.0);	
                    glVertex3f(x,           0.0, z);
                    glVertex3f(x+grid_size, 0.0, z);
                    glVertex3f(x+grid_size, 0.0, z+grid_size);
                    glVertex3f(x,           0.0, z+grid_size);
                }
            }
            glEnd();
        } else if (mGroundMode.getMode() == GroundMode::Chroma) {
            // Draw green translucent chroma key floor
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            
            glBegin(GL_QUADS);
            // Green chroma key color (RGB: 0, 255, 0) with transparency
            glColor4f(0.0, 1.0, 0.0, 0.3);
            
            glVertex3f(-mWidth/2.0, 0.0, -depth/2.0);
            glVertex3f( mWidth/2.0, 0.0, -depth/2.0);
            glVertex3f( mWidth/2.0, 0.0,  depth/2.0);
            glVertex3f(-mWidth/2.0, 0.0,  depth/2.0);
            glEnd();
            
            glDisable(GL_BLEND);
        }
        
        glEnable(GL_LIGHTING);
    }
}

void GLFWApp::_drawDesiredTorque()
{
    
}

void GLFWApp::DrawMuscleTorque()
{
    // mEnv->GetJointMoment();
    glDisable(GL_LIGHTING);
    for(auto j : mEnv->GetCharacter()->GetSkeleton()->getJoints())
    {
        Eigen::Isometry3d t = j->getChildBodyNode()->getTransform() * j->getTransformFromChildBodyNode();
        Eigen::Vector3d p;
        glPushMatrix();
        if(j->getNumDofs() == 3)
        {
            p = t * Eigen::Vector3d::Zero();
            glTranslated(p[0], p[1], p[2]);
            GUI::DrawSphere(0.005);
            {
                Eigen::Vector3d f = 0.01 * mMuscleTorque.segment(j->getIndexInSkeleton(0), 3);
                GUI::DrawLine(Eigen::Vector3d::Zero(), f, Eigen::Vector3d(1,0,0));
            }
        }
        else if(j->getNumDofs() == 1)
        {
            p = t * Eigen::Vector3d(0,0,0);
            glTranslatef(p[0], p[1], p[2]);   
            {
                double f  = 0.01 * mMuscleTorque[j->getIndexInSkeleton(0)];
                GUI::DrawLine(Eigen::Vector3d::Zero(), Eigen::Vector3d(f,0,0), Eigen::Vector3d(1,0,0));
            }
            glRotated(90, 0, 1, 0);
            GUI::DrawCylinder(0.005,0.005);            
        }
        glPopMatrix();
    }
    glEnable(GL_LIGHTING);
}

void GLFWApp::DrawCollision()
{
    glDisable(GL_LIGHTING);
    const auto result = mEnv->GetWorld()->getConstraintSolver()->getLastCollisionResult();
    for (const auto& contact : result.getContacts()) {
        Eigen::Vector3d v = contact.point;
        Eigen::Vector3d f = contact.force / 3000.0;
        
        // Draw arrow shaft
        glLineWidth(20.0); // Thicker line
        glColor3f(1.0, 0.4, 0.4);
        
        glBegin(GL_LINES);
        glVertex3f(v[0], v[1], v[2]);
        glVertex3f(v[0]+f[0], v[1]+f[1], v[2]+f[2]);
        glEnd();

        // Draw arrow head
        const float arrow_length = f.norm();
        const float head_size = 0.025;
        
        Eigen::Vector3d dir = f.normalized();
        Eigen::Vector3d up = Eigen::Vector3d::UnitY();
        Eigen::Vector3d right = dir.cross(up).normalized();
        if(right.norm() < 0.1) right = Eigen::Vector3d::UnitX();
        Eigen::Vector3d up2 = right.cross(dir).normalized();

        Eigen::Vector3d base = v + f;
        Eigen::Vector3d tip = base + dir * head_size;

        glBegin(GL_TRIANGLES);
        glVertex3f(tip[0], tip[1], tip[2]);
        glVertex3f(base[0] + right[0]*head_size, base[1] + right[1]*head_size, base[2] + right[2]*head_size);
        glVertex3f(base[0] - right[0]*head_size, base[1] - right[1]*head_size, base[2] - right[2]*head_size);

        glVertex3f(tip[0], tip[1], tip[2]);
        glVertex3f(base[0] + up2[0]*head_size, base[1] + up2[1]*head_size, base[2] + up2[2]*head_size);
        glVertex3f(base[0] - up2[0]*head_size, base[1] - up2[1]*head_size, base[2] - up2[2]*head_size);
        glEnd();
                
        glPushMatrix();
        glTranslated(v[0], v[1], v[2]);
        glColor3f(0.5, 0.2, 0.5);
        GUI::DrawSphere(0.01);
        glPopMatrix();
    }
    
    glEnable(GL_LIGHTING);    
}

void GLFWApp::DrawJoint()
{
    glDisable(GL_LIGHTING);
    glColor4f(0.5,0.5,0.5,1.0);
    for(auto j : mEnv->GetCharacter()->GetSkeleton()->getJoints())
    {
        Eigen::Isometry3d t = j->getChildBodyNode()->getTransform() * j->getTransformFromChildBodyNode();
        Eigen::Vector3d p;
        glPushMatrix();
        if(j->getNumDofs() == 3)
        {
            p = t * Eigen::Vector3d::Zero();
            glTranslated(p[0], p[1], p[2]);
            GUI::DrawSphere(0.01);
        }
        else if(j->getNumDofs() == 1)
        {
            p = t * Eigen::Vector3d(0,0,0);
            glTranslatef(p[0], p[1], p[2]);   
            glRotated(90, 0, 1, 0);
            GUI::DrawCylinder(0.01,0.01);
        }
        glPopMatrix();
    }
    glEnable(GL_LIGHTING);
}

void GLFWApp::DrawNodeCOM()
{
    for(auto bn : mEnv->GetCharacter()->GetSkeleton()->getBodyNodes())
    {
        Eigen::Vector3d p = bn->getCOM();
        glPushMatrix();
        glTranslated(p[0], p[1], p[2]);
        GUI::DrawSphere(0.005);
        glPopMatrix();
    }
}

void GLFWApp::DrawMuscles(const std::vector<Muscle*>& muscles)
{
    Eigen::Vector4d m_color;
    m_color << 0.55, 0.55, 0.55, 0.95;

    if(!mDrawMuscleTotal) return;

    if(mDrawMuscleMode){
        for (auto muscle : muscles)
        {            
            std::string name = muscle->GetName();
            if(mMuscleRenderMap[name]){
                double p = 0.0;
                if(mDrawMusclePassive){
                    double f_p = muscle->GetPassiveForce();
                    double f0 = muscle->GetF0();
                    p = f_p/f0;
                    Eigen::Vector4d color_;
                    color_ << m_color[0], m_color[1], m_color[2]+20.0*p, m_color[3];
                    glColor4dv(color_.data());
                }else{
                    Eigen::Vector4d color_;
                    color_ << m_color[0] + 3 * muscle->GetActivation(), m_color[1], m_color[2], m_color[3];
                    glColor4dv(color_.data());
                }

                mShapeRenderer.renderMuscle(muscle);

                if(mDrawMuscleAnchorMode){
                    const Eigen::Matrix3Xd& anchor_pos = muscle->GetAnchorPos();
                    for(int i = 0; i < anchor_pos.cols(); i++){
                        const float radius = 0.008f * sqrt(muscle->GetF0()/1000.0f);
                        glPushMatrix();
                        glTranslated(anchor_pos(0, i), anchor_pos(1, i), anchor_pos(2, i));
                        glColor4f(0.2, 1.2, 0.2, 0.6);
                        GUI::DrawSphere(radius);
                        glPopMatrix();
                    }
                }
            }
        }
    }else {
        for (auto muscle : muscles)
        {
            double f = muscle->GetF0();
            double a = muscle->GetActivation();
            bool isParted = false;
            Eigen::Vector4d color_;
            color_ << m_color[0] + 3 * a, m_color[1], m_color[2], m_color[3];
            if(mDrawMuscleRangePart && f >= mDrawMuscleRange){
                glColor4dv(color_.data());
                mShapeRenderer.renderMuscle(muscle);
                isParted = true;
            } else if(f < mDrawMuscleRange){
                glColor4dv(color_.data());
                mShapeRenderer.renderMuscle(muscle);
                isParted = true;
            }

            if(mDrawMuscleAnchorMode && isParted){
                const Eigen::Matrix3Xd& anchor_pos = muscle->GetAnchorPos();                
                for(int i = 0; i < anchor_pos.cols(); i++){
                    const float radius = 0.008f*sqrt(muscle->GetF0()/1000.0);
                    glPushMatrix();
                    glTranslated(anchor_pos(0, i), anchor_pos(1, i), anchor_pos(2, i));
                    glColor4f(0.2, 1.2, 0.2, 0.6);
                    GUI::DrawSphere(radius);
                    glPopMatrix();
                }
            }                          
        }
    }
}

void GLFWApp::DrawBodyNode(const BodyNode* bn)
{	
	if(!bn) return;

    glPushMatrix();
	Eigen::Affine3d tmp = bn->getRelativeTransform();
	glMultMatrixd(tmp.data());

    curBody = bn->getName();
	bn->eachShapeNodeWith<dart::dynamics::VisualAspect>([&](const ShapeNode* sn) {DrawShapeFrame(sn);});
	for(const auto& et : bn->getChildEntities()) DrawEntity(et);

	glPopMatrix();    
}

void GLFWApp::DrawShapeFrame(const ShapeFrame* sf)
{
	if(!sf)
		return;

	const auto& va = sf->getVisualAspect();

	if(!va || va->isHidden())
		return;

	glPushMatrix();
	Eigen::Affine3d tmp = sf->getRelativeTransform();
	glMultMatrixd(tmp.data());
    
    Eigen::Vector4d color = va->getRGBA();
    color[3] = 0.3;
    if(mIsDeviceDrawMode) color[3] = 1.0;
    DrawShape(sf->getShape().get(),color);

	glPopMatrix();
}

void GLFWApp::DrawShape(const Shape* shape, const Eigen::Vector4d& color)
{
	if(!shape)
		return;

    glDisable(GL_LIGHTING);
	glColor4d(color[0], color[1], color[2], color[3]);
	
    if(mDrawMode.getMode() == DrawMode::Mesh)
	{
        glEnable(GL_LIGHTING);        
		if (shape->is<MeshShape>())
		{
			const auto& mesh = dynamic_cast<const MeshShape*>(shape);
            if(mIsDeviceDrawMode) mShapeRenderer.renderMesh(mesh, false, mGroundHeight, Eigen::Vector4d(0.3,0.3,0.3,1.0));
            else mShapeRenderer.renderMesh(mesh, false, mGroundHeight, Eigen::Vector4d(0.55,0.55,0.55,1.0));
            // DrawShadow(mesh->getScale(), mesh->getMesh(), y);
        }        
        glDisable(GL_LIGHTING);
	}
    else if(mDrawMode.getMode() == DrawMode::Primitive)
	{
        glColor4dv(color.data());
        if (shape->is<SphereShape>())
		{
            const auto* sphere = dynamic_cast<const SphereShape*>(shape);
            GUI::DrawSphere(sphere->getRadius());
        }
		else if (shape->is<BoxShape>())
		{
			const auto* box = dynamic_cast<const BoxShape*>(shape);
			GUI::DrawCube(box->getSize());
		}
		else if (shape->is<CapsuleShape>())
		{
			const auto* capsule = dynamic_cast<const CapsuleShape*>(shape);
			GUI::DrawCapsule(capsule->getRadius(), capsule->getHeight());
		}	
        else if (shape->is<CylinderShape>())
		{
			const auto* cylinder = dynamic_cast<const CylinderShape*>(shape);
			GUI::DrawCylinder(cylinder->getRadius(), cylinder->getHeight());
		}
	}

    glEnable(GL_LIGHTING);	
}

void GLFWApp::DrawEntity(const Entity* entity)
{
	if (!entity) return;
	const auto& bn = dynamic_cast<const BodyNode*>(entity);
	if(bn)
	{
		DrawBodyNode(bn);
		return;
	}
	const auto& sf = dynamic_cast<const ShapeFrame*>(entity);
	if(sf)
	{
		DrawShapeFrame(sf);
		return;
	}
}

void GLFWApp::DrawAiMesh(const struct aiScene *sc, const struct aiNode* nd,const Eigen::Affine3d& M,double y)
{
	unsigned int i;
    unsigned int n = 0, t;
    Eigen::Vector3d v;
    Eigen::Vector3d dir(0.4,0,-0.4);
    glColor3f(0.3,0.3,0.3);
    
    // update transform
    
    // draw all meshes assigned to this node
    for (; n < nd->mNumMeshes; ++n) 
    {
        const struct aiMesh* mesh = sc->mMeshes[nd->mMeshes[n]];

        for (t = 0; t < mesh->mNumFaces; ++t) {
            const struct aiFace* face = &mesh->mFaces[t];
            GLenum face_mode;

            switch(face->mNumIndices) {
                case 1: face_mode = GL_POINTS; break;
                case 2: face_mode = GL_LINES; break;
                case 3: face_mode = GL_TRIANGLES; break;
                default: face_mode = GL_POLYGON; break;
            }
            glBegin(face_mode);
        	for (i = 0; i < face->mNumIndices; i++)
        	{
        		int index = face->mIndices[i];

        		v[0] = (&mesh->mVertices[index].x)[0];
        		v[1] = (&mesh->mVertices[index].x)[1];
        		v[2] = (&mesh->mVertices[index].x)[2];
        		v = M*v;
        		double h = v[1]-y;
        		
        		v += h*dir;
        		
        		v[1] = y+0.001;
        		glVertex3f(v[0],v[1],v[2]);
        	}
            glEnd();
        }
    }
    
    // draw all children
    for (n = 0; n < nd->mNumChildren; ++n)  DrawAiMesh(sc, nd->mChildren[n],M,y);
}

void GLFWApp::DrawShadow(const Eigen::Vector3d& scale, const aiScene* mesh,double y) 
{
	glDisable(GL_LIGHTING);
	glPushMatrix();
	glScalef(scale[0],scale[1],scale[2]);
	GLfloat matrix[16];
	glGetFloatv(GL_MODELVIEW_MATRIX, matrix);
	Eigen::Matrix3d A;
	Eigen::Vector3d b;
	A<<matrix[0],matrix[4],matrix[8],
	matrix[1],matrix[5],matrix[9],
	matrix[2],matrix[6],matrix[10];
	b<<matrix[12],matrix[13],matrix[14];

	Eigen::Affine3d M;
	M.linear() = A;
	M.translation() = b;
	M = (mViewMatrix.inverse()) * M;

	glPushMatrix();
	glLoadIdentity();
	glMultMatrixd(mViewMatrix.data());
	DrawAiMesh(mesh,mesh->mRootNode,M,y);
	glPopMatrix();
	glPopMatrix();
	glEnable(GL_LIGHTING);
}


void GLFWApp::DrawDevice()
{
    if(mDrawDevice)
	{
		mIsDeviceDrawMode = true;
        DrawSkeleton(mEnv->GetDevice()->GetSkeleton());
        if(mDrawDeviceTorque) DrawDeviceTorque();
        if(mDrawDeviceTorqueFromThigh) DrawDeviceTorqueFromThigh();
        if(mDrawHipTorque) DrawHipTorque();
        mIsDeviceDrawMode = false;
	}
}

void GLFWApp::DrawForceArrow(float force, const Eigen::Vector3d& dir, const Eigen::Vector3d& pos, const Eigen::Vector4d& color){
    Eigen::Vector3d dir_;
    if (force > 0) dir_ = dir;
    else dir_ = -dir;

    GUI::DrawArrow3D(pos, dir_, 0.1*abs(force), 0.015, color, 0.03);
}

void GLFWApp::DrawDeviceTorque()
{
    const auto& device = mEnv->GetDevice();    
    // For right rod
    if (auto joint_R = device->GetSkeleton()->getJoint("RodRight")) {
        auto ballJoint = dynamic_cast<dart::dynamics::BallJoint*>(joint_R);
        if (ballJoint) {
            Eigen::Isometry3d trans_R = ballJoint->getTransformFromParentBodyNode();
            Eigen::Vector3d p_R = (ballJoint->getParentBodyNode()->getTransform() * trans_R).translation();            
            // Assuming the BallJoint has 3 DOFs
            Eigen::Vector3d torque_R = Eigen::Vector3d(device->GetTorqueR(), 0, 0);
            
            // Transform torque to world frame
            Eigen::Vector3d torque_world_R = trans_R.linear() * torque_R;
            
            // Draw the torque vector
            DrawForceArrow(torque_world_R.norm(), torque_world_R.normalized(), p_R, color_green);
        }
    }

    // For left rod
    if (auto joint_L = device->GetSkeleton()->getJoint("RodLeft")) {
        auto ballJoint = dynamic_cast<dart::dynamics::BallJoint*>(joint_L);
        if (ballJoint) {
            Eigen::Isometry3d trans_L = ballJoint->getTransformFromParentBodyNode();
            Eigen::Vector3d p_L = (ballJoint->getParentBodyNode()->getTransform() * trans_L).translation();
            
            // Assuming the BallJoint has 3 DOFs
            Eigen::Vector3d torque_L = Eigen::Vector3d(device->GetTorqueL(), 0, 0);
            
            // Transform torque to world frame
            Eigen::Vector3d torque_world_L = trans_L.linear() * torque_L;
            
            // Draw the torque vector
            DrawForceArrow(torque_world_L.norm(), torque_world_L.normalized(), p_L, color_red);
        }
    }
}

void GLFWApp::DrawDeviceTorqueFromThigh()
{
    const auto& device = mEnv->GetDevice();
    const double torque_r_device = device->GetTorqueR();
    const double torque_l_device = -device->GetTorqueL();

    // Right thigh (femur) torque representation
    Eigen::Isometry3d trans_R = mEnv->GetCharacter()->GetSkeleton()->getBodyNode("FemurR")->getTransform();
    Eigen::Vector3d p_R = trans_R.translation();
    Eigen::Matrix3d rot_R = trans_R.rotation();
    Eigen::Vector3d dir_R_y = rot_R.col(1);
    Eigen::Vector3d dir_R_z = rot_R.col(2);
    Eigen::Vector3d dir_R1 = dir_R_z;
    Eigen::Vector3d dir_R2 = -dir_R_z;

    // Draw right thigh device torque arrow (red color, size proportional to torque magnitude)
    if(torque_r_device > 0) {
        // GUI::DrawArrow3D(p_R + 0.06*dir_R_y + 0.0699*dir_R1, dir_R1, 0.1*abs(torque_r_device), 0.015, color_red, 0.03);
        GUI::DrawArrow3D(p_R, dir_R1, 0.1*abs(torque_r_device), 0.015, color_red, 0.03);
    } else {
        // GUI::DrawArrow3D(p_R + 0.06*dir_R_y + 0.0699*dir_R2, dir_R2, 0.1*abs(torque_r_device), 0.015, color_red, 0.03);
        GUI::DrawArrow3D(p_R, dir_R2, 0.1*abs(torque_r_device), 0.015, color_red, 0.03);
    }

    // Left thigh (femur) torque representation
    Eigen::Isometry3d trans_L = mEnv->GetCharacter()->GetSkeleton()->getBodyNode("FemurL")->getTransform();
    Eigen::Vector3d p_L = trans_L.translation();
    Eigen::Matrix3d rot_L = trans_L.rotation();
    Eigen::Vector3d dir_L_y = rot_L.col(1);
    Eigen::Vector3d dir_L_z = rot_L.col(2);
    Eigen::Vector3d dir_L1 = dir_L_z;
    Eigen::Vector3d dir_L2 = -dir_L_z;

    // Draw left thigh device torque arrow (red color, size proportional to torque magnitude)
    if(torque_l_device > 0) {
        GUI::DrawArrow3D(p_L + 0.06*dir_L_y + 0.0699*dir_L1, dir_L1, 0.1*abs(torque_l_device), 0.015, color_red, 0.03);
    } else {
        GUI::DrawArrow3D(p_L + 0.06*dir_L_y + 0.0699*dir_L2, dir_L2, 0.1*abs(torque_l_device), 0.015, color_red, 0.03);
    }
}

void GLFWApp::DrawHipTorque()
{
    const Eigen::VectorXd torques_char = mEnv->GetDesiredTorque();
    const double torque_r_char = torques_char[6];
    const double torque_l_char = torques_char[15];

    Eigen::Isometry3d trans_R = mEnv->GetCharacter()->GetSkeleton()->getBodyNode("FemurR")->getTransform();
	Eigen::Vector3d p_R = trans_R.translation();
    Eigen::Matrix3d rot_R = trans_R.rotation();
    Eigen::Vector3d dir_R_x = rot_R.col(0);
    Eigen::Vector3d dir_R_y = rot_R.col(1);
    Eigen::Vector3d dir_R_z = rot_R.col(2);

    Eigen::Vector3d dir_R1 =  dir_R_z;
    Eigen::Vector3d dir_R2 = -dir_R_z;

    if(torque_r_char>0) GUI::DrawArrow3D(p_R+0.06*dir_R_y+0.0699*dir_R1, dir_R1, 0.03*torque_r_char, 0.015, color_blue, 0.03);
    else GUI::DrawArrow3D(p_R+0.06*dir_R_y+0.0699*dir_R2, dir_R2,-0.03*torque_r_char, 0.015, color_blue, 0.03);

	Eigen::Isometry3d trans_L = mEnv->GetCharacter()->GetSkeleton()->getBodyNode("FemurL")->getTransform();
	Eigen::Vector3d p_L = trans_L.translation();
	Eigen::Matrix3d rot_L = trans_L.rotation();
    Eigen::Vector3d dir_L_x = rot_L.col(0);
    Eigen::Vector3d dir_L_y = rot_L.col(1);
    Eigen::Vector3d dir_L_z = rot_L.col(2);

    Eigen::Vector3d dir_L1 = dir_L_z;
    Eigen::Vector3d dir_L2 = -dir_L_z;
    
    if(torque_l_char>0) GUI::DrawArrow3D(p_L+0.06*dir_R_y+0.0699*dir_L1, dir_L1, 0.03*torque_l_char, 0.015, color_blue, 0.03);
    else GUI::DrawArrow3D(p_L+0.06*dir_R_y+0.0699*dir_L2, dir_L2,-0.03*torque_l_char, 0.015, color_blue, 0.03);
}

void GLFWApp::DrawFootStep()
{
	glEnable(GL_COLOR_MATERIAL);
	  
    //Current Target Foot
	Eigen::Vector3d current_target = mEnv->GetCurrentTargetFoot();
    glColor4d(0.2,0.2,0.8,0.5);
    glPushMatrix();
    glTranslated(0, current_target[1], current_target[2]);
	GUI::DrawCube(Eigen::Vector3d(1.0, 0.0785, 0.0785));
    glPopMatrix();

    //Current Chagned Target Foot
	glColor4d(0.2, 0.8, 0.2, 0.5);
    glPushMatrix();
    glTranslated(0, current_target[1], current_target[2]);
	GUI::DrawCube(Eigen::Vector3d(1.0, 0.0785, 0.0785));
    glPopMatrix();

	//Current Next Foot;
	Eigen::Vector3d next_target = mEnv->GetNextTargetFoot();
    glColor4d(0.8, 0.2, 0.2, 0.5);
    glPushMatrix();
    glTranslated(0, next_target[1], next_target[2]);
	GUI::DrawCube(Eigen::Vector3d(1.0, 0.0785, 0.0785));
    glPopMatrix();

    glDisable(GL_COLOR_MATERIAL);    
}

void GLFWApp::DrawPhaseState()
{
    int right = mEnv->GetPhaseStateRight();
    int left = mEnv->GetPhaseStateLeft();

    glDisable(GL_LIGHTING);
    
    BodyNode* footR = mEnv->GetCharacter()->GetSkeleton()->getBodyNode("TalusR");
    BodyNode* footL = mEnv->GetCharacter()->GetSkeleton()->getBodyNode("TalusL");
        
    Eigen::Vector3d footR_com = footR->getTransform().translation();
    Eigen::Vector3d footL_com = footL->getTransform().translation();

    if(right) glColor3f(0.5f, 1.0f, 1.0f);
    else glColor3f(0.5f, 0.5f, 1.0f);

    glPushMatrix();
    glTranslated(footR_com[0], 0.02, footR_com[2] + 0.2);
    GUI::DrawCube(Eigen::Vector3d(0.1, 0.04, 0.1));
    glPopMatrix();

    if(left) glColor3f(0.5f, 1.0f, 1.0f);
    else glColor3f(0.5f, 0.5f, 1.0f);
          
    glPushMatrix();
    glTranslated(footL_com[0], 0.02, footL_com[2] + 0.2);
    GUI::DrawCube(Eigen::Vector3d(0.1, 0.04, 0.1));
    glPopMatrix();  
    glEnable(GL_LIGHTING);    
}

void GLFWApp::_drawPhase()
{
    const double phase = mEnv->GetPhase();
    const double global_phase = mEnv->GetGlobalPhase();

    glDisable(GL_LIGHTING);
    glPushMatrix();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glViewport(0,0,mWidth,mHeight);
    gluOrtho2D(0.0,(GLdouble)mWidth,0.0,(GLdouble)mHeight);

    glPushMatrix();

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // Draw phase clock
    glLineWidth(2.0);
    glColor3f(0.0f, 0.0f, 0.0f);
    glTranslatef(mWidth * 0.5, mHeight * 0.05, 0.0f);
    glBegin(GL_LINE_LOOP);
    for (int i = 0; i < 360; i ++) {
        double theta = i / 180.0 * M_PI;
        double x = mHeight * 0.04 * cos(theta);
        double y = mHeight * 0.04 * sin(theta);
        glVertex2d(x, y);
    }
    glEnd();

    glLineWidth(2.0);
    glColor3f(1,0,0);
    glBegin(GL_LINES);
    glVertex2d(0,0);
    glVertex2d(mHeight * 0.04 * sin(global_phase* 2 * M_PI),mHeight * 0.04 * cos(global_phase * M_PI * 2 ));
    glEnd();

    glColor3f(0,0,0);
    glBegin(GL_LINES);
    glVertex2d(0,0);
    glVertex2d(mHeight * 0.04 * sin(phase* 2 * M_PI),mHeight * 0.04 * cos(phase * M_PI * 2 ));
    glEnd();

    // Draw foot contact indicators
    const double indicator_size = mHeight * 0.03;  // Size of the indicator squares
    const double spacing = indicator_size * 2.8;  // Space between indicators
    
    // Position the indicators next to the phase clock
    // Left foot indicator (slightly to the left of center)
    glColor4f(1.0f, 0.0f, 0.0f, mEnv->GetPhaseStateRight() ? 1.0f : 0.2f);
    glBegin(GL_QUADS);
    glVertex2d(-spacing, -indicator_size * 0.5);
    glVertex2d(-spacing + indicator_size, -indicator_size * 0.5);
    glVertex2d(-spacing + indicator_size, indicator_size * 0.5);
    glVertex2d(-spacing, indicator_size * 0.5);
    glEnd();

    // Right foot indicator (slightly to the right of center)
    glColor4f(1.0f, 0.0f, 0.0f, mEnv->GetPhaseStateLeft() ? 1.0f : 0.2f);
    glBegin(GL_QUADS);
    glVertex2d(spacing - indicator_size, -indicator_size * 0.5);
    glVertex2d(spacing, -indicator_size * 0.5);
    glVertex2d(spacing, indicator_size * 0.5);
    glVertex2d(spacing - indicator_size, indicator_size * 0.5);
    glEnd();

    // Capture region overlay (center-origin inputs -> window space)
    if (mCaptureShowRect)
    {
        // We are already in orthographic projection for window-space drawing
        glLineWidth(2.0);
        glColor4f(1.0f, 0.5f, 0.0f, 1.0f);
        // Convert center-origin (+Y up) to top-left origin (+Y down) for overlay
        glBegin(GL_LINE_LOOP);
        glVertex2d(mCaptureX0, mHeight - mCaptureY0);
        glVertex2d(mCaptureX1, mHeight - mCaptureY0);
        glVertex2d(mCaptureX1, mHeight - mCaptureY1);
        glVertex2d(mCaptureX0, mHeight - mCaptureY1);
        glEnd();
    }

    glPopMatrix();
    glPopMatrix();
    glEnable(GL_LIGHTING);
}

