#include "core/Environment.h"
#include "core/Utils.h"
#include "util/path.h"
#include "util/struct.h"
#include "dart/collision/bullet/bullet.hpp"
#include <yaml-cpp/yaml.h>
#include <initializer_list>
#include <filesystem>
#ifdef LOG_VERBOSE
    #include "util/log.h"
#endif
#include <chrono>

#define SWING_PHASE false
#define STANCE_PHASE true
#define LEFT_STANCE 0
#define RIGHT_STANCE 1

using namespace dart::simulation;
using namespace dart::dynamics;
using namespace boost;
using namespace std;
namespace MASS
{

Environment::Environment(bool isRender, bool force_use_device):
    mForceUseDevice(force_use_device), mIsRender(isRender)
{
    mMedianFootX = mGlobalRatio * (mMaxFootX + mMinFootX) / 2.0f;
    mDiffFootX = mGlobalRatio * (mMaxFootX - mMinFootX) / 2.0f;
    mMedianArmX = mGlobalRatio * (mMaxArmX + mMinArmX) / 2.0f;
    mDiffArmX = mGlobalRatio * (mMaxArmX - mMinArmX) / 2.0f;
    mMedianTorsoX = mGlobalRatio * (mMaxTorsoX + mMinTorsoX) / 2.0f;
    mDiffTorsoX = mGlobalRatio * (mMaxTorsoX - mMinTorsoX) / 2.0f;
    mMedianShoulderZ = mGlobalRatio * (mMaxShoulderZ + mMinShoulderZ) / 2.0f;
    mDiffShoulderZ = mGlobalRatio * (mMaxShoulderZ - mMinShoulderZ) / 2.0f;
    mMedianTorsoXY = mGlobalRatio * (mMaxTorsoXY + mMinTorsoXY) / 2.0f;
    mDiffTorsoXY = mGlobalRatio * (mMaxTorsoXY - mMinTorsoXY) / 2.0f;

    mWorld = make_shared<World>();

#ifdef LOG_VERBOSE
	cout << "[VERBOSE] Environment::Environment" << endl;
#endif
}

void Environment::InitFromYaml(const string &path) {
#ifdef LOG_VERBOSE
    cout << "[VERBOSE] Environment::InitFromYaml" << endl;
#endif

    const auto abs_path = path_rel_to_abs(path);
    if (!filesystem::exists(abs_path)) {
        std::cerr << "Metadata file not found: " << abs_path << std::endl;
        return;
    }

    mCharacter = new Character();
    mDevice = new Device();

    try {
        YAML::Node root = YAML::LoadFile(abs_path);
        YAML::Node env = root["environment"];

        YAML::Emitter emitter;
        emitter << YAML::DoubleQuoted << YAML::Flow << root;
        emitter.SetIndent(2);
        emitter.SetMapFormat(YAML::Block);
        emitter.SetSeqFormat(YAML::Block);
        metadata = emitter.c_str();

        YAML::Node simulation = env["simulation"];
        mControlHz = simulation["control_hz"].as<int>();
        mSimulationHz = simulation["simulation_hz"].as<int>();
        if(simulation["device_hz"]) mDeviceHz = simulation["device_hz"].as<int>();
        mSelfCollision = simulation["self_collision"].as<bool>();

        if(env["contact"].IsDefined()){
            YAML::Node contact = env["contact"];
            mDebouncerAlpha = contact["debouncer_alpha"].as<double>();
        }
        
        YAML::Node state = env["state"];
        if (state["foot_clear"].IsDefined() && state["foot_clear"].as<bool>()) mStateType.add(StateType::foot_clear);
        if (state["sym_gait"].IsDefined() && state["sym_gait"].as<bool>()) mStateType.add(StateType::sym_gait);
        if (state["velocity"].IsDefined() && state["velocity"].as<bool>()) mStateType.add(StateType::velocity);
        if (state["dt_hist"].IsDefined() && state["dt_hist"].as<bool>()) mStateType.add(StateType::dt_hist);
        if (state["mass_ratio"].IsDefined() && state["mass_ratio"].as<bool>()) mStateType.add(StateType::mass_ratio);
        YAML::Node action = env["action"];
        mPdScale        = action["scale"].as<double>();
        mActionPhaseScale   = action["phase_scale"].as<double>();
        if (action["time_warp"].IsDefined() && action["time_warp"].as<bool>()) mActionType.add(ActionType::time_warp);
        if (action["upper"].IsDefined() && action["upper"].as<bool>()) mActionType.add(ActionType::upper);
        if (action["mirror_act"].IsDefined() && action["mirror_act"].as<bool>()) mActionType.add(ActionType::mirror_act);
        if (action["torque_clip"].IsDefined() && action["torque_clip"].as<bool>()) mActionType.add(ActionType::torque_clip);
        if (action["smooth"].IsDefined() && action["smooth"].as<double>() > 0) mActionAlpha = action["smooth"].as<double>();

        YAML::Node skeleton = env["skeleton"];
        mJointDamping = skeleton["damping"].as<double>();
        mJntType = static_cast<JntType>(skeleton["joint"].as<int>());
        const auto skel_file_path = skeleton["file"].as<string>(); _loadSkeletonFile(skel_file_path);
        const auto param_file_path = skeleton["parameter"].as<string>(); _loadSkeletonParam(param_file_path);

        if (skeleton["ankle"].IsDefined()) {
            mAnkleStiffness = skeleton["ankle"]["stiffness"].as<double>();
        }

        if (mJntType >= JntType::FullMuscle) {
            mUseMuscle = true;
            mCharacter->LoadMuscleYaml(skeleton["muscle"]);
            mMuscleGroups = mCharacter->GetMuscleGroupInCharacter();
        }

        YAML::Node device = env["device"];
        mUseDevice = device["enable"].as<bool>() || mForceUseDevice;
#ifdef LOG_VERBOSE
        cout << "[Environment] device.enable=" << device["enable"].as<bool>()
             << " force=" << mForceUseDevice
             << " => mUseDevice=" << mUseDevice << endl;
#endif
        if (mUseDevice) {
            mDevice->LoadYaml(device);
            const auto device_file_path = device["file"].as<string>(); _loadDeviceFile(device_file_path);
#ifdef LOG_VERBOSE
            cout << "[Environment] Device file: " << device_file_path << endl;
#endif
            const auto device_param_path = device["parameter"].as<string>(); _loadDeviceParam(device_param_path);
        }

        YAML::Node reward = env["reward"];
        mUseFootClear = reward["foot_clear"] ? reward["foot_clear"].as<bool>() : mUseFootClear;
        if (reward["device"].IsDefined()) {
            const auto device_reward = reward["device"];
            mDevRewardType       = device_reward["type"] ? device_reward["type"].as<int>() : mDevRewardType;
            mDevRewCoeff         = device_reward["coeff"] ? device_reward["coeff"].as<double>() : mDevRewCoeff;
            mDevRewMultiply      = device_reward["multiply"] ? device_reward["multiply"].as<bool>() : mDevRewMultiply;
            mDevPBias            = device_reward["bias_p"] ? device_reward["bias_p"].as<double>() : mDevPBias;
            mDevNBias            = device_reward["bias_n"] ? device_reward["bias_n"].as<double>() : mDevNBias;
            mDevPCoeff           = device_reward["coeff_p"] ? device_reward["coeff_p"].as<double>() : mDevPCoeff;
            mDevNCoeff           = device_reward["coeff_n"] ? device_reward["coeff_n"].as<double>() : mDevNCoeff;
        }
        if (reward["imitation"].IsDefined()) {
            const auto imitation = reward["imitation"];
            mImitationType = imitation["type"] ? imitation["type"].as<int>() : mImitationType;
            mImitationPosCoeff = imitation["pos_coeff"] ? imitation["pos_coeff"].as<double>() : mImitationPosCoeff;
            mImitationVelCoeff = imitation["vel_coeff"] ? imitation["vel_coeff"].as<double>() : mImitationVelCoeff;
        }

        YAML::Node sway = reward["sway"];
        mSwayType       = sway["type"].IsDefined() ? Utils::fromInt<SwayType>(sway["type"].as<int>()) : mSwayType;
        mSwayMarginRatio     = sway["margin_ratio"].IsDefined() ? sway["margin_ratio"].as<double>() : mSwayMarginRatio;
        mTiltRange = sway["tilt"].IsDefined() ? sway["tilt"].as<double>() : mTiltRange;
        mRotationRange = sway["rotation"].IsDefined() ? sway["rotation"].as<double>() : mRotationRange;
        mObliqueRange = sway["oblique"].IsDefined() ? sway["oblique"].as<double>() : mObliqueRange;
        mTiltRange = mTiltRange * M_PI / 180.0;
        mRotationRange = mRotationRange * M_PI / 180.0;
        mObliqueRange = mObliqueRange * M_PI / 180.0;

        mEnergy = make_unique<Energy>(mJntType);
        mEnergy->LoadYaml(reward["energy"]);
        YAML::Node velocity = reward["velocity"];
        mVelCoeff = velocity["coeff"].as<double>();
        mVelMargin = velocity["margin"].as<double>();
        YAML::Node head = reward["head"];
        mHeadMarginRatio = head["margin_ratio"].as<double>();

        YAML::Node motion = env["motion"];
        const auto cycle = motion["cycle"].as<bool>();
        const auto motion_file_path = motion["file"].as<string>(); mCharacter->LoadBVH(path_rel_to_abs(motion_file_path), cycle);
    }
    catch (const YAML::Exception& e) {
        std::cerr << "[Environment] Error processing metadata: " << e.what() << std::endl;
    }

    _initFinalize();
}

void Environment::Reset()
{
	mSimStep = 0;

	mWorld->reset();
	mWorld->getConstraintSolver()->setCollisionDetector(dart::collision::BulletCollisionDetector::create());
	mWorld->getConstraintSolver()->clearLastCollisionResult();
	_resetPhaseTime();
	mCharacter->Reset();
    mEnergy->Reset();
	tie(mTargetPositions, mTargetVelocities) = mCharacter->GetTargetPosAndVel(mLocalTime, mPhaseRatio / mControlHz);
    mBVHPositions = mTargetPositions;
    mBVHVelocities = mTargetVelocities;
	Eigen::VectorXd cur_pos = mTargetPositions;
	Eigen::VectorXd cur_vel = mTargetVelocities;

	double refVel = GetTargetVelocity();
	cur_vel.segment(3,3) = Eigen::Vector3d(0, 0, refVel);
	if(mActionType.has(ActionType::upper)){
		cur_pos.segment(mSkelChar->getJoint("ArmL")->getIndexInSkeleton(0), 3) = Eigen::Vector3d(0,0,-M_PI * 0.45);
		cur_pos.segment(mSkelChar->getJoint("ArmR")->getIndexInSkeleton(0), 3) = Eigen::Vector3d(0,0,M_PI * 0.45);
	}

	cur_pos[3] = 0.0;
	Utils::setSkelPosAndVel(mSkelChar, cur_pos, cur_vel);
	if (mIsRender) Utils::setSkelPosAndVel(mReferenceSkeleton, mBVHPositions, mBVHVelocities);
	mPdAction.setZero();

    mDesiredTorque = Eigen::VectorXd::Zero(mSkelDof);
    mSetDesiredTorque = false;

	mPrevCycleCOM = mSkelChar->getCOM();
	mCurCycleCOM = mSkelChar->getCOM();
	mCycleCount = 0;

	mCurCOM = mSkelChar->getCOM();
	mPrevCOM = mCurCOM;

    mFootPrevRz = 0.0;
    mFootPrevLz = 0.0;

	mPhaseDisplacement = 0;

	mComTrajectory.clear();
	_resetFoot();
	_resetContact();
    BodyNode* head = mSkelChar->getBodyNode("Head");
    mHeadPrevLinearVel = head->getCOMLinearVelocity();
    mHeadPrevAngularVel = head->getAngularVelocity();
    BodyNode* pelvis = mSkelChar->getBodyNode("Pelvis");
    mPelvisPrevLinearVel = pelvis->getCOMLinearVelocity();
    // mPelvisPrevAngularVel = pelvis->getAngularVelocity();

    if (mUseMuscle) _resetMuscle();
    if (mUseDevice) mDevice->Reset();
    if (mUseTerrain) mTerrainManager->reset(mSkelChar->getCOM());
}

void Environment::_updateDynamics(Record* pRecord, CBufferData* pGraphData)
{
    // moment
    if(mRECORD_MOMENT || pGraphData != nullptr){
        double weightInv = 1 / mCharacter->GetWeight();
        double hipR = mDesiredTorque.segment(6, 3).norm() * weightInv;
        double hipRx = mDesiredTorque[6] * weightInv;
        double kneeR = mDesiredTorque[9] * weightInv;
        double ankleR = mDesiredTorque.segment(10, 3).norm() * weightInv;
        double moment_tau = abs(hipR) + abs(kneeR) + abs(ankleR);
        if (pGraphData != nullptr){
            if (pGraphData->key_exists("moment_tau_hipR")) pGraphData->push("moment_tau_hipR", hipR);
            if (pGraphData->key_exists("moment_tau_hipRx")) pGraphData->push("moment_tau_hipRx", -hipRx);
            if (pGraphData->key_exists("moment_tau_kneeR")) pGraphData->push("moment_tau_kneeR", kneeR);
            if (pGraphData->key_exists("moment_tau_ankleR")) pGraphData->push("moment_tau_ankleR", ankleR);
        }

        if (mRECORD_MOMENT) {
            queue<pair<string, double>> record_buf;
            record_buf.emplace("meta_moment_tau", moment_tau);
            record_buf.emplace("meta_moment_tau_hipR", hipR);
            pRecord->add(mSimStep, record_buf);
        }
    }
    
    // power
    if (mRECORD_POWER || pGraphData != nullptr) {
        Eigen::VectorXd vel_char = mSkelChar->getVelocities();
        mPowerTau = mDesiredTorque.cwiseProduct(vel_char);
        double power_hip_r = mPowerTau.segment(6, 3).sum();
        double power_knee_r = mPowerTau[9];
        double power_ankle_r = mPowerTau.segment(10, 3).sum();
        const double power_tau = (power_hip_r > 0 ? power_hip_r : 0) + (power_knee_r > 0 ? power_knee_r : 0) + (power_ankle_r > 0 ? power_ankle_r : 0);
        if (pGraphData != nullptr){
            if (pGraphData->key_exists("power_tau")) pGraphData->push("power_tau", power_tau);
            if (pGraphData->key_exists("power_tau_hipR")) pGraphData->push("power_tau_hipR", power_hip_r);
            if (pGraphData->key_exists("power_tau_kneeR")) pGraphData->push("power_tau_kneeR", power_knee_r);
            if (pGraphData->key_exists("power_tau_ankleR")) pGraphData->push("power_tau_ankleR", power_ankle_r);        
        }

        if (mRECORD_POWER) {
            queue<pair<string, double>> record_buf;
            record_buf.emplace("meta_power", power_tau);
            record_buf.emplace("meta_power_HipR", power_hip_r > 0 ? power_hip_r : 0);
            // record_buf.emplace("power_KneeR", power_knee_r);
            // record_buf.emplace("power_AnkleR", power_ankle_r);
            pRecord->add(mSimStep, record_buf);
        }
    }

    updateMuscleTuple(pGraphData);
}

void Environment::updateMuscleTuple(CBufferData* pGraphData)
{
    if (!mUseMuscle) return;

    // compute muscle tuple
    const int muscle_size = mCharacter->GetMuscles().size();
    Eigen::MatrixXd JtA = Eigen::MatrixXd::Zero(mSkelDof, muscle_size);
    Eigen::VectorXd Jtp = Eigen::VectorXd::Zero(mSkelDof);

    Eigen::VectorXd tau_des = mDesiredTorque;

    auto start = chrono::high_resolution_clock::now();

    const bool mirror = ((mStateType.has(StateType::sym_gait) && GetPhase()>0.5) && mActionType.has(ActionType::mirror_act));
    int i = 0, index = 0;
    for (auto &muscle : mCharacter->GetMuscles())
    {
        Eigen::MatrixXd Jt_reduced;
        Eigen::VectorXd Fa, Fp;

        #ifdef LOG_VERBOSE
        if (i == 0) {
            const auto start = chrono::high_resolution_clock::now();
            muscle->UpdateGeometry();
            auto point1 = chrono::high_resolution_clock::now();
            static double dur1 = 0.0;
            dur1 = dur1 * 0.99 + chrono::duration_cast<chrono::nanoseconds>(point1 - start).count() * 0.01;
            cout << "      	  geometry update: " << dur1 << " ns" << endl;

            muscle->GetReducedJacobianTranspose(Jt_reduced);

            auto point2 = chrono::high_resolution_clock::now();
            static double dur2 = 0.0;
            dur2 = dur2 * 0.99 + chrono::duration_cast<chrono::nanoseconds>(point2 - point1).count() * 0.01;
            cout << "    	   muscle jacobian: " << dur2 << " ns" << endl;

            muscle->GetForceJacobianAndPassive(Fa, Fp);
            auto point3 = chrono::high_resolution_clock::now();
            static double dur3 = 0.0;
            dur3 = dur3 * 0.99 + chrono::duration_cast<chrono::nanoseconds>(point3 - point2).count() * 0.01;
            cout << "     	   muscle force: " << dur3 << " ns" << endl;

            Eigen::VectorXd JtA_reduced = Jt_reduced * Fa;
            Eigen::VectorXd JtP_reduced = Jt_reduced * Fp;
            for (int j=0; j< muscle->GetRelDofs(); j++)
            {
                int rel_idx = muscle->related_dof_indices[j];
                Jtp[rel_idx] += JtP_reduced[j];
                JtA(rel_idx, i) = JtA_reduced[j];
            }
            mMT.JtA_reduced.segment(index, JtA_reduced.rows()) = JtA_reduced;
            index += JtA_reduced.rows();
            i++;

            auto point4 = chrono::high_resolution_clock::now();
            static double dur4 = 0.0;
            dur4 = dur4 * 0.99 + chrono::duration_cast<chrono::nanoseconds>(point4 - point3).count() * 0.01;
            cout << "     	   block add: " << dur4 << " ns" << endl;
        } else {            
        #endif
            muscle->UpdateGeometry();
            muscle->GetReducedJacobianTranspose(Jt_reduced);
            muscle->GetForceJacobianAndPassive(Fa, Fp);
            Eigen::VectorXd JtA_reduced = Jt_reduced * Fa;
            Eigen::VectorXd JtP_reduced = Jt_reduced * Fp;
            for (int j=0; j< muscle->GetRelDofs(); j++)
            {
                int rel_idx = muscle->related_dof_indices[j];
                Jtp[rel_idx] += JtP_reduced[j];
                JtA(rel_idx, i) = JtA_reduced[j];
            }
            mMT.JtA_reduced.segment(index, JtA_reduced.rows()) = JtA_reduced;
            index += JtA_reduced.rows();
            i++;
        #ifdef LOG_VERBOSE
        }
        #endif
    }

    #ifdef LOG_VERBOSE
    auto point1 = chrono::high_resolution_clock::now();
    static double dur1 = 0.0;
    dur1 = dur1 * 0.99 + chrono::duration_cast<chrono::microseconds>(point1 - start).count() * 0.01;
    cout << "        muscle tuple: " << dur1 << " us" << endl;
    #endif
    if (mirror) {
        tau_des = mCharacter->GetMirrorPosition(tau_des);
        for(int j=0; j<muscle_size; j+=2)
        {
            Eigen::VectorXd tmp = JtA.col(j);
            JtA.col(j)   = mCharacter->GetMirrorPosition(JtA.col(j+1));
            JtA.col(j+1) = mCharacter->GetMirrorPosition(tmp);
        }
        int j = 0;
        int jndex = 0;
        for(auto& muscle : mCharacter->GetMuscles())
        {
            Eigen::VectorXd JtA_reduced = Eigen::VectorXd::Ones(muscle->GetRelDofs());
            for(int k=0; k< muscle->GetRelDofs(); k++) JtA_reduced[k] = JtA(muscle->related_dof_indices[k], j);

            mMT.JtA_reduced.segment(jndex, JtA_reduced.rows()) = JtA_reduced;
            jndex += JtA_reduced.rows();
            j++;
        }
        Jtp = mCharacter->GetMirrorPosition(Jtp);
    }
    #ifdef LOG_VERBOSE
    auto point2 = chrono::high_resolution_clock::now();
    static double dur2 = 0.0;
    dur2 = dur2 * 0.99 + chrono::duration_cast<chrono::microseconds>(point2 - point1).count() * 0.01;
    cout << "        mirror: " << dur2 << " us" << endl;
    #endif

    mMT.JtA = JtA.block(mRootDof, 0, mMcnDof, muscle_size);
    tau_des -= Jtp;
    mMT.tau_active = tau_des.segment(mRootDof, mMcnDof);
    mMT.JtP = Jtp.segment(mRootDof, mMcnDof);

    if (pGraphData != nullptr) {
        Eigen::VectorXd vel_char = mSkelChar->getVelocities();
        const double weightInv = 1.0 / mSkelChar->getMass();
        const auto muscle_moment = (JtA * mActivations + Jtp) * weightInv;
        mPowerMuscle = muscle_moment.cwiseProduct(vel_char);
        const double power_muscle_hipR = mPowerMuscle.segment(6, 3).sum();
        const double power_muscle_hipRx = mPowerMuscle[6];
        const double power_muscle_kneeR = mPowerMuscle.segment(9, 1).sum();
        const double power_muscle_ankleR = mPowerMuscle.segment(10, 3).sum();
        const double power_muscle = (power_muscle_hipR > 0 ? power_muscle_hipR : 0) + (power_muscle_kneeR > 0 ? power_muscle_kneeR : 0) + (power_muscle_ankleR > 0 ? power_muscle_ankleR : 0);
        
        const double moment_muscle_hipR = muscle_moment.segment(6, 3).norm();
        const double moment_muscle_hipRx = muscle_moment[6];
        const double moment_muscle_hipRy = muscle_moment[7];
        const double moment_muscle_hipRz = muscle_moment[8];
        const double moment_muscle_kneeR = muscle_moment[9];
        const double moment_muscle_ankleRx = -muscle_moment[10];
        // if (pGraphData->key_exists("moment_muscle_hipR")) pGraphData->push("moment_muscle_hipR", moment_muscle_hipR);
        if (pGraphData->key_exists("moment_muscle_hipRx_ef")) pGraphData->push("moment_muscle_hipRx_ef", -moment_muscle_hipRx);
        // if (pGraphData->key_exists("moment_muscle_hipRy_ie")) pGraphData->push("moment_muscle_hipRy_ie", moment_muscle_hipRy);
        // if (pGraphData->key_exists("moment_muscle_hipRz_aa")) pGraphData->push("moment_muscle_hipRz_aa", moment_muscle_hipRz);
        if (pGraphData->key_exists("moment_muscle_kneeR")) pGraphData->push("moment_muscle_kneeR", moment_muscle_kneeR);
        if (pGraphData->key_exists("moment_muscle_ankleRx")) pGraphData->push("moment_muscle_ankleRx", moment_muscle_ankleRx);
        
        if (pGraphData->key_exists("power_muscle")) pGraphData->push("power_muscle", power_muscle);
        if (pGraphData->key_exists("power_muscle_hipR")) pGraphData->push("power_muscle_hipR", power_muscle_hipR);
        if (pGraphData->key_exists("power_muscle_hipRx_ef")) pGraphData->push("power_muscle_hipRx_ef", power_muscle_hipRx);
        if (pGraphData->key_exists("power_muscle_kneeR")) pGraphData->push("power_muscle_kneeR", power_muscle_kneeR);
        if (pGraphData->key_exists("power_muscle_ankleR")) pGraphData->push("power_muscle_ankleR", power_muscle_ankleR);
    }
}

void Environment::Step(Record* pRecord, CBufferData* pGraphData, bool actuate_character, bool update_device_torque)
{
    #ifdef LOG_VERBOSE
    cout << "====================== step start ======================" << endl;
    auto start = chrono::high_resolution_clock::now();
    #endif

    mPrevCOM = mCurCOM;

    mIsGaitCycleComplete = false;
    if(mUseMuscle && update_device_torque && mDevice->GetAngleType() == 3) {
        cout << "super device is not supported from 53efd6bc67ea4e0a31edf648ae761795623c2c02" << endl;
        exit(-1);
        if (mJntType == JntType::LowerMuscle) cout << "since setforce is called twice in the ::step, you must investigate the simulation is working properly" << endl;
        const Eigen::VectorXd super_torque = mDevice->SuperTorque(mDesiredTorque);
        mSkelChar->setForces(super_torque);
        mDesiredTorque -= super_torque;
    }else mDevice->Step(update_device_torque);

    #ifdef LOG_VERBOSE
    auto point1 = chrono::high_resolution_clock::now();
    static double dur1 = 0.0;
    dur1 = dur1 * 0.99 + chrono::duration_cast<chrono::microseconds>(point1 - start).count() * 0.01;
    cout << "    device step: " << dur1 << " us" << endl;
    #endif

    if (mAnkleStiffness > 0) {
        Eigen::VectorXd ext_torque = mSkelChar->getForces();
        const double ankle_angleR = mSkelChar->getPositions()[10];
        const double ankle_angleL = mSkelChar->getPositions()[22];
        double ankle_torqueR = 0.0, ankle_torqueL = 0.0;
        if (ankle_angleR < 0.0) ankle_torqueR = -mAnkleStiffness * ankle_angleR;
        if (ankle_angleL < 0.0) ankle_torqueL = -mAnkleStiffness * ankle_angleL;
        ext_torque[10] += ankle_torqueR;
        ext_torque[22] += ankle_torqueL;
        mSkelChar->setForces(ext_torque);
    }

    mDesiredTorque = _compute_desired_torque();
    mDesiredTorqueHistory.push_back(mDesiredTorque);
    if (mDesiredTorqueHistory.size() > mDtHistSize) mDesiredTorqueHistory.pop_front();

    Eigen::VectorXd ext_torque = mSkelChar->getForces();
    if (actuate_character) {
        if (mUseMuscle) mCharacter->setActivation(mActivations);
        if (mJntType == JntType::Torque) {
            ext_torque += mDesiredTorque;
        }else if (mJntType == JntType::LowerMuscle) {
            Eigen::VectorXd upper_torque = mDesiredTorque;
            upper_torque.head(mRootDof + mMcnDof).setZero();
            ext_torque += upper_torque;
        }
    }
    mSkelChar->setForces(ext_torque);

    #ifdef LOG_VERBOSE
    auto point2 = chrono::high_resolution_clock::now();
    static double dur2 = 0.0;
    dur2 = dur2 * 0.99 + chrono::duration_cast<chrono::microseconds>(point2 - point1).count() * 0.01;
    cout << "    muscle step: " << dur2 << " us" << endl;
    #endif

    // if(mUseMuscle && update_device_torque && mDevice->GetAngleType() == 3) mDesiredTorque -= mDevice->SuperTorque(mDesiredTorque);

	mWorld->step();

    #ifdef LOG_VERBOSE
    auto point3 = chrono::high_resolution_clock::now();
    static double dur3 = 0.0;
    dur3 = dur3 * 0.99 + chrono::duration_cast<chrono::microseconds>(point3 - point2).count() * 0.01;
    cout << "    dynamics step: " << dur3 << " us" << endl;
    #endif

    _updateDynamics(pRecord, pGraphData);

    #ifdef LOG_VERBOSE
    auto point4 = chrono::high_resolution_clock::now();
    static double dur4 = 0.0;
    dur4 = dur4 * 0.99 + chrono::duration_cast<chrono::microseconds>(point4 - point3).count() * 0.01;
    cout << "    update dynamics: " << dur4 << " us" << endl;
    #endif
    _updateCom(pRecord, pGraphData);
    _updateMetabolic(pRecord, pGraphData); // must be called after _updateCom due to the use of mCurCOM
    _updateFoot(pRecord, pGraphData); // must be called after _updateCom and _updateMetabolic due to the calling of mEnergy->HeelStrikeCb() must be called after AccumActivation()
    _updateKinematics(pRecord, pGraphData);
    _updateRefChar(pGraphData);
    _updateDevice(pRecord, pGraphData);
    _updateActivation(pRecord, pGraphData);

    #ifdef LOG_VERBOSE
    auto point5 = chrono::high_resolution_clock::now();
    static double dur5 = 0.0;
    dur5 = dur5 * 0.99 + chrono::duration_cast<chrono::microseconds>(point5 - point4).count() * 0.01;
    cout << "    update other: " << dur5 << " us" << endl;
    #endif

	mSimStep++;

    if (mUseTerrain) mTerrainManager->update(mSkelChar->getCOM());

    #ifdef LOG_VERBOSE
    auto point6 = chrono::high_resolution_clock::now();
    static double dur6 = 0.0;
    dur6 = dur6 * 0.99 + chrono::duration_cast<chrono::microseconds>(point6 - start).count() * 0.01;
    cout << "    total: " << dur6 << " us" << endl;
    #endif
}

int Environment::IsEndOfEpisode() const
{
    const double root_y = mSkelChar->getCOM()[1];
    bool isFallDown = (root_y < 0.7 * mGlobalRatio);
    bool hasInvalidValues = (dart::math::isNan(mSkelChar->getPositions()) || dart::math::isNan(mSkelChar->getVelocities()));
    if (hasInvalidValues) cout << "[ERROR] Nan detected in IsEndOfEpisode" << endl;
    if (isFallDown || hasInvalidValues) return 1; // terminated
    else if (mSimStep >= 20000) return 2; // truncated
    return 0;
}


Eigen::VectorXd Environment::GetMuscleState(bool mirror)
{
	MuscleTuple mt = GetMuscleTuple();
	Eigen::VectorXd passive_f = mt.JtP;
	Eigen::VectorXd muscleState;

    muscleState = Eigen::VectorXd::Zero(passive_f.size() * 3);
    Eigen::VectorXd min_tau = Eigen::VectorXd::Zero(passive_f.size());
    Eigen::VectorXd max_tau = Eigen::VectorXd::Zero(passive_f.size());
    int muscleSize = mCharacter->GetMuscles().size();
    for(int i = 0; i < passive_f.size(); i++){
        for(int j = 0; j < muscleSize; j++){
            if(mt.JtA(i, j) > 0) max_tau[i] += mt.JtA(i, j);
            else min_tau[i] += mt.JtA(i, j);
        }
    }
    muscleState << passive_f, 0.5 * min_tau, 0.5 * max_tau;

	return muscleState;
}

Eigen::VectorXf Environment::GetState()
{
    Eigen::VectorXd char_state = GetStateChar();
	mNumCharState = char_state.rows();
    Eigen::VectorXd state(mNumCharState + mDeviceStateNum);
	if(mUseDevice && mDevice->IsUseState()){
        state << char_state, GetStateExo();
	}else state << char_state, Eigen::VectorXd::Zero(mDeviceStateNum);
    return state.cast<float>();
}

Eigen::VectorXd Environment::GetStateExo(){
    bool mirror = (mStateType.has(StateType::sym_gait) && GetPhase()>0.5);
    const auto state = mDevice->GetState(mirror);
    return state;
}

Eigen::VectorXd Environment::GetStateChar()
{
    const bool mirror = ((mStateType.has(StateType::sym_gait) && GetPhase()>0.5));

	Eigen::VectorXd state;
    const Eigen::VectorXd skel_com_vel = mSkelChar->getCOMLinearVelocity();
    const Eigen::VectorXd skel_com_pos = mSkelChar->getCOM();
    Eigen::Vector3d com_pos = skel_com_pos - Eigen::Vector3d(0.0, -0.98, 0.0);

    int num_body_nodes = mSkelChar->getNumBodyNodes();
	Eigen::VectorXd p, v;
	p.resize(num_body_nodes * (3 + 6));
	v.resize((num_body_nodes + 1) * 3 + num_body_nodes * 3);

    if (mirror) {
        int jdx = 0;
        std::vector<Eigen::Matrix3d> body_node_transforms = mCharacter->getBodyNodeTransform();
        for (auto j_pair : mCharacter->getPairs())
        {
            const auto& first_bn = j_pair.second->getChildBodyNode();
            const auto& second_bn = j_pair.first->getChildBodyNode();
            int first_idx = first_bn->getIndexInSkeleton();
            int second_idx = second_bn->getIndexInSkeleton();

            Eigen::Vector3d first_pos = first_bn->getCOM() - com_pos;
            first_pos[0] *= -1;
            Eigen::Vector3d second_pos = second_bn->getCOM() - com_pos;
            second_pos[0] *= -1;

            Eigen::AngleAxisd first_rot = Eigen::AngleAxisd(first_bn->getTransform().linear());
            first_rot.axis() = Eigen::Vector3d(first_rot.axis()[0], -first_rot.axis()[1], -first_rot.axis()[2]);

            Eigen::AngleAxisd second_rot = Eigen::AngleAxisd(second_bn->getTransform().linear());
            second_rot.axis() = Eigen::Vector3d(second_rot.axis()[0], -second_rot.axis()[1], -second_rot.axis()[2]);

            Eigen::Matrix3d first_rot_mat = first_rot.toRotationMatrix() * body_node_transforms[jdx].transpose();
            Eigen::Matrix3d second_rot_mat = second_rot.toRotationMatrix() * body_node_transforms[jdx];

            p.segment<3>(first_idx * 3) = first_pos;
            p.segment<3>(second_idx * 3) = second_pos;

            p.segment<6>(num_body_nodes * 3 + first_idx * 6) << first_rot_mat(0, 0), first_rot_mat(0, 1), first_rot_mat(0, 2), first_rot_mat(1, 0), first_rot_mat(1, 1), first_rot_mat(1, 2);
            p.segment<6>(num_body_nodes * 3 + second_idx * 6) << second_rot_mat(0, 0), second_rot_mat(0, 1), second_rot_mat(0, 2), second_rot_mat(1, 0), second_rot_mat(1, 1), second_rot_mat(1, 2);

            Eigen::Vector3d first_vel = first_bn->getCOMLinearVelocity() - skel_com_vel;
            first_vel[0] *= -1;

            Eigen::Vector3d second_vel = second_bn->getCOMLinearVelocity() - skel_com_vel;
            second_vel[0] *= -1;

            v.segment<3>(first_idx * 3) = first_vel;
            v.segment<3>(second_idx * 3) = second_vel;

            Eigen::Vector3d first_ang = 0.1 * first_bn->getAngularVelocity();
            first_ang[1] *= -1;
            first_ang[2] *= -1;
            v.segment<3>((num_body_nodes + 1) * 3 + first_idx * 3) = first_ang;

            Eigen::Vector3d second_ang = 0.1 * second_bn->getAngularVelocity();
            second_ang[1] *= -1;
            second_ang[2] *= -1;
            v.segment<3>((num_body_nodes + 1) * 3 + second_idx * 3) = second_ang;
            jdx++;
        }
        v.segment<3>(num_body_nodes * 3) = skel_com_vel;
        v.segment<3>(num_body_nodes * 3)[0] *= -1;

    } else {
        for (int i=0; i<num_body_nodes; i++){
            const auto& bn = mSkelChar->getBodyNode(i);
            const auto& R = bn->getTransform().linear();
            p.segment<3>(3 * i) = bn->getCOM() - com_pos;
            p.segment<6>(num_body_nodes * 3 + i * 6).head<3>() = R.row(0);
            p.segment<6>(num_body_nodes * 3 + i * 6).tail<3>() = R.row(1);

            v.segment<3>(3 * i) = bn->getCOMLinearVelocity() - skel_com_vel;
            v.segment<3>(3 * i + 3 * (num_body_nodes + 1)) = 0.1 * bn->getAngularVelocity();
        }
        v.segment<3>(num_body_nodes * 3) = skel_com_vel;
    }

    Eigen::VectorXd nextTargetFoot = mNextTargetFoot - mSkelChar->getRootBodyNode()->getCOM();
    if (mirror) nextTargetFoot[0] *= -1;

	Eigen::VectorXd muscleState;
	if(mUseMuscle) muscleState = 0.008 * GetMuscleState();

    Eigen::VectorXd skel_param = Eigen::VectorXd::Zero(mSkelLengthParams.size() + mSkelMassParams.size());
    for(int i = 0; i < mSkelLengthParams.size(); i++) skel_param[i] = mSkelLengthParams[i].current_ratio;
    for(int i = 0; i < mSkelMassParams.size(); i++) skel_param[i + mSkelLengthParams.size()] = mSkelMassParams[i].current_ratio;

    const int static_size = 5 + 3 + skel_param.rows() + p.rows() + v.rows();
    int dynamic_size = 0;
    if (mUseMuscle) dynamic_size += (mNumMuscleState + mMuscleLengthParams.size() + mMuscleForceParams.size());
    if (mActionType.has(ActionType::time_warp)) dynamic_size += 1;
    if (mStateType.has(StateType::foot_clear)) dynamic_size += 2;
    if (mStateType.has(StateType::dt_hist)) dynamic_size += mPdDof * mDtHistSize;
    if (mStateType.has(StateType::mass_ratio)) dynamic_size += 1;
    const int total_size = static_size + dynamic_size;

    state = Eigen::VectorXd::Zero(total_size);
	state <<
        com_pos[1], GetPhase(), mGlobalRatio, mPhaseRatio, GetTargetVelocity(),
        nextTargetFoot, skel_param,
        p, v,
        Eigen::VectorXd::Zero(dynamic_size);

	int idx = static_size;

	if(mUseMuscle){
		state.segment(idx, muscleState.rows()) = muscleState;
		idx += muscleState.rows();
        for(auto ml : mMuscleLengthParams) state[idx++] = ml.current_ratio;
        for(auto mf : mMuscleForceParams) state[idx++] = mf.current_ratio;
	}

    if(mActionType.has(ActionType::time_warp)) {
        double phase = GetGlobalPhase();
        if (mirror) phase = fmod(phase + 0.5, 1.0);
        state[idx++] = phase;
    }

	if(mStateType.has(StateType::foot_clear)){
        if (mirror) {
            state[idx++] = mContactPhaseL; state[idx++] = mContactPhaseR;
        }else {
            state[idx++] = mContactPhaseR; state[idx++] = mContactPhaseL;
        }
	}

    if (mStateType.has(StateType::dt_hist)) {
        for (int i = 0; i < mDtHistSize; i++) {
            if (mirror) {
                Eigen::VectorXd desired_torque = mCharacter->GetMirrorPosition(mDesiredTorqueHistory[i]);
                state.segment(idx, mPdDof) = desired_torque.segment(mRootDof, mPdDof);
            }else {
                state.segment(idx, mPdDof) = mDesiredTorqueHistory[i].segment(mRootDof, mPdDof);
            }
            idx += mPdDof;
        }
    }

    if (mStateType.has(StateType::mass_ratio)) {
        state[idx++] = mMassRatio;
    }

	if(idx != state.rows()){
		cout << "[Warning] GetState " << endl;
		exit(-1);
	}
	return state;
}

unordered_map<string, float> Environment::GetParam() const {
    unordered_map<string, float> ps = {
            {"reward_loco", mWeightLoco},
            {"reward_head", mHeadExpCoeff},
            {"reward_sway", mSwayCoeff},
            {"reward_metabolic", mWeightMeta},
            {"reward_velocity_coeff", mVelCoeff},
            {"reward_imitation", mWeightImitation},
            {"reward_imitation_pos_coeff", mImitationPosCoeff},
            {"reward_imitation_vel_coeff", mImitationVelCoeff},
            {"stride", mStrideRatio},
            {"phase", mPhaseRatio},
            // {"length", mGlobalRatio},
            {"mass", mMassRatio},
            {"muscle_force", mMuscleForceRatio},
    };
    for (const auto& m: mMuscleForceParams) ps[m.name] = m.current_ratio;
    for (const auto& m: mMuscleLengthParams) ps[m.name] = m.current_ratio;

    if (mUseDevice) {
        ps["k"] = mDeviceK;
        ps["delay"] = mDeviceDelay;
        ps["reward_device"] = mWeightDev;
        ps["reward_device_multiply"] = static_cast<float>(mDevRewMultiply);
        ps["reward_device_type"] = static_cast<float>(mDevRewardType);
        ps["reward_device_coeff"] = mDevRewCoeff;
        ps["reward_velocity_coeff"] = mVelCoeff;
        ps["device_type"] = static_cast<float>(mDevice->GetAngleType());
        ps["device_weight_scaler"] = mDevice->GetWeightScaler();
        ps["device_force_weight"] = static_cast<float>(mDevice->IsWeightForced());
        ps["device_enable_weight"] = static_cast<float>(mDevice->IsWeightEnabled());
        ps["device_zero_state"] = static_cast<float>(mDevice->IsZeroState());
        ps["device_virtual_coupling"] = static_cast<float>(mDevice->IsVirtualCoupling());
        ps["device_coeff_p"] = mDevPCoeff;
        ps["device_coeff_n"] = mDevNCoeff;
        ps["device_bias_p"] = mDevPBias;
        ps["device_bias_n"] = mDevNBias;
    }
    return ps;
}

// TODO: comparison process has too much overhead
void Environment::SetParam(const std::unordered_map<std::string, float>& params, bool allow_duplicated_param)
{
    bool device_modified = false;
	bool muscle_force_modified = false;
	bool muscle_length_modified = false;

    for (const auto &[key, value]: params){
        if (Utils::case_insensitive_compare(key, "reward_loco")) {
            mWeightLoco = value;
        }else if (Utils::case_insensitive_starts_with(key, "param_idx")) {
            continue;
        }else if(Utils::case_insensitive_compare(key, "reward_head")) {
            mHeadExpCoeff = value;
        }else if(Utils::case_insensitive_compare(key, "reward_sway")) {
            mSwayCoeff = value;
        }else if(Utils::case_insensitive_compare(key, "reward_metabolic")) {
            mWeightMeta = value;
        }else if(Utils::case_insensitive_compare(key, "reward_velocity_coeff")) {
            mVelCoeff = value;
        }else if(Utils::case_insensitive_compare(key, "reward_imitation")) {
            mWeightImitation = value;
        }else if(Utils::case_insensitive_compare(key, "reward_imitation_pos_coeff")) {
            mImitationPosCoeff = value;
        }else if(Utils::case_insensitive_compare(key, "reward_imitation_vel_coeff")) {
            mImitationVelCoeff = value;
        }else if(Utils::case_insensitive_compare(key, "reward_device")) {
            if (mUseDevice) mWeightDev = value;
            else cout << "[Warning] SetParam: unused key (" << key << ")" << endl;        
        }else if(Utils::case_insensitive_compare(key, "reward_device_coeff")) {
            if (mUseDevice) mDevRewCoeff = value;
            else cout << "[Warning] SetParam: unused key (" << key << ")" << endl;
        }else if(Utils::case_insensitive_compare(key, "device_coeff_p")) {
            if (mUseDevice) mDevPCoeff = value;
            else cout << "[Warning] SetParam: unused key (" << key << ")" << endl;
        }else if(Utils::case_insensitive_compare(key, "device_coeff_n")) {
            if (mUseDevice) mDevNCoeff = value;
            else cout << "[Warning] SetParam: unused key (" << key << ")" << endl;
        }else if(Utils::case_insensitive_compare(key, "device_bias_p")) {
            if (mUseDevice) mDevPBias = value;
            else cout << "[Warning] SetParam: unused key (" << key << ")" << endl;
        }else if(Utils::case_insensitive_compare(key, "device_bias_n")) {
            if (mUseDevice) mDevNBias = value;
            else cout << "[Warning] SetParam: unused key (" << key << ")" << endl;
        }else if(Utils::case_insensitive_compare(key, "reward_device_type")) {
            if (mUseDevice) mDevRewardType = static_cast<int>(value);
            else cout << "[Warning] SetParam: unused key (" << key << ")" << endl;
        }else if(Utils::case_insensitive_compare(key, "reward_device_multiply")) {
            if (mUseDevice) mDevRewMultiply = static_cast<bool>(value);
            else cout << "[Warning] SetParam: unused key (" << key << ")" << endl;
        }else if(Utils::case_insensitive_compare(key, "stride")) {
            mStrideRatio = value;
        }else if(Utils::case_insensitive_compare(key, "phase")) {
            mPhaseRatio = value;
        // }else if(Utils::case_insensitive_compare(key, "length")) {
            // SetSkelLength(value);
        }else if(Utils::case_insensitive_compare(key, "mass")) {
            SetSkelMass(value);
        }else if(Utils::case_insensitive_compare(key, "muscle_force")) {
        	if (muscle_force_modified && !allow_duplicated_param)
        	{
        		cout << "[Warning] Multiple muscle force parameters are modified. Ignoring the " << key << endl;
        		continue;
        	}
        	muscle_force_modified = true;
            mMuscleForceRatio = value;
        	for (const auto& m: mCharacter->GetMuscles()) m->change_f(value);
        }else if(Utils::case_insensitive_compare(key, "shortening_multiplier")) {
            if(mUseMuscle) {
                SetShorteningMultiplier(value);
            } else {
                cout << "[Warning] SetParam: unused key (" << key << ") - muscles not enabled" << endl;
            }
        }else if(Utils::case_insensitive_compare(key, "device_virtual_coupling")) {
            if(mUseDevice){
                mDevice->SetVirtualCoupling(value);
            }else cout << "[Warning] SetParam: unused key (" << key << ")" << endl;
        }else if(Utils::case_insensitive_compare(key, "k")) {
            if(mUseDevice){
                mDeviceK = value;
                device_modified = true;
            }else cout << "[Warning] SetParam: unused key (" << key << ")" << endl;
        }else if(Utils::case_insensitive_compare(key, "delay")){
            if(mUseDevice){
                mDeviceDelay = value;
                device_modified = true;
            }else cout << "[Warning] SetParam: unused key (" << key << ")" << endl;
        }else if(Utils::case_insensitive_compare(key, "device_zero_state")){
            if(mUseDevice){
                mDevice->SetZeroState(static_cast<bool>(value));
            }else cout << "[Warning] SetParam: unused key (" << key << ")" << endl;
        }else if(Utils::case_insensitive_compare(key, "device_virtual_coupling")){
            if(mUseDevice){
                mDevice->SetVirtualCoupling(static_cast<bool>(value));
            }else cout << "[Warning] SetParam: unused key (" << key << ")" << endl;
        }else if(Utils::case_insensitive_compare(key, "device_enable_weight")){
            if(mUseDevice){
                mDevice->SetWeightEnabled(static_cast<bool>(value));
            }else cout << "[Warning] SetParam: unused key (" << key << ")" << endl;
        }else if(Utils::case_insensitive_compare(key, "device_force_weight")){
            if(mUseDevice){
                mDevice->SetWeightForced(static_cast<bool>(value));
            }else cout << "[Warning] SetParam: unused key (" << key << ")" << endl;
        }else if(Utils::case_insensitive_compare(key, "device_weight_scaler")){
            if(mUseDevice){
                mDevice->SetWeightScaler(value);
            }else cout << "[Warning] SetParam: unused key (" << key << ")" << endl;
        }else if(Utils::case_insensitive_compare(key, "device_type")){
            if(mUseDevice) mDevice->SetAngleType(static_cast<int>(value));
            else cout << "[Warning] SetParam: unused key (" << key << ")" << endl;
        }else if (Utils::case_insensitive_compare(key, "device_virtual_coupling")){
            if(mUseDevice){
                mDevice->SetVirtualCoupling(static_cast<bool>(value));
            }else cout << "[Warning] SetParam: unused key (" << key << ")" << endl;
        }else if (Utils::case_insensitive_starts_with(key, "muscle_force_")) {
        	if (muscle_force_modified && !allow_duplicated_param)
        	{
        		cout << "[Warning] Multiple muscle force parameters are modified. Ignoring the " << key << endl;
        		continue;
        	}
        	muscle_force_modified = true;

            for (auto& mp: mMuscleForceParams) {
                if (Utils::case_insensitive_compare(key, mp.name)) {
                    mp.current_ratio = value > mp.min_ratio ? value : mp.min_ratio;
                    for (const auto& m: mp.muscle) m->change_f(mp.current_ratio);
                    break;
                }
            }
        }else if (Utils::case_insensitive_starts_with(key, "muscle_length_")) {
        	if (muscle_length_modified && !allow_duplicated_param)
        	{
        		cout << "[Warning] Multiple muscle length parameters are modified. Ignoring the " << key << endl;
        		continue;
        	}
        	muscle_length_modified = true;

            for (auto& mp: mMuscleLengthParams) {
                if (Utils::case_insensitive_compare(key, mp.name)) {
                    mp.current_ratio = value > mp.min_ratio ? value : mp.min_ratio;
                    for (const auto& m: mp.muscle) m->change_l(mp.current_ratio);
                    break;
                }
            }
        }else cout << "[Warning] SetParam: unknown key (" << key << ")" << endl;
    }
    if (device_modified) mDevice->SetParam(mDeviceK, mDeviceK, mDeviceDelay);
}

void Environment::SetSkelMass(double ratio){
    mMassRatio = ratio;
    double l_ratio = 1.0 + (mGlobalRatio - 1.0) * 0.3;
    for (size_t i = 0; i < mSkelInfos.size(); ++i)
    {
        if (std::get<0>(mSkelInfos[i]) != "Head") {
            auto& modified_info = std::get<1>(mSkelInfos[i]);
            const auto& ref_info = std::get<1>(mSkelInfos_ref[i]);

            modified_info.value[1] = ref_info.value[1] * l_ratio;
            modified_info.value[2] = ref_info.value[2] * l_ratio;
            modified_info.value[5] = ref_info.value[5] * mMassRatio;
        }
    }
    mCharacter->ModifySkeletonLengthAndMass(mSkelInfos, mMassRatio);
    if(mUseDevice){
        for(size_t i = 0; i < mSkelDeviceInfos.size(); ++i)
        {
            auto& modified_info = std::get<1>(mSkelDeviceInfos[i]);
            const auto& ref_info = std::get<1>(mSkelDeviceInfos_ref[i]);

            modified_info.value[1] = ref_info.value[1] * l_ratio;
            modified_info.value[2] = ref_info.value[2] * l_ratio;
            modified_info.value[5] = ref_info.value[5] * mMassRatio;
        }
        mDevice->RemoveConstraint();
        mDevice->SetOffsetModify(mCharacter->GetOffsetModify());
        mDevice->ModifySkeletonLength(mSkelDeviceInfos, true);
        mDevice->AddConstraint();
    }
}

void Environment::SetShorteningMultiplier(double multiplier) {
    if (!mUseMuscle) return;
    for (auto muscle : mCharacter->GetMuscles()) {
        muscle->SetShorteningMultiplier(multiplier);
    }
}

double Environment::GetShorteningMultiplier() const {
    if (!mUseMuscle || mCharacter->GetMuscles().empty()) return 1.0;
    return mCharacter->GetMuscles()[0]->GetShorteningMultiplier();
}

void Environment::SetSkelLength(double ratio) {
    mGlobalRatio = ratio;
    mMassRatio = mGlobalRatio * (1.0 + (mGlobalRatio - 1.0) * 0.2) * (1.0 + (mGlobalRatio - 1.0) * 0.2);
    double l_ratio = 1.0 + (mGlobalRatio - 1.0) * 0.2;
    for (auto &s_info: mSkelInfos) {
        if (get<0>(s_info) != "Head") {
            get<1>(s_info).value[0] = mGlobalRatio; // lx
            get<1>(s_info).value[1] = l_ratio; // ly
            get<1>(s_info).value[2] = l_ratio; // lz
            get<1>(s_info).value[5] = mMassRatio; // m
        }
    }
    mCharacter->ModifySkeletonLengthAndMass(mSkelInfos, mMassRatio);
    if(mUseDevice){
        for(auto& s_info : mSkelDeviceInfos)
        {
            get<1>(s_info).value[0] = mGlobalRatio;
            get<1>(s_info).value[1] = l_ratio;
            get<1>(s_info).value[2] = l_ratio;
        }
        mDevice->RemoveConstraint();
        mDevice->SetOffsetModify(mCharacter->GetOffsetModify());
        mDevice->ModifySkeletonLength(mSkelDeviceInfos, true);
        mDevice->AddConstraint();
    }
}

void Environment::SetActivationLevels(const Eigen::VectorXd &a)
{
    bool mirror = (mStateType.has(StateType::sym_gait) && GetPhase()>0.5);
    if(mirror)
    {
        for(int i = 0; i < a.rows(); i+= 2)
        {
            mActivations[i] = a[i + 1];
            mActivations[i + 1] = a[i];
        }
    }
    else mActivations = a;
}

void Environment::SetAction(const Eigen::VectorXd &action, CBufferData* pGraphData)
{
    mPdAction = action.head(mPdDof) * mPdScale;
	if(!mActionType.has(ActionType::upper)) mPdAction.tail(mPdDof - 30).setZero();

    bool mirror = (mStateType.has(StateType::sym_gait) && GetPhase()>0.5);
	if (mirror) mPdAction = _mirrorAction(mPdAction);

    mPhaseDisplacement = 0.0;
	if(mActionType.has(ActionType::time_warp)) mPhaseDisplacement = action[mPdDof] * mActionPhaseScale;
	if(mPhaseDisplacement < -(1.0/ mControlHz)) mPhaseDisplacement = -(1.0 / mControlHz);

    _updatePhase(pGraphData);
}

void Environment::_updatePhase(CBufferData* pGraphData){
	mLocalTime += ((mPhaseDisplacement + (1.0 / mControlHz)) * mPhaseRatio);
	mGlobalTime += (1.0 / mControlHz) * mPhaseRatio;

	double current_gp = GetGlobalPhase();
	int current_gs = mGlobalTime / mMaxTime;
	if(mActionType.has(ActionType::time_warp) && !mStateType.has(StateType::velocity))
	{
		double localtime_min = (current_gs + (current_gp < 0.5 ? 0 : 0.5)) * mMaxTime;
		double localtime_max = (current_gs + (current_gp < 0.5 ? 0.5 : 1.0)) * mMaxTime;
		if (mLocalTime < localtime_min + 1E-6) mLocalTime = localtime_min + 1E-6;
		if (mLocalTime > localtime_max - 1E-6) mLocalTime = localtime_max - 1E-6;
	}

    if(pGraphData != nullptr){
        if(pGraphData->key_exists("phase_displacement")) pGraphData->push("phase_displacement", mPhaseDisplacement);
        if(pGraphData->key_exists("phase_lTime")) pGraphData->push("phase_lTime", mLocalTime);
    }

    tie(mTargetPositions, mTargetVelocities) = mCharacter->GetTargetPosAndVel(mLocalTime, mPhaseRatio / mControlHz);
    tie(mBVHPositions, mBVHVelocities) = mCharacter->GetTargetPosAndVel(mGlobalTime, mPhaseRatio / mControlHz);
}

// double Environment::_smoothDtReward(CBufferData* pGraphData) {
//     if (!mSmoothDt) return 1.0;
//     // compute the smoothness of the desired torque history
//     double smoothness = 0.0;
//     for (int i = 1; i < mDtHistSize; i++) {
//         const Eigen::VectorXd ddt = mDesiredTorqueHistory[i] - mDesiredTorqueHistory[i-1];
//         smoothness += ddt.squaredNorm();
//     }
//     smoothness /= (mDtHistSize - 1);
//     const double reward = exp(-mSmoothDtCoeff * smoothness);

//     if (pGraphData != nullptr) {
//         if (pGraphData->key_exists("smooth_dt")) pGraphData->push("smooth_dt", smoothness);
//     }

//     return reward;
// }

double Environment::_imitationReward(CBufferData* pGraphData) {
    if (mImitationType < 0) return 0.0;

    double w_p = 0.0, w_v = 0.0;
    Eigen::VectorXd pos = mSkelChar->getPositions();
    Eigen::VectorXd vel = mSkelChar->getVelocities();

    Eigen::VectorXd pos_diff = mSkelChar->getPositionDifferences(mBVHPositions, pos);
    Eigen::VectorXd vel_diff = mSkelChar->getVelocityDifferences(mBVHVelocities, vel);

    if (mImitationType == 0) {
        pos_diff.head<36>().setZero();
        vel_diff.head<36>().setZero();
    } else {
        cout << "[Warning] Imitation type " << mImitationType << " is not implemented" << endl;
        exit(1);
    }
    const double d_p = pos_diff.squaredNorm();
    const double d_v = vel_diff.squaredNorm();
    const double r_p = exp(-mImitationPosCoeff * d_p);
    const double r_v = exp(-mImitationVelCoeff * d_v);

    double r_total = (0.1 + 0.9 * r_p) * (0.1 + 0.9 * r_v);
    if (r_total > mImitationRewClip) r_total = mImitationRewClip;

    if(pGraphData != nullptr){
        if(pGraphData->key_exists("rew_imit_pos")) pGraphData->push("rew_imit_pos", r_p);
        if(pGraphData->key_exists("rew_imit_vel")) pGraphData->push("rew_imit_vel", r_v);
    }
    return r_total;
}

double Environment::GetReward(CBufferData* pGraphData)
{
    const double sway = _swayReward(pGraphData);
    const double imitation = _imitationReward(pGraphData);
    const double head = _headReward(pGraphData);
    const double velocity = _velocityReward(pGraphData);
    const double footstep = _stepReward(pGraphData);
	const double loco = velocity * footstep;
	double meta = mEnergy->MetabolicReward();
	double dev = _deviceReward();

    double gait = loco * sway * head;
    double r_total = gait * mWeightLoco;
    if(mDevRewMultiply) r_total *= dev;
    else r_total += dev * mWeightDev;
    r_total += imitation * mWeightImitation;
    r_total += mWeightMeta * meta;

	if (dart::math::isNan(r_total)) {
        cout << "[ERROR] Nan detected in reward calculation" << endl;
        return 0;
    }

    if(pGraphData != nullptr){
        // pGraphData->push_ma("rew_loco", loco, 0.02);
        // pGraphData->push_ma("rew_footstep", footstep, 0.02);
        // pGraphData->push_ma("rew_velocity", velocity, 0.02);
        // pGraphData->push_ma("rew_gait", gait, 0.02);
        // pGraphData->push_ma("rew_meta", meta, 0.02);
        // pGraphData->push_ma("rew_head", head, 0.02);
        // pGraphData->push_ma("rew_abduction", abduction, 0.02);
        // pGraphData->push_ma("rew_dev", dev, 0.02);
        // pGraphData->push_ma("rew_total", r_total, 0.02);

        if(pGraphData->key_exists("rew_loco")) pGraphData->push("rew_loco", loco);
        if(pGraphData->key_exists("rew_footstep")) pGraphData->push("rew_footstep", footstep);
        if(pGraphData->key_exists("rew_velocity")) pGraphData->push("rew_velocity", velocity);
        if(pGraphData->key_exists("rew_gait")) pGraphData->push("rew_gait", gait);
        if(pGraphData->key_exists("rew_meta")) pGraphData->push("rew_meta", meta);
        if(pGraphData->key_exists("rew_meta_act")) pGraphData->push("rew_meta_act", mEnergy->GetActReward());
        if(pGraphData->key_exists("rew_meta_torque")) pGraphData->push("rew_meta_torque", mEnergy->GetTorqueReward());
        if(pGraphData->key_exists("rew_head")) pGraphData->push("rew_head", head);
        if(pGraphData->key_exists("rew_sway")) pGraphData->push("rew_sway", sway);
        if(pGraphData->key_exists("rew_dev")) pGraphData->push("rew_dev", dev);
        if(pGraphData->key_exists("rew_imit")) pGraphData->push("rew_imit", imitation);
        if(pGraphData->key_exists("rew_total")) pGraphData->push("rew_total", r_total);
    }

    mRewardMap["loco"] = loco;
    mRewardMap["gait"] = gait;
    mRewardMap["meta"] = meta;
    mRewardMap["head"] = head;
    mRewardMap["sway"] = sway;
    mRewardMap["dev"] = dev;
    mRewardMap["imitation"] = imitation;
    mRewardMap["total"] = r_total;
	return r_total;
}

double Environment::GetAvgVelocity() const
{
	int horizon = mMaxTime * mSimulationHz / mPhaseRatio;
	Eigen::Vector3d avg_vel;
	if(mComTrajectory.size() > horizon) avg_vel = ((mComTrajectory.back() - mComTrajectory[mComTrajectory.size() - horizon]) / horizon) * mSimulationHz;
	else if (mComTrajectory.size() <= 1) avg_vel = mSkelChar->getCOMLinearVelocity();
	else avg_vel = ((mComTrajectory.back() - mComTrajectory.front()) / (mComTrajectory.size() - 1)) * mSimulationHz;
	return avg_vel[2];
}

void Environment::LoadRecordConfig(const std::string &_config_path){
    const auto config_path = path_rel_to_abs(_config_path);
    if (!filesystem::exists(config_path)) {
        std::cerr << "Record config file not found: " << config_path << std::endl;
        return;
    }

    try {
        // Load and parse the YAML configuration file
        YAML::Node config = YAML::LoadFile(config_path);

        // Ensure the 'record' node exists and is a map
        if (!config["record"] || !config["record"].IsMap()) {
            std::cerr << "Invalid or missing 'record' section in config file: " << config_path << std::endl;
            return;
        }

        YAML::Node record = config["record"];

        // Parse 'metabolic' section
        if (record["metabolic"] && record["metabolic"].IsMap()) {
            YAML::Node metabolic = record["metabolic"];
            mRECORD_METABOLIC          = metabolic["enabled"]   && metabolic["enabled"].as<bool>()  || mRECORD_METABOLIC;
            mRECORD_MINE               = metabolic["mine"]      && metabolic["mine"].as<bool>()     || mRECORD_MINE;
            mRECORD_HOUD               = metabolic["houd"]      && metabolic["houd"].as<bool>()     || mRECORD_HOUD;
            mRECORD_A                  = metabolic["a"]         && metabolic["a"].as<bool>()        || mRECORD_A;
            mRECORD_A2                 = metabolic["a2"]        && metabolic["a2"].as<bool>()       || mRECORD_A2;
            mRECORD_A3                 = metabolic["a3"]        && metabolic["a3"].as<bool>()       || mRECORD_A3;

            mRECORD_M05A                = metabolic["m05a"]        && metabolic["m05a"].as<bool>()       || mRECORD_M05A;
            mRECORD_M05A2               = metabolic["m05a2"]       && metabolic["m05a2"].as<bool>()      || mRECORD_M05A2;
            mRECORD_M05A3               = metabolic["m05a3"]       && metabolic["m05a3"].as<bool>()      || mRECORD_M05A3;

            mRECORD_MA                 = metabolic["ma"]        && metabolic["ma"].as<bool>()       || mRECORD_MA;
            mRECORD_MA2                = metabolic["ma2"]       && metabolic["ma2"].as<bool>()      || mRECORD_MA2;
            mRECORD_MA3                = metabolic["ma3"]       && metabolic["ma3"].as<bool>()      || mRECORD_MA3;

            mRECORD_M2A                = metabolic["m2a"]       && metabolic["m2a"].as<bool>()      || mRECORD_M2A;
            mRECORD_M2A2               = metabolic["m2a2"]      && metabolic["m2a2"].as<bool>()     || mRECORD_M2A2;
            mRECORD_M2A3               = metabolic["m2a3"]      && metabolic["m2a3"].as<bool>()     || mRECORD_M2A3;

            mRECORD_A15                = metabolic["a15"]       && metabolic["a15"].as<bool>()      || mRECORD_A15;
            mRECORD_M05A15             = metabolic["m05a15"]    && metabolic["m05a15"].as<bool>()   || mRECORD_M05A15;
            mRECORD_MA15               = metabolic["ma15"]      && metabolic["ma15"].as<bool>()     || mRECORD_MA15;
            mRECORD_M2A15              = metabolic["m2a15"]     && metabolic["m2a15"].as<bool>()    || mRECORD_M2A15;

            mRECORD_A125               = metabolic["a125"]      && metabolic["a125"].as<bool>()     || mRECORD_A125;
            mRECORD_M05A125            = metabolic["m05a125"]   && metabolic["m05a125"].as<bool>()  || mRECORD_M05A125;
            mRECORD_MA125              = metabolic["ma125"]     && metabolic["ma125"].as<bool>()    || mRECORD_MA125;
            mRECORD_M2A125             = metabolic["m2a125"]    && metabolic["m2a125"].as<bool>()   || mRECORD_M2A125;

            mRECORD_BHAR               = metabolic["bhar"]      && metabolic["bhar"].as<bool>()     || mRECORD_BHAR;
        }

        if (record["activation"] && record["activation"].IsMap()) {
            YAML::Node activation = record["activation"];
            mRECORD_ACTIVATION_ARM  = activation["arm"]     && activation["arm"].as<bool>()     || mRECORD_ACTIVATION_ARM;
            mRECORD_ACTIVATION_LEG  = activation["leg"]     && activation["leg"].as<bool>()     || mRECORD_ACTIVATION_LEG;
        }

        if (record["angle"] && record["angle"].IsMap()) {
            YAML::Node angle = record["angle"];
            mRECORD_ANGLE       = angle["enabled"]  && angle["enabled"].as<bool>()  || mRECORD_ANGLE;
            mRECORD_ANGLE_HIP   = angle["hip"]      && angle["hip"].as<bool>()      || mRECORD_ANGLE_HIP;
            mRECORD_ANGLE_KNEE  = angle["knee"]     && angle["knee"].as<bool>()     || mRECORD_ANGLE_KNEE;
            mRECORD_ANGLE_ANKLE = angle["ankle"]    && angle["ankle"].as<bool>()    || mRECORD_ANGLE_ANKLE;
        }

        if (record["velocity"] && record["velocity"].IsMap()) {
            YAML::Node velocity = record["velocity"];
            mRECORD_VEL         = velocity["enabled"]   && velocity["enabled"].as<bool>()   || mRECORD_VEL;
            mRECORD_VEL_HIP     = velocity["hip"]       && velocity["hip"].as<bool>()       || mRECORD_VEL_HIP;
            mRECORD_VEL_KNEE    = velocity["knee"]      && velocity["knee"].as<bool>()      || mRECORD_VEL_KNEE;
            mRECORD_VEL_ANKLE   = velocity["ankle"]     && velocity["ankle"].as<bool>()     || mRECORD_VEL_ANKLE;
        }

        // Parse other boolean settings
        mRECORD_POWER       = record["power"]       && record["power"].as<bool>()   || mRECORD_POWER;
        mRECORD_MOMENT      = record["moment"]      && record["moment"].as<bool>()  || mRECORD_MOMENT;
        mRECORD_FOOT        = record["foot"]        && record["foot"].as<bool>()    || mRECORD_FOOT;
        mRECORD_CONTACT     = record["contact"]     && record["contact"].as<bool>() || mRECORD_CONTACT;
        mRECORD_LEFT        = record["left"]        && record["left"].as<bool>()    || mRECORD_LEFT;
        mRECORD_GRF         = record["grf"]         && record["grf"].as<bool>()     || mRECORD_GRF;

#ifdef LOG_VERBOSE
        cout << "Record config file loaded from: " << config_path << endl;
        cout << "Record config: " << record << endl;
#endif

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

vector<string> Environment::GetMuscleNames() const {
    vector<string> muscle_names;
    for(const auto& muscle: mCharacter->GetMuscles()) muscle_names.push_back(muscle->GetName());
    return muscle_names;
}

unordered_map<string, int> Environment::GetMuscleIndices(const vector<string>& names) const{
    unordered_map<string, int> muscle_indices;
    for(int i = 0; i < mCharacter->GetMuscles().size(); i++){
        const auto& muscle = mCharacter->GetMuscles()[i];
        const auto muscle_name = muscle->GetName();
        if (std::find(names.begin(), names.end(), muscle_name) != names.end()){
            muscle_indices[muscle_name] = i;
        }
    }
    return muscle_indices;
}

vector<string> Environment::GetRecordFields() const
{
	vector<string> field_names;
	field_names.emplace_back("step");
	field_names.emplace_back("cycle");
    field_names.emplace_back("time");
    field_names.emplace_back("com_z");

    if (mRECORD_METABOLIC) {
        if (mRECORD_BHAR) field_names.emplace_back("meta_bhar");
        if (mRECORD_MINE) field_names.emplace_back("meta_mine");
        if (mRECORD_HOUD) field_names.emplace_back("meta_houd");
        if (mRECORD_A) field_names.emplace_back("meta_a");
        if (mRECORD_A2) field_names.emplace_back("meta_a2");
        if (mRECORD_A3) field_names.emplace_back("meta_a3");
        if (mRECORD_M05A) field_names.emplace_back("meta_m05a");
        if (mRECORD_M05A2) field_names.emplace_back("meta_m05a2");
        if (mRECORD_M05A3) field_names.emplace_back("meta_m05a3");
        if (mRECORD_MA) field_names.emplace_back("meta_ma");
        if (mRECORD_MA2) {
            field_names.emplace_back("meta_ma2");
            // field_names.emplace_back("meta_ma2_leg");
            // field_names.emplace_back("meta_ma2_glt_med");
            // field_names.emplace_back("meta_ma2_quadriceps");
            // field_names.emplace_back("meta_ma2_hamstrings");
            // field_names.emplace_back("meta_ma2_he");
        }
        if (mRECORD_MA3) field_names.emplace_back("meta_ma3");
        if (mRECORD_M2A) field_names.emplace_back("meta_m2a");
        if (mRECORD_M2A2) field_names.emplace_back("meta_m2a2");
        if (mRECORD_M2A3) field_names.emplace_back("meta_m2a3");
        
        if (mRECORD_A15) field_names.emplace_back("meta_a15");
        if (mRECORD_M05A15) field_names.emplace_back("meta_m05a15");
        if (mRECORD_MA15) field_names.emplace_back("meta_ma15");
        if (mRECORD_M2A15) field_names.emplace_back("meta_m2a15");

        if (mRECORD_A125) field_names.emplace_back("meta_a125");
        if (mRECORD_M05A125) field_names.emplace_back("meta_m05a125");
        if (mRECORD_MA125) field_names.emplace_back("meta_ma125");
        if (mRECORD_M2A125) field_names.emplace_back("meta_m2a125");
    }

    if (mRECORD_ANGLE) {
        if (mRECORD_ANGLE_HIP) {
            field_names.emplace_back("sway_Torso_X");
            field_names.emplace_back("angle_HipR");
            field_names.emplace_back("angle_HipAbR");
            field_names.emplace_back("angle_HipIRR");
            field_names.emplace_back("dev_angleR");
            if (mRECORD_LEFT) field_names.emplace_back("angle_HipL");
        }
        if (mRECORD_ANGLE_KNEE) {
            field_names.emplace_back("angle_KneeR");
            if (mRECORD_LEFT) field_names.emplace_back("angle_KneeL");
        }
        if (mRECORD_ANGLE_ANKLE) {
            field_names.emplace_back("angle_AnkleR");
            if (mRECORD_LEFT) field_names.emplace_back("angle_AnkleL");
        }
    }

    if (mRECORD_VEL) {
        if (mRECORD_VEL_HIP) {
            field_names.emplace_back("velocity_HipR");
            if (mRECORD_LEFT) field_names.emplace_back("velocity_HipL");
        }
        if (mRECORD_VEL_KNEE) {
            field_names.emplace_back("velocity_KneeR");
            if (mRECORD_LEFT) field_names.emplace_back("velocity_KneeL");
        }
        if (mRECORD_VEL_ANKLE) {
            field_names.emplace_back("velocity_AnkleR");
            if (mRECORD_LEFT) field_names.emplace_back("velocity_AnkleL");
        }
    }

    if (mRECORD_MOMENT) {
        field_names.emplace_back("meta_moment_tau");
        field_names.emplace_back("meta_moment_tau_hipR");
        field_names.emplace_back("dev_moment_HipR");
        if (mRECORD_LEFT) field_names.emplace_back("dev_moment_HipL");
    }

    if (mRECORD_POWER) {
        field_names.emplace_back("meta_power");
        field_names.emplace_back("meta_power_HipR");
        field_names.emplace_back("dev_power_HipR");
        if (mRECORD_LEFT) field_names.emplace_back("dev_power_HipL");
    }

    if (mRECORD_FOOT) {
        field_names.emplace_back("foot_R");
        if (mRECORD_LEFT) field_names.emplace_back("foot_L");
    }

    if (mRECORD_CONTACT) {
        field_names = Record::cat_vec(field_names, Record::apply_prefix("contact", {
                "R", "GRF_R"
        }));
        field_names.emplace_back("foot_R");
        if (mRECORD_LEFT) {
            field_names = Record::cat_vec(field_names, Record::apply_prefix("contact", {
                    "L", "GRF_L"
            }));
            field_names.emplace_back("foot_L");
        }
    }

    if (mRECORD_GRF) {
        field_names.emplace_back("grf_r_x");
        field_names.emplace_back("grf_r_y");
        field_names.emplace_back("grf_r_z");
    }

    for(const auto& muscle: mCharacter->GetMuscles()) {
        const auto muscle_type = muscle->getType();
        if ((mRECORD_ACTIVATION_LEG  && (muscle_type.has(MuscleType::leg))) ||
            (mRECORD_ACTIVATION_ARM  && (muscle_type.has(MuscleType::arm)))) {
            const auto muscle_name = muscle->GetName();
            if(muscle_name.find("R_") != string::npos)field_names.emplace_back("act_" + muscle_name);
            else if(mRECORD_LEFT) field_names.emplace_back("act_" + muscle_name);
        }
    }
	return field_names;
}

void Environment::_initFinalize()
{
	mCharacter->SetWorld(mWorld);
	mCharacter->SetUseMuscle(mUseMuscle);
	mCharacter->SetHz(mSimulationHz, mControlHz);
	mCharacter->SetSelfCollisionCheck(mSelfCollision);

	mWorld->getConstraintSolver()->setCollisionDetector(dart::collision::BulletCollisionDetector::create());
	mWorld->setGravity(Eigen::Vector3d(0, -9.8, 0.0));
	mWorld->setTimeStep(1.0 / mSimulationHz);
	mWorld->addSkeleton(mSkelChar);
    
    if (mUseTerrain) {
        TerrainConfig config;    
        mTerrainManager = std::make_unique<TerrainManager>(config);
        mTerrainManager->initialize(mWorld);
    } else {
        mGround = BuildGround(Eigen::Vector3d(50.0, 1.0, mWorldSize));
        mWorld->addSkeleton(mGround);
    }

    mSkelDof = (int)mSkelChar->getNumDofs();
    mRootDof = (int)mSkelChar->getRootBodyNode()->getParentJoint()->getNumDofs();
    mPdDof = mSkelDof - mRootDof;
    if (mJntType == JntType::FullMuscle) mMcnDof = mPdDof;
    else if (mJntType == JntType::LowerMuscle) mMcnDof = 18;

    mPdAction = Eigen::VectorXd::Zero(mPdDof);
	mTargetPositions = Eigen::VectorXd::Zero(mSkelDof);
	mTargetVelocities = Eigen::VectorXd::Zero(mSkelDof);
	mBVHPositions = Eigen::VectorXd::Zero(mSkelDof);
	mBVHVelocities = Eigen::VectorXd::Zero(mSkelDof);
	mDesiredTorque = Eigen::VectorXd::Zero(mSkelDof);
	mDesiredTorqueHistory.clear();
    for (int i = 0; i < mDtHistSize; i++) mDesiredTorqueHistory.push_back(Eigen::VectorXd::Zero(mSkelDof));
    mSetDesiredTorque = false;
    mSkelLengthParamNum = (int)mSkelLengthParams.size();
    mSkelMassParamNum = (int)mSkelMassParams.size();

	mMaxTime = (double)mCharacter->GetBVH()->GetMaxTime();

    _initMuscle();
    _initMotion();
    _initContact();
    _initDevice();

	mWorld->reset();
	mCharacter->Reset();

	mCycleTime = 0.0; mCycleDist = 0.0; mCurCycleTime = mPrevCycleTime = mWorld->getTime();
	mCurCycleCOM = mPrevCycleCOM = mSkelChar->getCOM();
	mNumState = GetState().rows();

	if (mIsRender) mWorld->addSkeleton(mReferenceSkeleton);
    SetParam(GetParam(), true);
    Reset();
}

void Environment::_initMotion()
{
	mCharacter->GetBVH()->SetPureMotions();
	mCharacter->SetMirrorMotion();
	mCharacter->GetBVH()->ResetModifiedMotions();
}

void Environment::_initMuscle()
{
    if (!mUseMuscle) return;

	int num_rel_dofs = 0;
	for (auto m : mCharacter->GetMuscles())num_rel_dofs += m->GetRelDofs();

    const int num_muscle = mCharacter->GetMuscles().size();
    mMT.JtA = Eigen::MatrixXd::Zero(mMcnDof, num_muscle);
    mMT.JtA_reduced = Eigen::VectorXd::Zero(num_rel_dofs);
    mMT.tau_active = Eigen::VectorXd::Zero(mMcnDof);
    mMT.JtP = Eigen::VectorXd::Zero(mMcnDof);
    mActivations = Eigen::VectorXd::Zero(num_muscle);

    const auto& mg = mCharacter->GetMuscleGroupInCharacter();
    mMuscleLengthParams.emplace_back(_build_muscle_param("muscle_length_Hyperlordosis", 0.1));
    mMuscleLengthParams.emplace_back(_build_muscle_param("muscle_length_Equinus", 0.5));
    mMuscleForceParams.emplace_back(_build_muscle_param("muscle_force_Waddling", 0.0));
    mMuscleForceParams.emplace_back(_build_muscle_param("muscle_force_Calcaneal", 0.0));
    mMuscleForceParams.emplace_back(_build_muscle_param("muscle_force_Footdrop", 0.0));

    mMuscleLengthParamNum = static_cast<int>(mMuscleLengthParams.size());
    mMuscleForceParamNum = static_cast<int>(mMuscleForceParams.size());

    if (mUseConstraint){
		for(int i=0; i<mCharacter->GetChangedMuscles().size(); i++)
		{
			Muscle* m = mCharacter->GetChangedMuscles()[i];
			m->set_l_mt_max(1.1);
            const auto constraint = make_shared<MuscleLimitConstraint>(m);
			mCharacter->AddMuscleLimitConstraint(constraint);
            mWorld->getConstraintSolver()->addConstraint(constraint);
		}
	}

    mNumMuscleState = GetMuscleState().rows();
    mEnergy->LoadMuscles(mCharacter->GetMuscles());
}

void Environment::_initDevice()
{
    if (mUseDevice){
#ifdef LOG_VERBOSE
        cout << "[Environment] _initDevice: simHz=" << mSimulationHz
             << " devHz=" << mDeviceHz
             << " deviceStateNum will be set after init" << endl;
#endif
        mDevice->SetHz(mSimulationHz, mDeviceHz);
        mDevice->SetWorld(mWorld);
        mDevice->SetCharacter(mCharacter);
        mDevice->AddConstraint();
        mDevice->Initialize();
        mDeviceStateNum = mDevice->GetState().rows();
#ifdef LOG_VERBOSE
        cout << "[Environment] _initDevice done: deviceStateNum=" << mDeviceStateNum << endl;
#endif
    } else {
#ifdef LOG_VERBOSE
        cout << "[Environment] _initDevice: SKIPPED (mUseDevice=false)" << endl;
#endif
    }
}

void Environment::_initContact()
{
    mContact = new Contact(mWorld, mSkelChar);
    if (mUseTerrain) mContact->Initialize(mTerrainManager.get());
    else mContact->Initialize(mGround->getBodyNode("ground"));
    mContact->setDebouncerAlpha(mDebouncerAlpha);
}

void Environment::_resetPhaseTime()
{
	double t = 0;
	double phase_state = dart::math::Random::uniform(0.0, 1.0);
	t = (phase_state * mMaxTime) - fmod(t, mCharacter->GetBVH()->GetTimeStep());

	mLocalTime = t;
	mGlobalTime = t;

	mWorld->setTime(t);

	mCurCycleTime = mWorld->getTime();
	mPrevCycleTime = mWorld->getTime();
}

void Environment::_resetMuscle()
{
	mMT.JtA.setZero();
	mMT.JtA_reduced.setZero();
	mMT.tau_active.setZero();
	mActivations.setZero();
    for (auto& muscle : mCharacter->GetMuscles()) muscle->Reset();
}

void Environment::_resetContact()
{
    mRightPhaseUpTime = 0.0; mLeftPhaseUpTime = 0.0;
	mContact->Reset();
	mContact->SetContact(true, true);
}

void Environment::_resetFoot()
{
    const auto footR = mSkelChar->getBodyNode("TalusR")->getCOM();
    const auto footL = mSkelChar->getBodyNode("TalusL")->getCOM();
    double phase = GetPhase();
	if(phase >= 0.81 || phase < 0.31) {
		mCurrentStance = LEFT_STANCE;
        mCurrentStanceFoot = footL;
	} else {
        mCurrentStanceFoot = footR;
	}
    if (phase < 0.31) {
        mBlockSwingR = true;
        mContactPhaseR = false;
        mBlockSwingL = true;
        mContactPhaseL = true;
    } else if (phase < 0.41) {
        mBlockSwingR = false;
        mContactPhaseR = true;
        mBlockSwingL = false;
        mContactPhaseL = true;
    } else if (phase < 0.81) {
        mBlockSwingR = false;
        mContactPhaseR = true;
        mBlockSwingL = true;
        mContactPhaseL = false;
    } else if (phase < 0.91) {
        mBlockSwingR = false;
        mContactPhaseR = true;
        mBlockSwingL = true;
        mContactPhaseL = true;
    } else {
        mBlockSwingR = true;
        mContactPhaseR = false;
        mBlockSwingL = true;
        mContactPhaseL = true;
    }

    mCurrentStanceFoot[1] = 0;
    
    mFootPrevRz = footR[2];
    mFootPrevLz = footL[2];
    mStrideLengthR = 0;
    mStrideLengthL = 0;
    mCurrentTargetFoot = mCurrentStanceFoot;
    mNextTargetFoot = mCurrentStanceFoot;
    mNextTargetFoot[2] += GetTargetStride() / 2;
    mNextTargetFoot[0] *= -1;
    
}

void Environment::_updateDevice(Record* pRecord, CBufferData* pGraphData){
    mDevice->AddMoment(mSimStep, mRECORD_MOMENT ? pRecord : nullptr, pGraphData);
    mDevice->AddPower(mSimStep, mRECORD_POWER ? pRecord : nullptr, pGraphData);
    mDevice->AddVelocity(mSimStep, pRecord, pGraphData);
    mDevice->AddAngle(mSimStep, mRECORD_ANGLE_HIP ? pRecord : nullptr, pGraphData);
}

void Environment::_updateCom(Record *pRecord, CBufferData* pGraphData) {
    mCurCOM = mSkelChar->getBodyNode("Pelvis")->getCOM();
    mComTrajectory.push_back(mCurCOM);

    if (pRecord == nullptr) return;
    queue<pair<string, double>> record_buf;
    // record_buf.emplace("com_x", mCurCOM[0]);
    // record_buf.emplace("com_y", mCurCOM[1]);
    record_buf.emplace("com_z", mCurCOM[2]);
    record_buf.emplace("time", mWorld->getTime());
    pRecord->add(mSimStep, record_buf);
}

void Environment::_updateMetabolic(Record *pRecord, CBufferData* pGraphData) {
    // const double accum_divisor = (mEnergy->GetEnergyMode() == EnergyMode::MA2COT || mEnergy->GetEnergyMode() == EnergyMode::MA2COT2) ? abs(mCurCOM[2] - mPrevCOM[2]) : 1.0;
    const double accum_divisor = 1.0;
    JntType jntType = mEnergy->GetJntType();
    if (jntType != JntType::Torque) {
        if (mEnergy->GetEnergyMode() == EnergyMode::BHAR) {
            mEnergy->AccumBHAR(accum_divisor, pGraphData);
        } else {
            mEnergy->AccumActivation(mActivations, accum_divisor, pGraphData);
        }
    }
    if (jntType != JntType::FullMuscle) mEnergy->AccumTorque(mDesiredTorque, mSkelChar->getVelocities(), accum_divisor, pGraphData);

    if (!mUseMuscle || (!mRECORD_METABOLIC && (pRecord == nullptr)) && (pGraphData == nullptr)) return;

    AccumData buf;
    queue<pair<string, double>> record_buf;
    double scaling_factor = 1e4 / (mCharacter->GetWeight() * mSimulationHz);

    if (mRECORD_BHAR || (pGraphData != nullptr && pGraphData->key_exists("bhar"))) buf.add_key("bhar");
    if (pGraphData != nullptr && pGraphData->key_exists("bhar_activation")) buf.add_key("bhar_activation");
    if (pGraphData != nullptr && pGraphData->key_exists("bhar_maintenance")) buf.add_key("bhar_maintenance");
    if (pGraphData != nullptr && pGraphData->key_exists("bhar_shortening")) buf.add_key("bhar_shortening");
    if (pGraphData != nullptr && pGraphData->key_exists("bhar_mechanical_work")) buf.add_key("bhar_mechanical_work");

    if (pGraphData != nullptr && pGraphData->key_exists("umberger")) buf.add_key("umberger");
    if (pGraphData != nullptr && pGraphData->key_exists("umberger_activation")) buf.add_key("umberger_activation");
    if (pGraphData != nullptr && pGraphData->key_exists("umberger_maintenance")) buf.add_key("umberger_maintenance");
    if (pGraphData != nullptr && pGraphData->key_exists("umberger_shortening")) buf.add_key("umberger_shortening");
    if (pGraphData != nullptr && pGraphData->key_exists("umberger_mechanical_work")) buf.add_key("umberger_mechanical_work");

    if (mRECORD_MINE || (pGraphData != nullptr && pGraphData->key_exists("mine"))) buf.add_key("mine");
    if (mRECORD_HOUD || (pGraphData != nullptr && pGraphData->key_exists("houd"))) buf.add_key("houd");

    if (mRECORD_A  || (pGraphData != nullptr && pGraphData->key_exists("a"))) buf.add_key("a");
    if (mRECORD_A2 || (pGraphData != nullptr && pGraphData->key_exists("a2"))) buf.add_key("a2");
    if (mRECORD_A3 || (pGraphData != nullptr && pGraphData->key_exists("a3"))) buf.add_key("a3");

    if (mRECORD_M05A || (pGraphData != nullptr && pGraphData->key_exists("m05a"))) buf.add_key("m05a");
    if (mRECORD_M05A2 || (pGraphData != nullptr && pGraphData->key_exists("m05a2"))) buf.add_key("m05a2");
    if (mRECORD_M05A3 || (pGraphData != nullptr && pGraphData->key_exists("m05a3"))) buf.add_key("m05a3");

    if (mRECORD_M2A  || (pGraphData != nullptr && pGraphData->key_exists("m2a"))) buf.add_key("m2a");
    if (mRECORD_M2A2 || (pGraphData != nullptr && pGraphData->key_exists("m2a2"))) buf.add_key("m2a2");
    if (mRECORD_M2A3 || (pGraphData != nullptr && pGraphData->key_exists("m2a3"))) buf.add_key("m2a3");

    if (mRECORD_A15 || (pGraphData != nullptr && pGraphData->key_exists("a15"))) buf.add_key("a15");
    if (mRECORD_M05A15 || (pGraphData != nullptr && pGraphData->key_exists("m05a15"))) buf.add_key("m05a15");
    if (mRECORD_MA15 || (pGraphData != nullptr && pGraphData->key_exists("ma15"))) buf.add_key("ma15");
    if (mRECORD_M2A15 || (pGraphData != nullptr && pGraphData->key_exists("m2a15"))) buf.add_key("m2a15");

    if (mRECORD_A125 || (pGraphData != nullptr && pGraphData->key_exists("a125"))) buf.add_key("a125");
    if (mRECORD_M05A125 || (pGraphData != nullptr && pGraphData->key_exists("m05a125"))) buf.add_key("m05a125");
    if (mRECORD_MA125 || (pGraphData != nullptr && pGraphData->key_exists("ma125"))) buf.add_key("ma125");
    if (mRECORD_M2A125 || (pGraphData != nullptr && pGraphData->key_exists("m2a125"))) buf.add_key("m2a125");

    if (mRECORD_MA || (pGraphData != nullptr && pGraphData->key_exists("ma"))) buf.add_key("ma");
    if (mRECORD_MA3 || (pGraphData != nullptr && pGraphData->key_exists("ma3"))) buf.add_key("ma3");
    if (mRECORD_MA2) { 
        buf.add_key("ma2"); 
        // buf.add_key("ma2_leg"); buf.add_key("ma2_torso");
        // buf.add_key("ma2_quadriceps"); buf.add_key("ma2_hamstrings"); buf.add_key("ma2_he");
        // buf.add_key("ma2_hf"); buf.add_key("ma2_ke"); buf.add_key("ma2_kf"); buf.add_key("ma2_ae");
        // buf.add_key("ma2_af"); buf.add_key("ma2_fl");
    }
    if (pGraphData != nullptr && pGraphData->key_exists("ma2")) { 
        buf.add_key("ma2"); buf.add_key("ma2_leg"); buf.add_key("ma2_torso");
        buf.add_key("ma2_quadriceps"); buf.add_key("ma2_hamstrings"); buf.add_key("ma2_he");
        buf.add_key("ma2_hf"); buf.add_key("ma2_ke"); buf.add_key("ma2_kf"); buf.add_key("ma2_ae");
        buf.add_key("ma2_af"); buf.add_key("ma2_fl");
    }

    for (auto muscle : mCharacter->GetMuscles()) {
        const auto muscle_type = muscle->getType();
        if (mRECORD_BHAR || (pGraphData != nullptr && pGraphData->key_exists("bhar"))) buf.accumulate("bhar", muscle->RateBhar04());
        if (pGraphData != nullptr && pGraphData->key_exists("bhar_activation")) buf.accumulate("bhar_activation", muscle->RateBhar04_Activation());
        if (pGraphData != nullptr && pGraphData->key_exists("bhar_maintenance")) buf.accumulate("bhar_maintenance", muscle->RateBhar04_Maintenance());
        if (pGraphData != nullptr && pGraphData->key_exists("bhar_shortening")) buf.accumulate("bhar_shortening", muscle->RateBhar04_Shortening());
        if (pGraphData != nullptr && pGraphData->key_exists("bhar_mechanical_work")) buf.accumulate("bhar_mechanical_work", muscle->RateBhar04_MechWork());

        if (pGraphData != nullptr && pGraphData->key_exists("umberger")) buf.accumulate("umberger", muscle->RateUmberger03());
        if (pGraphData != nullptr && pGraphData->key_exists("umberger_activation")) buf.accumulate("umberger_activation", muscle->RateUmberger03_Activation());
        if (pGraphData != nullptr && pGraphData->key_exists("umberger_maintenance")) buf.accumulate("umberger_maintenance", muscle->RateUmberger03_Maintenance());
        if (pGraphData != nullptr && pGraphData->key_exists("umberger_shortening")) buf.accumulate("umberger_shortening", muscle->RateUmberger03_Shortening());
        if (pGraphData != nullptr && pGraphData->key_exists("umberger_mechanical_work")) buf.accumulate("umberger_mechanical_work", muscle->RateUmberger03_MechWork());

        if (mRECORD_MINE || (pGraphData != nullptr && pGraphData->key_exists("mine"))) buf.accumulate("mine", muscle->RateMine97());
        if (mRECORD_HOUD || (pGraphData != nullptr && pGraphData->key_exists("houd"))) buf.accumulate("houd", muscle->RateHoud06());

        if (mRECORD_A || (pGraphData != nullptr && pGraphData->key_exists("a"))) buf.accumulate("a", muscle->RateMA());
        if (mRECORD_A2 || (pGraphData != nullptr && pGraphData->key_exists("a2"))) buf.accumulate("a2", muscle->RateA2());
        if (mRECORD_A3 || (pGraphData != nullptr && pGraphData->key_exists("a3"))) buf.accumulate("a3", muscle->RateA3());

        if (mRECORD_M05A || (pGraphData != nullptr && pGraphData->key_exists("m05a"))) buf.accumulate("m05a", muscle->RateM05A());
        if (mRECORD_M05A2 || (pGraphData != nullptr && pGraphData->key_exists("m05a2"))) buf.accumulate("m05a2", muscle->RateM05A2());
        if (mRECORD_M05A3 || (pGraphData != nullptr && pGraphData->key_exists("m05a3"))) buf.accumulate("m05a3", muscle->RateM05A3());

        if (mRECORD_M2A || (pGraphData != nullptr && pGraphData->key_exists("m2a"))) buf.accumulate("m2a", muscle->RateM2A());
        if (mRECORD_M2A2 || (pGraphData != nullptr && pGraphData->key_exists("m2a2"))) buf.accumulate("m2a2", muscle->RateM2A2());
        if (mRECORD_M2A3 || (pGraphData != nullptr && pGraphData->key_exists("m2a3"))) buf.accumulate("m2a3", muscle->RateM2A3());

        if (mRECORD_A15 || (pGraphData != nullptr && pGraphData->key_exists("a15"))) buf.accumulate("a15", muscle->RateA15());
        if (mRECORD_M05A15 || (pGraphData != nullptr && pGraphData->key_exists("m05a15"))) buf.accumulate("m05a15", muscle->RateM05A15());
        if (mRECORD_MA15 || (pGraphData != nullptr && pGraphData->key_exists("ma15"))) buf.accumulate("ma15", muscle->RateMA15());
        if (mRECORD_M2A15 || (pGraphData != nullptr && pGraphData->key_exists("m2a15"))) buf.accumulate("m2a15", muscle->RateM2A15());

        if (mRECORD_A125 || (pGraphData != nullptr && pGraphData->key_exists("a125"))) buf.accumulate("a125", muscle->RateA125());
        if (mRECORD_M05A125 || (pGraphData != nullptr && pGraphData->key_exists("m05a125"))) buf.accumulate("m05a125", muscle->RateM05A125());
        if (mRECORD_MA125 || (pGraphData != nullptr && pGraphData->key_exists("ma125"))) buf.accumulate("ma125", muscle->RateMA125());
        if (mRECORD_M2A125 || (pGraphData != nullptr && pGraphData->key_exists("m2a125"))) buf.accumulate("m2a125", muscle->RateM2A125());

        if (mRECORD_MA || (pGraphData != nullptr && pGraphData->key_exists("ma"))) buf.accumulate("ma", muscle->RateMA());
        if (mRECORD_MA3 || (pGraphData != nullptr && pGraphData->key_exists("ma3"))) buf.accumulate("ma3", muscle->RateMA3());
        if (mRECORD_MA2) {
            const double ma2 = muscle->RateMA2();
            buf.accumulate("ma2", ma2);
        };        
        if (pGraphData != nullptr && pGraphData->key_exists("ma2")) {
            const double ma2 = muscle->RateMA2();
            buf.accumulate("ma2", ma2);
            if (muscle_type.has(MuscleType::leg)) buf.accumulate("ma2_leg", ma2);
            if (muscle_type.has(MuscleType::torso)) buf.accumulate("ma2_torso", ma2);
            if (muscle_type.has(MuscleType::quadriceps)) buf.accumulate("ma2_quadriceps", ma2);
            if (muscle_type.has(MuscleType::hamstrings)) buf.accumulate("ma2_hamstrings", ma2);
            if (muscle_type.has(MuscleType::he)) buf.accumulate("ma2_he", ma2);
            if (muscle_type.has(MuscleType::hf)) buf.accumulate("ma2_hf", ma2);
            if (muscle_type.has(MuscleType::ke)) buf.accumulate("ma2_ke", ma2);
            if (muscle_type.has(MuscleType::kf)) buf.accumulate("ma2_kf", ma2);
            if (muscle_type.has(MuscleType::ae)) buf.accumulate("ma2_ae", ma2);
            if (muscle_type.has(MuscleType::af)) buf.accumulate("ma2_af", ma2);
            if (muscle_type.has(MuscleType::fl)) buf.accumulate("ma2_fl", ma2);
        };
    }
    unordered_map<string, double> cost = buf.multipleBy(scaling_factor);

    if(pGraphData != nullptr){
        double ma_alpha = 1;
        if (pGraphData->key_exists("bhar")) pGraphData->push_ma("bhar", cost["bhar"], ma_alpha);
        if (pGraphData->key_exists("bhar_activation")) pGraphData->push_ma("bhar_activation", cost["bhar_activation"], ma_alpha);
        if (pGraphData->key_exists("bhar_maintenance")) pGraphData->push_ma("bhar_maintenance", cost["bhar_maintenance"], ma_alpha);
        if (pGraphData->key_exists("bhar_shortening")) pGraphData->push_ma("bhar_shortening", cost["bhar_shortening"], ma_alpha);
        if (pGraphData->key_exists("bhar_mechanical_work")) pGraphData->push_ma("bhar_mechanical_work", cost["bhar_mechanical_work"], ma_alpha);
        if (pGraphData->key_exists("umberger")) pGraphData->push_ma("umberger", cost["umberger"], ma_alpha);
        if (pGraphData->key_exists("umberger_activation")) pGraphData->push_ma("umberger_activation", cost["umberger_activation"], ma_alpha);
        if (pGraphData->key_exists("umberger_maintenance")) pGraphData->push_ma("umberger_maintenance", cost["umberger_maintenance"], ma_alpha);
        if (pGraphData->key_exists("umberger_shortening")) pGraphData->push_ma("umberger_shortening", cost["umberger_shortening"], ma_alpha);
        if (pGraphData->key_exists("umberger_mechanical_work")) pGraphData->push_ma("umberger_mechanical_work", cost["umberger_mechanical_work"], ma_alpha);

        if (pGraphData->key_exists("mine")) pGraphData->push_ma("mine", cost["mine"], ma_alpha);
        if (pGraphData->key_exists("houd")) pGraphData->push_ma("houd", cost["houd"], ma_alpha);

        if (pGraphData->key_exists("ma2_leg")) pGraphData->push_ma("ma2_leg", cost["ma2_leg"]);
        if (pGraphData->key_exists("ma2_torso")) pGraphData->push_ma("ma2_torso", cost["ma2_torso"]);
        if (pGraphData->key_exists("ma2_quadriceps")) pGraphData->push_ma("ma2_quadriceps", cost["ma2_quadriceps"]);
        if (pGraphData->key_exists("ma2_hamstrings")) pGraphData->push_ma("ma2_hamstrings", cost["ma2_hamstrings"]);
        if (pGraphData->key_exists("ma2_he")) pGraphData->push_ma("ma2_he", cost["ma2_he"]);
        if (pGraphData->key_exists("ma2_hf")) pGraphData->push_ma("ma2_hf", cost["ma2_hf"]);
        if (pGraphData->key_exists("ma2_ke")) pGraphData->push_ma("ma2_ke", cost["ma2_ke"]);
        if (pGraphData->key_exists("ma2_kf")) pGraphData->push_ma("ma2_kf", cost["ma2_kf"]);
        if (pGraphData->key_exists("ma2_ae")) pGraphData->push_ma("ma2_ae", cost["ma2_ae"]);
        if (pGraphData->key_exists("ma2_af")) pGraphData->push_ma("ma2_af", cost["ma2_af"]);
        if (pGraphData->key_exists("ma2_fl")) pGraphData->push_ma("ma2_fl", cost["ma2_fl"]);

        if (pGraphData->key_exists("a")) pGraphData->push_ma("a", cost["a"]); 
        if (pGraphData->key_exists("a2")) pGraphData->push_ma("a2", cost["a2"]); 
        if (pGraphData->key_exists("a3")) pGraphData->push_ma("a3", cost["a3"]); 

        if (pGraphData->key_exists("ma")) pGraphData->push_ma("ma", cost["ma"]);
        if (pGraphData->key_exists("ma15")) pGraphData->push("ma15", cost["ma15"]);
        if (pGraphData->key_exists("ma2")) pGraphData->push("ma2", cost["ma2"]);
        if (pGraphData->key_exists("ma3")) pGraphData->push_ma("ma3", cost["ma3"]);
    }

    if(mRECORD_METABOLIC){
        if (mRECORD_BHAR) record_buf.emplace("meta_bhar", cost["bhar"]);
        if (mRECORD_MINE) record_buf.emplace("meta_mine", cost["mine"]);
        if (mRECORD_HOUD) record_buf.emplace("meta_houd", cost["houd"]);
        if (mRECORD_A) record_buf.emplace("meta_a", cost["a"]);
        if (mRECORD_A2) record_buf.emplace("meta_a2", cost["a2"]);
        if (mRECORD_A3) record_buf.emplace("meta_a3", cost["a3"]);

        if (mRECORD_M05A) record_buf.emplace("meta_m05a", cost["m05a"]);
        if (mRECORD_M05A2) record_buf.emplace("meta_m05a2", cost["m05a2"]);
        if (mRECORD_M05A3) record_buf.emplace("meta_m05a3", cost["m05a3"]);

        if (mRECORD_M2A) record_buf.emplace("meta_m2a", cost["m2a"]);
        if (mRECORD_M2A2) record_buf.emplace("meta_m2a2", cost["m2a2"]);
        if (mRECORD_M2A3) record_buf.emplace("meta_m2a3", cost["m2a3"]);

        if (mRECORD_A15) record_buf.emplace("meta_a15", cost["a15"]);
        if (mRECORD_M05A15) record_buf.emplace("meta_m05a15", cost["m05a15"]);
        if (mRECORD_MA15) record_buf.emplace("meta_ma15", cost["ma15"]);
        if (mRECORD_M2A15) record_buf.emplace("meta_m2a15", cost["m2a15"]);

        if (mRECORD_A125) record_buf.emplace("meta_a125", cost["a125"]);
        if (mRECORD_M05A125) record_buf.emplace("meta_m05a125", cost["m05a125"]);
        if (mRECORD_MA125) record_buf.emplace("meta_ma125", cost["ma125"]);
        if (mRECORD_M2A125) record_buf.emplace("meta_m2a125", cost["m2a125"]);

        if (mRECORD_MA) record_buf.emplace("meta_ma", cost["ma"]);
        if (mRECORD_MA2) {
            record_buf.emplace("meta_ma2", cost["ma2"]);
            // record_buf.emplace("meta_ma2_leg", cost["ma2_leg"]);
            // record_buf.emplace("meta_ma2_glt_med", cost["ma2_glt_med"]);
            // record_buf.emplace("meta_ma2_quadriceps", cost["ma2_quadriceps"]);
            // record_buf.emplace("meta_ma2_hamstrings", cost["ma2_hamstrings"]);
            // record_buf.emplace("meta_ma2_he", cost["ma2_he"]);
        }
        if (mRECORD_MA3) record_buf.emplace("meta_ma3", cost["ma3"]);
        pRecord->add(mSimStep, record_buf);
    }
}

void Environment::_updateActivation(Record* pRecord, CBufferData* pGraphData)
{
    if (!mUseMuscle) return;

    if (pGraphData != nullptr){
        BufferData buf{
            "actHe_glt_max", "actHe_sem_bra", "actHe_sem_ten", 
            "actHf_illc", "actHf_psoas", "actHf_rec_fem", 
            "actKe_vas_lat", "actKe_vas_med", 
            "actAe_gas_med", "actAe_gas_lat", "actAe_sol", "actAe_tibp",
            "actAf_tiba",
        };
        for(const auto& muscle : mCharacter->GetMuscles()){
            if(muscle->GetName().find("R_Gluteus_Maximus") != string::npos) buf.push("actHe_glt_max", muscle->GetActivation());
            if(muscle->GetName().find("R_Semimembranosus") != string::npos) buf.push("actHe_sem_bra", muscle->GetActivation());
            if(muscle->GetName().find("R_Semitendinosus") != string::npos) buf.push("actHe_sem_ten", muscle->GetActivation());

            if(muscle->GetName().find("R_iliacus") != string::npos) buf.push("actHf_illc", muscle->GetActivation());
            if(muscle->GetName().find("R_Psoas_Major") != string::npos) buf.push("actHf_psoas", muscle->GetActivation());
            if(muscle->GetName().find("R_Rectus_Femoris") != string::npos) buf.push("actHf_rec_fem", muscle->GetActivation());
            
            if(muscle->GetName().find("R_Vastus_Lateralis") != string::npos) buf.push("actKe_vas_lat", muscle->GetActivation());
            if(muscle->GetName().find("R_Vastus_Medialis") != string::npos) buf.push("actKe_vas_med", muscle->GetActivation());

            if(muscle->GetName().find("R_Gastrocnemius_Lateral_Head") != string::npos) buf.push("actAe_gas_lat", muscle->GetActivation());
            if(muscle->GetName().find("R_Gastrocnemius_Medial_Head") != string::npos) buf.push("actAe_gas_med", muscle->GetActivation());
            if(muscle->GetName().find("R_Soleus") != string::npos) buf.push("actAe_sol", muscle->GetActivation());
            if(muscle->GetName().find("R_Tibialis_Posterior") != string::npos) buf.push("actAe_tibp", muscle->GetActivation());

            if(muscle->GetName().find("R_Tibialis_Anterior") != string::npos) buf.push("actAf_tiba", muscle->GetActivation());
        }
        const auto averaged_act = buf.average();
        for(const auto& [key, value] : averaged_act) {
            pGraphData->push(key, value);
        }
    //     for(const auto& [group_name, muscles]: mMuscleForceGroups){
    //         double force = 0;
    //         for(const auto& muscle: muscles) force += muscle->GetForce();
    //         pGraphData->push_ma("force_" + group_name, force);
    //     }

    //     double force_total = 0;
    //     for(auto muscle : mCharacter->GetMuscles()) force_total += muscle->GetForce();
    //     pGraphData->push_ma("force_total", force_total);
    }
    if ((mRECORD_ACTIVATION_LEG || mRECORD_ACTIVATION_GAIT || mRECORD_ACTIVATION_ARM) && (pRecord != nullptr)) {
        queue<pair<string, double>> record_buf;
        for(const auto& muscle : mCharacter->GetMuscles()) {
            const auto muscle_type = muscle->getType();
            if ((mRECORD_ACTIVATION_LEG  && muscle_type.has(MuscleType::leg)) ||
                (mRECORD_ACTIVATION_ARM  && muscle_type.has(MuscleType::arm))) {
                const auto record_name = "act_" + muscle->GetName();
                if (record_name.find("R_") != string::npos){
                    record_buf.emplace(record_name, muscle->GetActivation());
                }else{
                    if(mRECORD_LEFT){
                        record_buf.emplace(record_name, muscle->GetActivation());
                    }
                }
            }
        }
        pRecord->add(mSimStep, record_buf);
    }
}

void Environment::_updateKinematics(Record* pRecord, CBufferData* pGraphData)
{
    // update angle
    if (pGraphData != nullptr) {
        if(pGraphData->key_exists("sway_Torso_X")) {
            const double rootx = mSkelChar->getRootBodyNode()->getCOM()[0];
            const double torsoX = mSkelChar->getBodyNode("Torso")->getCOM()[0] - rootx;
            pGraphData->push("sway_Torso_X", torsoX);
        }

        if (pGraphData->key_exists("angle_HipR")) {
            const double angleHipR = mSkelChar->getJoint("FemurR")->getPosition(0) * 180.0 / M_PI;
            pGraphData->push("angle_HipR", -angleHipR);
        }

        if (pGraphData->key_exists("angle_HipIRR")) {
            const double angleHipR = mSkelChar->getJoint("FemurR")->getPosition(1) * 180.0 / M_PI;
            pGraphData->push("angle_HipIRR", -angleHipR);
        }

        if (pGraphData->key_exists("angle_HipAbR")) {
            const double angleHipR = mSkelChar->getJoint("FemurR")->getPosition(2) * 180.0 / M_PI;
            pGraphData->push("angle_HipAbR", -angleHipR);
        }

        if (pGraphData->key_exists("angle_KneeR")) {
            const double angleKneeR = mSkelChar->getJoint("TibiaR")->getPosition(0) * 180.0 / M_PI;
            pGraphData->push("angle_KneeR", angleKneeR);
        }

        if (pGraphData->key_exists("angle_AnkleR")) {
            const double angleTalusR = mSkelChar->getJoint("TalusR")->getPosition(0) * 180.0 / M_PI;
            pGraphData->push("angle_AnkleR", -angleTalusR);
        }

        if (pGraphData->key_exists("angle_Rotation")) {
            const double angleRotation = mSkelChar->getJoint("Pelvis")->getPosition(1) * 180.0 / M_PI;
            pGraphData->push("angle_Rotation", angleRotation);
        }

        if (pGraphData->key_exists("angle_Obliquity")) {
            const double angleObliquity = mSkelChar->getJoint("Pelvis")->getPosition(2) * 180.0 / M_PI;
            pGraphData->push("angle_Obliquity", angleObliquity);
        }

        if (pGraphData->key_exists("angle_Tilt")) {
            const double angleTilt = mSkelChar->getJoint("Pelvis")->getPosition(0) * 180.0 / M_PI;
            pGraphData->push("angle_Tilt", angleTilt);
        }

        const double max_joint_velocity = mSkelChar->getVelocities().maxCoeff();
        const double max_joint_acceleration = mSkelChar->getAccelerations().maxCoeff();
        double max_body_velocity = 0;
        double max_body_acceleration = 0;
        for (int i = 0; i < mSkelChar->getNumBodyNodes(); i++) {
            max_body_velocity = max(max_body_velocity, mSkelChar->getBodyNode(i)->getCOMLinearVelocity().maxCoeff());
            max_body_acceleration = max(max_body_acceleration, mSkelChar->getBodyNode(i)->getCOMLinearAcceleration().maxCoeff());
        }
        if (pGraphData->key_exists("max_joint_velocity")) pGraphData->push("max_joint_velocity", max_joint_velocity);
        if (pGraphData->key_exists("max_joint_acceleration")) pGraphData->push("max_joint_acceleration", max_joint_acceleration);
        if (pGraphData->key_exists("max_body_velocity")) pGraphData->push("max_body_velocity", max_body_velocity);
        if (pGraphData->key_exists("max_body_acceleration")) pGraphData->push("max_body_acceleration", max_body_acceleration);
    }

    if (mRECORD_ANGLE && pRecord != nullptr) {
        
        queue<pair<string, double>> record_buf;
        if (mRECORD_ANGLE_HIP) {
            const double rootx = mSkelChar->getRootBodyNode()->getCOM()[0];
            const double torsoX = mSkelChar->getBodyNode("Torso")->getCOM()[0] - rootx;
            record_buf.emplace("sway_Torso_X", torsoX);

            const double angleHipR = mSkelChar->getJoint("FemurR")->getPosition(0) * 180.0 / M_PI;
            const double angleHipAbR = mSkelChar->getJoint("FemurR")->getPosition(2) * 180.0 / M_PI;
            const double angleHipIRR = mSkelChar->getJoint("FemurR")->getPosition(1) * 180.0 / M_PI;
            record_buf.emplace("angle_HipR", -angleHipR);
            record_buf.emplace("angle_HipAbR", -angleHipAbR);
            record_buf.emplace("angle_HipIRR", -angleHipIRR);
        }
        if (mRECORD_ANGLE_KNEE) {
            const double angleKneeR = mSkelChar->getJoint("TibiaR")->getPosition(0) * 180.0 / M_PI;
            record_buf.emplace("angle_KneeR", angleKneeR);
        }
        if (mRECORD_ANGLE_ANKLE) {
            const double angleTalusR = mSkelChar->getJoint("TalusR")->getPosition(0) * 180.0 / M_PI;
            record_buf.emplace("angle_AnkleR", -angleTalusR);
        }
        pRecord->add(mSimStep, record_buf);
    }
}

Eigen::VectorXd Environment::_compute_desired_torque()
{
	Eigen::VectorXd p_des = mTargetPositions;
	p_des.tail(mPdDof) += mPdAction;
    Eigen::VectorXd tau = mCharacter->GetSPDForces(p_des);
    if (!mSetDesiredTorque) mSetDesiredTorque = true;
    else if (mActionAlpha > 0) {
        tau = tau * mActionAlpha + mDesiredTorque * (1 - mActionAlpha);
    }
    tau.head(6).setZero();
    return tau;
}

void Environment::_updateRefChar(CBufferData* pGraphData) {
    if (pGraphData != nullptr) Utils::setSkelPosAndVel(mReferenceSkeleton, mTargetPositions, mTargetVelocities);
}

void Environment::_updateFoot(Record* pRecord, CBufferData* pGraphData)
{
    bool log_foot_info = false;
    log_foot_info &= mIsRender;

	mContact->Set();
    // grf
    if (mRECORD_GRF && pRecord != nullptr) {
        queue<pair<string, double>> record_buf;
        const auto grf_r = mContact->GetGRF_3d(CONTACT_RIGHT);
        const double BW = mSkelChar->getMass() * 9.81;
        record_buf.emplace("grf_r_x", grf_r.x() / BW);
        record_buf.emplace("grf_r_y", grf_r.y() / BW);
        record_buf.emplace("grf_r_z", grf_r.z() / BW);
        pRecord->add(mSimStep, record_buf);
    }
    double phase = GetPhase();
    bool isRightStance = (phase < 0.5);
	double time = mWorld->getTime();
	const double footRx = mSkelChar->getBodyNode("TalusR")->getCOM()[0];
	const double footLx = mSkelChar->getBodyNode("TalusL")->getCOM()[0];
	const double footRz = mSkelChar->getBodyNode("TalusR")->getCOM()[2];
	const double footLz = mSkelChar->getBodyNode("TalusL")->getCOM()[2];
    const double grfR = mContact->GetGRF(CONTACT_RIGHT) / (9.81 * mSkelChar->getMass());
    const double grfL = mContact->GetGRF(CONTACT_LEFT) / (9.81 * mSkelChar->getMass());

	const double stride = GetTargetStride();
	double step_min = mStepMinRatio * stride;
    if (mStateType.has(StateType::velocity)) step_min *= mTargetCOMVelocity / GetReferenceTargetVelocity();

	const bool contactR = mContact->isContact(CONTACT_RIGHT) ? STANCE_PHASE : SWING_PHASE;
    const bool contactL = mContact->isContact(CONTACT_LEFT) ? STANCE_PHASE : SWING_PHASE;

	if(mContactPhaseR != contactR)
	{
		if(contactR == STANCE_PHASE && mContactPhaseR == SWING_PHASE && mContactPhaseL == STANCE_PHASE && (footRz - mFootPrevRz) > step_min && mBlockSwingR && grfR > mGRFPhaseChangeRatio)
		{
            if (log_foot_info) cout << setprecision(3) << "(right) step curr: " << footRz - mFootPrevRz << ", step min: " << step_min << endl;
            mSwingTimeR = time - mRightPhaseUpTime;
            mRightPhaseUpTime = time;

            mCurrentTargetFoot = mNextTargetFoot;
            mNextTargetFoot = mCurrentStanceFoot + Eigen::Vector3d::UnitZ() * stride;
            mCurrentStanceFoot = mSkelChar->getBodyNode("TalusR")->getCOM();
            mStrideLengthR = footRz - mFootPrevRz;
            mFootPrevRz = footRz;

            mContactPhaseR = STANCE_PHASE;
            mBlockSwingL = false;

            mCurCycleCOM = mSkelChar->getBodyNode("Pelvis")->getCOM();
            mCycleDist = (mCurCycleCOM - mPrevCycleCOM).norm();
            mPrevCycleCOM = mCurCycleCOM;

            mCurCycleTime = mWorld->getTime();
            mCycleTime = mCurCycleTime - mPrevCycleTime;
            mPrevCycleTime = mCurCycleTime;
            mCurrentCadence = 2 / mCycleTime;
            mStanceRatioR = mStanceTimeR / mCycleTime;
            mStanceRatioL = mStanceTimeL / mCycleTime;
            mIsGaitCycleComplete = true;
            mCycleCount++;
            // mEnergy->HeelStrikeCb();
            if (log_foot_info) cout << "stance right" << endl;
		}

		if(contactR == SWING_PHASE && mContactPhaseR == STANCE_PHASE && mContactPhaseL == STANCE_PHASE && !mBlockSwingR)
		{
            mStanceTimeR = time - mRightPhaseUpTime;
            mRightPhaseUpTime = time;
            mPhaseTotalR = mStanceTimeR + mSwingTimeR;

            mContactPhaseR = SWING_PHASE;
            mBlockSwingR = true;
            if (log_foot_info) cout << "swing right" << endl;
		}
	}
	if(mContactPhaseL != contactL)
	{
		if(contactL == STANCE_PHASE && mContactPhaseL == SWING_PHASE && mContactPhaseR == STANCE_PHASE && (footLz - mFootPrevLz) > step_min && mBlockSwingL && grfL > mGRFPhaseChangeRatio)
		{
                if (log_foot_info) cout << setprecision(3) << "(left) step curr: " << footLz - mFootPrevLz << ", step min: " << step_min << endl;
				mSwingTimeL = time - mLeftPhaseUpTime;
                mLeftPhaseUpTime = time;

                mCurrentTargetFoot = mNextTargetFoot;
                mNextTargetFoot = mCurrentStanceFoot + Eigen::Vector3d::UnitZ() * stride;
                mCurrentStanceFoot = mSkelChar->getBodyNode("TalusL")->getCOM();
                mStrideLengthL = footLz - mFootPrevLz;
                mFootPrevLz = footLz;
				
                mContactPhaseL = STANCE_PHASE;
                mBlockSwingR = false;
				// mEnergy->HeelStrikeCb();
                if (log_foot_info) cout << "stance left" << endl;
		}

		if(contactL == SWING_PHASE && mContactPhaseL == STANCE_PHASE && mContactPhaseR == STANCE_PHASE && !mBlockSwingL)
		{
            mStanceTimeL = time - mLeftPhaseUpTime;
            mLeftPhaseUpTime = time;
            mPhaseTotalL = mStanceTimeL + mSwingTimeL;

            mContactPhaseL = SWING_PHASE;
            mBlockSwingL = true;
            if (log_foot_info) cout << "swing left" << endl;
		}
	}

    if(pGraphData != nullptr){
        if(pGraphData->key_exists("contact_phaseR")) pGraphData->push("contact_phaseR", mContactPhaseR);
        if(pGraphData->key_exists("contact_GRF_R")) pGraphData->push_ma("contact_GRF_R", grfR / (9.81 * mSkelChar->getMass()), 0.1);
        if(pGraphData->key_exists("footstep")) pGraphData->push("footstep", (mStrideLengthR + mStrideLengthL) / 2);

        // GRF 3D components
        const auto grf_r_3d = mContact->GetGRF_3d(CONTACT_RIGHT);
        const double BW = mSkelChar->getMass() * 9.81;
        if(pGraphData->key_exists("grf_x")) pGraphData->push("grf_x", grf_r_3d.x() / BW);
        if(pGraphData->key_exists("grf_y")) pGraphData->push("grf_y", grf_r_3d.y() / BW);
        if(pGraphData->key_exists("grf_z")) pGraphData->push("grf_z", grf_r_3d.z() / BW);
    }

    if (pRecord == nullptr) return;
    queue<pair<string, double>> record_buf;
    record_buf.emplace("cycle", mCycleCount);
    if (mRECORD_FOOT) {
        record_buf.emplace("foot_R", footRz);
        if (mRECORD_LEFT) record_buf.emplace("foot_L", footLz);
    }
    if (mRECORD_CONTACT) {
        record_buf.emplace("contact_R", mContactPhaseR);
        record_buf.emplace("contact_GRF_R", grfR);
        if (mRECORD_LEFT){
            record_buf.emplace("contact_L", contactL);
            record_buf.emplace("contact_GRF_L", grfL);
        }
    }
    pRecord->add(mSimStep, record_buf);
}

double Environment::_syncArmReward()
{
    // Sync Femur and Arm's angle difference between xy plane
    // If FemurR > FemurL, ArmR > ArmL, vice versa, reward is 1.0 * mSyncArmWeight
    if (!mSyncArm) return 0.0f;

    const double femurR_XY = GetAngleXY(mSkelChar->getBodyNode("Pelvis"), mSkelChar->getBodyNode("FemurR"));
    const double femurL_XY = GetAngleXY(mSkelChar->getBodyNode("Pelvis"), mSkelChar->getBodyNode("FemurL"));
    const double armR_XY = GetAngleXY(mSkelChar->getBodyNode("Head"), mSkelChar->getBodyNode("ForeArmR"));
    const double armL_XY = GetAngleXY(mSkelChar->getBodyNode("Head"), mSkelChar->getBodyNode("ForeArmL"));
    const double femur_diff = femurR_XY - femurL_XY;
    const double arm_diff = armR_XY - armL_XY;
    const double reward = (femur_diff * arm_diff < 0) ? mSyncArmWeight : 0.0f;
    return reward;
}


double Environment::_swayReward(CBufferData* pGraphData)
{
    if (mCycleCount < 1) return 1.0;

    // No deflection along the x-axis
    const double rootx = mSkelChar->getRootBodyNode()->getCOM()[0];
    const double footRx = abs(abs(mSkelChar->getBodyNode("TalusR")->getCOM()[0] - rootx) - mMedianFootX);
    const double footLx = abs(abs(mSkelChar->getBodyNode("TalusL")->getCOM()[0] - rootx) - mMedianFootX);
    const double footRXZ = GetAngleXZ(mSkelChar->getBodyNode("TalusR"));
    const double footLXZ = GetAngleXZ(mSkelChar->getBodyNode("TalusL"));
    
    if (pGraphData != nullptr) {
        if(pGraphData->key_exists("sway_Foot_R")) pGraphData->push_ma("sway_Foot_R", footRx);
        if(pGraphData->key_exists("sway_Foot_RXZ")) pGraphData->push("sway_Foot_RXZ", footRXZ);
    }

    if (mSwayType == SwayType::none) return 1.0;

    double deflection = 0.0;
    if (mSwayType.has(SwayType::foot_x)) {
        if (footRx > mDiffFootX * mSwayMarginRatio) deflection += footRx;
        if (footLx > mDiffFootX * mSwayMarginRatio) deflection += footLx;
    }

    if (mSwayType.has(SwayType::foot_angle)) {
        if (footRXZ > 0.2) deflection += footRXZ;
        else if (footRXZ < 0) deflection += -footRXZ + 0.2;

        if (footLXZ > 0.2) deflection += footLXZ;
        else if (footLXZ < 0) deflection += -footLXZ + 0.2;
    }

    if (mSwayType.has(SwayType::arm)) {
        const double armRx = abs(mSkelChar->getBodyNode("ForeArmR")->getCOM()[0] - rootx);
        const double armLx = abs(mSkelChar->getBodyNode("ForeArmL")->getCOM()[0] - rootx);
        const double swayArmR = abs(armRx - mMedianArmX);
        const double swayArmL = abs(armLx - mMedianArmX);
        if (swayArmR > mDiffArmX * mSwayMarginRatio) deflection += swayArmR;
        if (swayArmL > mDiffArmX * mSwayMarginRatio) deflection += swayArmL;
    }

    if (mSwayType.has(SwayType::spine)) {
        const double spine_lat_med = abs(mSkelChar->getJoint("Spine")->getPosition(0));
        const double spine_sup_inf = abs(mSkelChar->getJoint("Spine")->getPosition(1));
        const double spine_dor_ven = abs(mSkelChar->getJoint("Spine")->getPosition(2));
        const double torso_lat_med = abs(mSkelChar->getJoint("Torso")->getPosition(0));
        const double torso_sup_inf = abs(mSkelChar->getJoint("Torso")->getPosition(1));
        const double torso_dor_ven = abs(mSkelChar->getJoint("Torso")->getPosition(2));
        if (spine_lat_med > 0.052) deflection += spine_lat_med; // 0.052 rad = 3 deg
        if (spine_lat_med > 0.026) deflection += spine_sup_inf; // 0.026 rad = 1.5 deg
        deflection += spine_dor_ven;
        if (torso_lat_med > 0.052) deflection += torso_lat_med; // 0.052 rad = 3 deg
        if (torso_lat_med > 0.026) deflection += torso_sup_inf; // 0.026 rad = 1.5 deg
        deflection += torso_dor_ven;
    }

    if (mSwayType.has(SwayType::pelvis)) {
        const double tilt = abs(mSkelChar->getJoint("Pelvis")->getPosition(0) - mTiltRange / 2.0);
        const double rotation = abs(mSkelChar->getJoint("Pelvis")->getPosition(1));
        const double oblique = abs(mSkelChar->getJoint("Pelvis")->getPosition(2));
        if (tilt > mTiltRange / 2.0) deflection += tilt;
        if (rotation > mRotationRange) deflection += rotation;
        if (oblique > mObliqueRange) deflection += oblique;
    }

    return min(1.0, exp(- mSwayCoeff * deflection));
}

double Environment::_stepReward(CBufferData* pGraphData)
{
	if(mStateType.has(StateType::velocity) && mCycleCount < 2) return 1.0;

    const auto stride = GetTargetStride();
    // const auto margin_ratio = (mVelMargin > 0.05) ? mVelMargin : 0.05;
    const auto margin_ratio = mVelMargin;
    const auto stride_margin = stride * margin_ratio;
    double foot_diff = 0;
    const auto foot_diff_R = abs(mStrideLengthR - stride);
    const auto foot_diff_L = abs(mStrideLengthL - stride);
    const auto foot_diff_x = abs(mCurrentTargetFoot[0] - mCurrentStanceFoot[0]);
    // cout << "ep: " << mSimStep << ", foot_diff_R: " << foot_diff_R << ", foot_diff_L: " << foot_diff_L << ", foot_diff_x: " << foot_diff_x << " margin: " << mVelMargin << endl;

    if (foot_diff_R > stride_margin) foot_diff += max(foot_diff_R, 3 * stride_margin);
    if (foot_diff_L > stride_margin) foot_diff += max(foot_diff_L, 3 * stride_margin);
    if (foot_diff_x > stride * 0.01) foot_diff += foot_diff_x;

	double footClearPenalty = 1.0;
    if(mUseFootClear && 
    (
        (mContactPhaseR==SWING_PHASE && mContact->isContact(CONTACT_RIGHT)) || 
        (mContactPhaseL==SWING_PHASE && mContact->isContact(CONTACT_LEFT)) ||
        (mContactPhaseL==STANCE_PHASE && !mContact->isContact(CONTACT_LEFT)) ||
        (mContactPhaseR==STANCE_PHASE && !mContact->isContact(CONTACT_RIGHT))
    )
    ) footClearPenalty = 0.75;

	double r = Utils::exp_of_squared(foot_diff, 2.5) * footClearPenalty;
	if(pGraphData != nullptr && pGraphData->key_exists("rew_step")) pGraphData->push("rew_step", r);
    
    // cout << "foot clear penalty: " << footClearPenalty << " contactR: " << mContact->isContact(CONTACT_RIGHT) << " contactL: " << mContact->isContact(CONTACT_LEFT) << endl;
	return r;
}

double Environment::_headReward(CBufferData* pGraphData)
{
    double reward = 1.0;
    BodyNode* head = mSkelChar->getBodyNode("Head");
    Eigen::Vector3d head_vel = head->getCOMLinearVelocity();
    Eigen::Vector3d head_angvel = head->getAngularVelocity();
    BodyNode* pelvis = mSkelChar->getBodyNode("Pelvis");
    Eigen::Vector3d pelvis_vel = pelvis->getCOMLinearVelocity();
    // Eigen::Vector3d pelvis_angvel = pelvis->getAngularVelocity();

    if (mCycleCount >= 1) {
        double rotationDifference = Eigen::AngleAxisd(head->getTransform().linear()).angle();
        Eigen::Vector3d linearVelocityDifference = head_vel - mHeadPrevLinearVel - (pelvis_vel - mPelvisPrevLinearVel);
        linearVelocityDifference[0] *= 2.0; linearVelocityDifference[1] *= mWeightAccY; linearVelocityDifference[2] *= 0.5;

        Eigen::Vector3d angularVelocityDifference = head_angvel - mHeadPrevAngularVel;

        // double linearAcceleration = linearVelocityDifference.norm() * mControlHz / 40; // 1000 is a scaler to unify mHeadExpCoeff for acc and rot
        double linearAcceleration = linearVelocityDifference.norm() * mControlHz / 30; // 1000 is a scaler to unify mHeadExpCoeff for acc and rot
        // double angularAcceleration = angularVelocityDifference.norm() * mControlHz / 1000;
        double angularAcceleration = angularVelocityDifference.norm() * mControlHz / 500;

        const double w_alive = 0.1;
        if (rotationDifference > mHeadMarginRatio * 0.006) {
            const double rewardRotation = w_alive + (1.0 - w_alive) * Utils::exp_of_squared(rotationDifference, mHeadExpCoeff);
            reward *= rewardRotation;
        }
        if (linearAcceleration > mHeadMarginRatio * 0.0035) {
            const double rewardLinearAcc = w_alive + (1.0 - w_alive) * Utils::exp_of_squared(linearAcceleration, mHeadExpCoeff);
            reward *= rewardLinearAcc;
        }
        if (angularAcceleration > mHeadMarginRatio * 0.015) {
            const double rewardAngularAcc = w_alive + (1.0 - w_alive) * Utils::exp_of_squared(angularAcceleration, mHeadExpCoeff);
            reward *= rewardAngularAcc;
        }

        if (pGraphData != nullptr) {
            if (pGraphData->key_exists("head_rot")) pGraphData->push("head_rot", rotationDifference);
            if (pGraphData->key_exists("head_linacc")) pGraphData->push("head_linacc", linearAcceleration);
            if (pGraphData->key_exists("head_rotacc")) pGraphData->push("head_rotacc", angularAcceleration);

            Eigen::Vector3d relvel = mSkelChar->getBodyNode("Head")->getCOMLinearVelocity() - mSkelChar->getBodyNode("Pelvis")->getCOMLinearVelocity();
            if (pGraphData->key_exists("head_relvel")) pGraphData->push("head_relvel", relvel.norm());
        }
    }

    mHeadPrevLinearVel = head_vel;
    mHeadPrevAngularVel = head_angvel;
    mPelvisPrevLinearVel = pelvis_vel;
    // mPelvisPrevAngularVel = pelvis_angvel;

    return reward;
}

double Environment::_velocityReward(CBufferData* pGraphData)
{
	const int horizon = floor(mMaxTime*(double)mSimulationHz / (mPhaseRatio*sqrt(1/mGlobalRatio)));
    const auto target = GetTargetVelocity();
	Eigen::Vector3d refVel = {0.0, 0.0, target};

	if(mComTrajectory.size() > horizon) mAvgVel = ((mComTrajectory.back() - mComTrajectory[mComTrajectory.size() - horizon]) / horizon) * mSimulationHz;
	else if (mComTrajectory.size() <= 1) mAvgVel = refVel;
	else mAvgVel = ((mComTrajectory.back()-mComTrajectory.front()) / (mComTrajectory.size()-1))*mSimulationHz;

    Eigen::Vector3d diff = mAvgVel - refVel;
    if (diff.norm() < target * mVelMargin) diff.setZero();

    const double r = Utils::exp_of_squared(diff, mVelCoeff);
    if(pGraphData != nullptr && pGraphData->key_exists("rew_avg_vel")) pGraphData->push("rew_avg_vel", r);
	return r;

}
double Environment::_deviceReward()
{
    if (Utils::close(mDeviceK, 0.0) || mDevRewardType < 0) {
        if (mDevRewMultiply) return 1.0f;
        else return 0.0f;
    }
    const double deviceK = abs(mDeviceK);
    const double denominator = deviceK * mDevRewCoeff;
    const double power_r = mDevice->GetPowerR(), power_l = mDevice->GetPowerL();
    double r_r, r_l;
	if(mDevRewardType==0) { // minimize negative power
        if (mDevRewMultiply) {
            r_r = 1.0; r_l = 1.0;
            if (power_r < 0) r_r *= exp(power_r / denominator);
            if (power_l < 0) r_l *= exp(power_l / denominator);
        } else {
            r_r = 1.0; r_l = 1.0;
            if (power_r < 0) r_r += power_r / denominator;
            if (power_l < 0) r_l += power_l / denominator;
        }
    } else if(mDevRewardType==1) { // maximize positive power
        if (mDevRewMultiply) {
            r_r = 0.5; r_l = 0.5; // alive bonus
            if (power_r > 0) r_r *= exp(power_r / denominator);
            if (power_l > 0) r_l *= exp(power_l / denominator);
        } else {
            r_r = 0.0; r_l = 0.0;
            if (power_r > 0) r_r += power_r / denominator;
            if (power_l > 0) r_l += power_l / denominator;
        }
    } else if(mDevRewardType==2) { // apply 1 and 2 both
        if (mDevRewMultiply) {
            if (power_r > 0) r_r = 0.5 * exp(power_r / (10 * denominator));
            else r_r = 1.0 * exp(power_r / denominator);
            if (power_l > 0) r_l = 0.5 * exp(power_l / (10 * denominator));
            else r_l = 1.0 * exp(power_l / denominator);
        } else {
            if (power_r > 0) r_r = power_r / (10 * denominator);
            else r_r = 1.0 + power_r / denominator;
            if (power_l > 0) r_l = power_l / (10 * denominator);
            else r_l = 1.0 + power_l / denominator;
        }
    } else if(mDevRewardType==3) { // apply 1 and 2 both (parametrized)
        if (mDevRewMultiply) {
            cerr << "[Error] multiply is not supported for device reward type 3" << endl;
            return 1.0f;
        } else {
            if (power_r > 0) r_r = mDevPBias + power_r / (mDevPCoeff * deviceK);
            else r_r = mDevNBias + power_r / (mDevNCoeff * deviceK);
            if (power_l > 0) r_l = mDevPBias + power_l / (mDevPCoeff * deviceK);
            else r_l = mDevNBias + power_l / (mDevNCoeff * deviceK);
        }
    } else {
        // after the commit a4515cce6b7c21662704e8eb3a951d61892d0d7e, miscellanous reward types are not supported
        cerr << "Invalid device reward type: " << mDevRewardType << endl;
        return 0.0f;
    }
    return (r_r + r_l) / 2.0f;
}

Eigen::VectorXd Environment::_mirrorAction(Eigen::VectorXd action)
{
	Eigen::VectorXd full_action = Eigen::VectorXd::Zero(mSkelDof);
	full_action.tail(action.rows()) = action;
	return mCharacter->GetMirrorPosition(full_action).tail(action.rows());
}

void Environment::_loadSkeletonFile(const string& rel_path){
    const auto abs_path = path_rel_to_abs(rel_path).string();
    const auto loadObj = true;
    mCharacter->LoadSkeleton(abs_path, loadObj, mJointDamping);
    mSkelChar = mCharacter->GetSkeleton();
    if (mIsRender) {
        mReferenceSkeleton = BuildFromFile(abs_path, true, mJointDamping, Eigen::Vector4d(1, 0, 0, 0.1), false);
        mReferenceSkeleton->setName("ReferenceHuman");
    }
}

void Environment::_loadSkeletonParam(const string& rel_path){
    const auto abs_path = path_rel_to_abs(rel_path).string();
    mSkelInfos = Character::LoadSkelParamFile(abs_path);
    mSkelInfos_ref = mSkelInfos;
    mCharacter->ModifySkeletonLengthAndMass(mSkelInfos);
}

void Environment::_loadDeviceFile(const string& rel_path){
    const auto abs_path = path_rel_to_abs(rel_path).string();
    const auto load_obj = true;
    if (mUseDevice) mDevice->LoadSkeleton(abs_path, load_obj, mJointDamping);
}
void Environment::_loadDeviceParam(const string& rel_path) {
    const auto abs_path = path_rel_to_abs(rel_path).string();
    if (mUseDevice) {
        mSkelDeviceInfos = mDevice->LoadSkelParamFile(abs_path);
        mSkelDeviceInfos_ref = mSkelDeviceInfos;
        mDevice->ModifySkeletonLength(mSkelDeviceInfos, false);
    }
}
MuscleParam Environment::_build_muscle_param(const string& param_name, double min_val, bool isGroup){
    MuscleParam mp(&mMuscleGroups, {}, param_name, min_val);
    mp.isGroup = isGroup;
    bool isValid = false;
    for(const auto& m: mCharacter->GetMuscles()){
        if(mp.Compare(m->GetName())) {
            mCharacter->AddChangedMuscle(m);
            mp.muscle.push_back(m);
            isValid = true;
            if(!isGroup) break;
        }
    }
    if(!isValid) {
        cout << "Invalid muscle parameter: " << param_name << endl;
        exit(-1);
    }
    return mp;
}

} // namespace MASS
