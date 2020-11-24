#pragma once

#include <utility>

#include "BVH.h"
#include "Muscle.h"
#include "Character.h"
#include "Contact.h"
#include "DARTHelper.h"
#include "Device.h"
#include "GaitData.h"
#include "Energy.h"
#include "Record.h"
#include "TerrainManager.h"
#include "Utils.h"
#include "util/struct.h"
#include "dart/dart.hpp"
#include "boost/circular_buffer.hpp"

using namespace dart::simulation;
using namespace dart::dynamics;
namespace MASS
{
	struct MuscleParam
	{
		double min_ratio;
		double max_ratio;
		double current_ratio;
		std::string name;
		std::vector<Muscle*> muscle;
        std::map<std::string, std::vector<std::string>>* pMuscleGroups;
		bool isGroup;
        MuscleParam(): MuscleParam(nullptr, {}, "", 0.0f, 0.0f, 0.0f) {}
        MuscleParam(std::map<std::string, std::vector<std::string>>* pMuscleGroups, std::vector<Muscle*> muscle,
                    const std::string& name, double min_ratio, double max_ratio = 1.0, double current_ratio = 1.0, bool isGroup = true)
                : min_ratio(min_ratio), max_ratio(max_ratio), current_ratio(current_ratio), name(name), muscle(muscle),
                  pMuscleGroups(pMuscleGroups), isGroup(isGroup) {}

		bool Compare(const std::string& m_name)
		{
			if (isGroup){
				if(pMuscleGroups->count(name)>0){
					for(std::string &mc_name : (*pMuscleGroups)[name]){
						if(m_name.find(mc_name) != std::string::npos)  return true;
					}
					return false;
				}
				else return (m_name.find(name) != std::string::npos);
			}
			else return name == m_name;
		}

		void SetMuscleGroups(std::map<std::string, std::vector<std::string>>* mg){pMuscleGroups = mg;}
	};

	struct SkelParam
	{
		double min_ratio;
		double max_ratio;
		double current_ratio;
		std::string name;
		BodyNode* body_node;
		bool Compare(const std::string& s_name) {return name == s_name;}
	}; 

	struct MuscleTuple
	{
		Eigen::MatrixXd JtA;
		Eigen::VectorXd JtA_reduced;
		Eigen::VectorXd tau_active;
		Eigen::VectorXd JtP;
	};

	class Environment
	{
	public:
		Environment(bool isRender = false, bool force_use_device = false);
		~Environment() = default;

		// Gym Interface
		void Reset();
		void Step(Record* pRecord = nullptr, CBufferData* pGraphData = nullptr, bool actuate_character = true, bool update_device_torque = true);
		double GetReward(CBufferData* pGraphData = nullptr);

		void InitFromYaml(const std::string &path);
        void LoadRecordConfig(const std::string &config_path);

		void SetActivationLevels(const Eigen::VectorXd &a);
		void SetParam(const std::unordered_map<std::string, float>& params, bool allow_duplicated_param = false);
		void SetAction(const Eigen::VectorXd &a, CBufferData* pGraphData);
		void SetAction(const Eigen::VectorXd &a) {SetAction(a, nullptr);}
        void SetSkelLength(double ratio);
        void SetSkelMass(double ratio);
        void SetDesiredTorque(const Eigen::VectorXd &a) {mDesiredTorque = a;}
		void SetActionAlpha(double alpha) {mActionAlpha = alpha;}
		void SetGroundSize(double size) {mWorldSize = size;}
		void SetGRFPhaseChangeRatio(double ratio) {mGRFPhaseChangeRatio = ratio;}
		void SetStepMinRatio(double ratio) {mStepMinRatio = ratio;}
		void SetDeviceZeroState(bool zero_state) {if (mDevice != nullptr) mDevice->SetZeroState(zero_state); else std::cerr << "Device is not initialized" << std::endl;}
		void SetDeviceWeightEnabled(bool enabled) {if (mDevice != nullptr) mDevice->SetWeightEnabled(enabled); else std::cerr << "Device is not initialized" << std::endl;}
		void SetDeviceWeightForced(bool forced) {if (mDevice != nullptr) mDevice->SetWeightForced(forced); else std::cerr << "Device is not initialized" << std::endl;}
		void SetAnkleStiffness(double stiffness) {mAnkleStiffness = stiffness;}
		void SetShorteningMultiplier(double multiplier);
		void GravityOn() {mWorld->setGravity(Eigen::Vector3d(0, -9.81, 0.0));}
		void GravityOff() {mWorld->setGravity(Eigen::Vector3d(0, 0, 0.0));}

		Eigen::VectorXf GetState();
		Eigen::VectorXd GetStateChar();
		Eigen::VectorXd GetStateExo();
		Eigen::VectorXd GetMuscleState(bool mirror = false);
        Eigen::VectorXd GetDesiredTorque() {return mDesiredTorque;}
		const WorldPtr &GetWorld() { return mWorld; }
		const SkeletonPtr &GetGround() { return mGround; }
        const SkeletonPtr &GetReferenceSkeleton() { return mReferenceSkeleton; }
        Character* GetCharacter() { return mCharacter; }
		Energy* GetEnergy() { return mEnergy.get(); }
		TerrainManager* GetTerrainManager() { return mTerrainManager.get(); }
		const MuscleTuple &GetMuscleTuple() const { return mMT; }
		Device* GetDevice() { return mDevice; }
		Contact* GetContact() { return mContact; }
		int GetRelatedDof() const { return mMT.JtA_reduced.rows(); }
		int GetPdDof() const {return mPdDof; }
		int GetMcnDof() const {return mMcnDof; }
		int GetNumState() const { return mNumState; }
		int GetNumMuscles() const { return mCharacter->GetMuscles().size(); }
		int GetNumAction() const { return mPdDof + (mActionType.has(ActionType::time_warp)); }
		int GetNumSteps() const { return mSimulationHz / mControlHz; }
        int GetPhaseStateRight() const { return mContactPhaseR; }
        int GetPhaseStateLeft() const { return mContactPhaseL; }
        int GetMuscleLengthParamNum() const { return mMuscleLengthParamNum; }
        int GetMuscleForceParamNum() const { return mMuscleForceParamNum; }
        int GetCycleCount() const {return mCycleCount;}
        int GetControlHz() const { return mControlHz; }
        int GetSimStep() const { return mSimStep; }
        double GetAnkleStiffness() const { return mAnkleStiffness; }
        double GetShorteningMultiplier() const;
        double GetDeviceK() const { return mDeviceK; }
        double GetDeviceDelay() const { return mDeviceDelay; }
        double GetPhase() const {return fmod(mLocalTime, mMaxTime) / mMaxTime;}
        double GetGlobalPhase() const {return fmod(mGlobalTime, mMaxTime) / mMaxTime;};
        double GetAvgVelocity() const;
        double GetTargetVelocity() const {return mStateType.has(StateType::velocity) ? mTargetCOMVelocity: GetReferenceTargetVelocity();};
        double GetReferenceTargetVelocity() const {return (sqrt(mGlobalRatio) * mStrideRatio * mRefStride) * (mPhaseRatio / mMaxTime);};
        double GetTargetStride() const {return mStateType.has(StateType::velocity) ? -1.0: mStrideRatio * mRefStride;};
        double GetTargetCadence() const {return mStateType.has(StateType::velocity) ? -1.0: (mPhaseRatio / mMaxTime) * 2.0;};
        double GetStrideLengthR() const {return mStrideLengthR;}
        double GetStrideLengthL() const {return mStrideLengthL;}
        double GetCurrentCadence() const {return mCurrentCadence;}
        double GetCycleDist() const {return mCycleDist;}
        double GetCycleTime() const {return mCycleTime;}
        double GetStanceRatioR() const {return mStanceRatioR;}
        double GetStanceRatioL() const {return mStanceRatioL;}
        double GetPhaseTotalR() const {return mPhaseTotalR;}
        double GetPhaseTotalL() const {return mPhaseTotalL;}
		double GetActionAlpha() const { return mActionAlpha; }
		double GetActionPhaseScale() const { return mActionPhaseScale; }
		double GetActionScale() const { return mPdScale; }
		double GetGRFPhaseChangeRatio() const { return mGRFPhaseChangeRatio; }
		double GetStepMinRatio() const { return mStepMinRatio; }
        bool GetUseDevice() const { return mUseDevice; }
        bool GetUseMuscle() const { return mUseMuscle; }
		bool GetUseTerrain() const { return mUseTerrain; }
        int IsEndOfEpisode() const;
        bool IsGaitCycleComplete() const { return mIsGaitCycleComplete; }
		bool IsControlStep() const {return mSimStep % GetNumSteps() == 0;}
        const std::string &GetMetadata() const { return metadata; }
        const Eigen::Vector3d &GetNextTargetFoot() const {return mNextTargetFoot;}
        const Eigen::Vector3d &GetCurrentTargetFoot() const {return mCurrentTargetFoot;}
        const Eigen::VectorXd &GetAction() const {return mPdAction; }
		const Eigen::VectorXd &GetActivationLevels() const { return mActivations; }
		const Eigen::VectorXd &GetTargetPositions() const {return mTargetPositions; }
        const Eigen::VectorXd &GetBVHPositions() const {return mBVHPositions; }
        const Eigen::VectorXd &GetBVHVelocities() const {return mBVHVelocities; }
        const Eigen::VectorXd &GetPowerTau() const {return mPowerTau; }
        const Eigen::VectorXd &GetPowerMuscle() const {return mPowerMuscle; }
        const std::vector<MuscleParam> &GetMuscleLengthParams() const { return mMuscleLengthParams; }
		const std::vector<MuscleParam> &GetMuscleForceParams() const { return mMuscleForceParams; }
		const std::map<std::string, double> &GetRewardMap() const { return mRewardMap; }
        std::unordered_map<std::string, float> GetParam() const;
        std::unordered_map<std::string, int> GetMuscleIndices(const std::vector<std::string>& names) const;
        std::vector<std::string> GetRecordFields() const;
        std::vector<std::string> GetMuscleNames() const;

		void updateMuscleTuple(CBufferData* pGraphData = nullptr);

        const double mMaxFootX = 0.111f;
        const double mMinFootX = 0.078f;
        const double mMaxArmX = 0.298f;
        const double mMinArmX = 0.244f;
        const double mMaxTorsoX = 0.026f;
        const double mMinTorsoX = 0.0f;
        const double mMaxShoulderZ = 0.037f;
        const double mMinShoulderZ = 0.015f;
        const double mMaxTorsoXY = 0.001f;
        const double mMinTorsoXY = -0.025;
        double mMedianFootX, mMedianArmX, mDiffFootX, mDiffArmX, mMedianTorsoX, mDiffTorsoX, mMedianShoulderZ, mDiffShoulderZ, mMedianTorsoXY, mDiffTorsoXY;
		double mTiltRange = 2.0, mRotationRange = 3.0, mObliqueRange = 4.0;
		double mDiffHeadRot = 0.006, mDiffHeadLinacc = 0.0035, mDiffHeadRotacc = 0.015;
        double mHeadMarginRatio = 1.0;

    private:
		double _velocityReward(CBufferData* pGraphData = nullptr);
		double _headReward(CBufferData* pGraphData = nullptr);
		double _stepReward(CBufferData* pGraphData = nullptr);
        double _swayReward(CBufferData* pGraphData = nullptr);
		double _imitationReward(CBufferData* pGraphData = nullptr);
        double _syncArmReward();
		double _deviceReward();
        Eigen::VectorXd _compute_desired_torque();
        Eigen::VectorXd _mirrorAction(Eigen::VectorXd action);
		MuscleParam _build_muscle_param(const std::string& param_name, double min_val, bool isGroup = true);
        void _loadSkeletonFile(const std::string& rel_path);
        void _loadSkeletonParam(const std::string& rel_path);
        void _loadDeviceFile(const std::string& rel_path);
        void _loadDeviceParam(const std::string& rel_path);
		void _updateRefChar(CBufferData* pGraphData = nullptr);
		void _updatePhase(CBufferData* pGraphData = nullptr);
		void _updateFoot(Record* pRecord, CBufferData* pGraphData = nullptr);
        void _updateCom(Record* pRecord, CBufferData* pGraphData = nullptr);
        void _updateMetabolic(Record* pRecord, CBufferData* pGraphData = nullptr);
		void _updateActivation(Record* pRecord, CBufferData* pGraphData = nullptr);
		void _updateKinematics(Record* pRecord, CBufferData* pGraphData = nullptr);
		void _updateDevice(Record* pRecord, CBufferData* pGraphData = nullptr);
		void _updateDynamics(Record* pRecord, CBufferData* pGraphData = nullptr);
		void _initFinalize();
		void _initMotion();
		void _initMuscle();
		void _initDevice();
		void _initContact();
		void _resetMuscle();
		void _resetContact();
		void _resetFoot();
		void _resetPhaseTime();

		std::string metadata;
		WorldPtr mWorld;
		SkeletonPtr mGround, mSkelChar;
		Character* mCharacter;
		Device* mDevice;
		std::unique_ptr<Energy> mEnergy;
		std::unique_ptr<TerrainManager> mTerrainManager;
		MuscleTuple mMT;
		Eigen::VectorXd mPdAction;
        std::map<std::string, double> mRewardMap;

		bool mUseTerrain = false;

		double mWorldSize = 500.0;
		int mControlHz = 50, mSimulationHz = 500, mDeviceHz = 100;
		int mDeviceStateNum;
        int mDevRewardType = -1;
		int mCurrentStance; // 0: Left // 1 : Right

		bool mIsFootSymMode = false;
		bool mSelfCollision = false;
		double mPdScale = 0.04, mActionPhaseScale = 0.001;
		double mJointDamping = 0.2;
		
		bool mActionUpper = false;
		bool mContactPhaseR, mContactPhaseL, mBlockSwingR, mBlockSwingL;

		double mWeightLoco = 2.0, mWeightMeta = 1.0, mWeightDev = 0.0, mWeightAccY = 0.5, mWeightImitation = 1.0;
		double mImitationPosCoeff = 1.0, mImitationVelCoeff = 0.1, mImitationRewClip = 0.9;
		int mImitationType = -1;

		std::map<std::string, std::vector<std::string>> mMuscleGroups;
		Eigen::VectorXd mTargetPositions, mTargetVelocities, mBVHPositions, mBVHVelocities;
		Eigen::Vector3d mCurrentTargetFoot, mNextTargetFoot, mCurrentStanceFoot;
		Eigen::Vector3d mHeadPrevLinearVel, mHeadPrevAngularVel, mPelvisPrevLinearVel, mPelvisPrevAngularVel;
		Eigen::VectorXd mActivations;
		Eigen::VectorXd mDesiredTorque;
		std::deque<Eigen::VectorXd> mDesiredTorqueHistory;
		const int mDtHistSize = 4;

		Eigen::VectorXd mPowerTau, mPowerMuscle;

		std::vector<Eigen::Vector3d> mComTrajectory;

		Eigen::Vector3d mCurCOM, mPrevCOM;
		Eigen::Vector3d mPrevCycleCOM, mCurCycleCOM;
		
		JntType mJntType = JntType::Torque;
		Utils::FlagSet<StateType> mStateType = StateType::base;
		Utils::FlagSet<ActionType> mActionType = ActionType::base;
		Utils::FlagSet<SwayType> mSwayType = SwayType::none;

		bool mUseDevice = false, mUseMuscle = false;
				
		double mCurCycleTime, mPrevCycleTime, mCycleTime, mCycleDist = 1.34;

		int mSkelDof, mNumState, mNumCharState, mNumMuscleState, mPdDof, mRootDof, mMcnDof;
        int mCycleCount, mSimStep;
		bool mIsGaitCycleComplete = false;
		double mFootPrevRz, mFootPrevLz;
				
		// Contact
		Contact* mContact;
		double mGRFPhaseChangeRatio = 0.75, mStepMinRatio = 0.2;
        double mRightPhaseUpTime, mLeftPhaseUpTime, mStanceTimeR, mStanceTimeL, mStanceRatioR, mStanceRatioL, 
			mSwingTimeR, mSwingTimeL, mPhaseTotalR, mPhaseTotalL;

		// Muscle
		int mMuscleLengthParamNum, mMuscleForceParamNum;
		std::vector<MuscleParam> mMuscleLengthParams, mMuscleForceParams;
		
		// Skel
		int mSkelLengthParamNum, mSkelMassParamNum;
		std::vector<SkelParam> mSkelMassParams, mSkelLengthParams;

		SkeletonPtr mReferenceSkeleton;

		bool mUseConstraint = false;
		
		double mGlobalTime = 0.0, mLocalTime = 0.0, mMaxTime = 1.0;
		double mRefStride = 1.34, mStrideRatio = 1.0;
		Eigen::Vector3d mAvgVel, mAvgHeadVel;

		std::vector<BoneInfo> mSkelInfos, mSkelInfos_ref;
		std::vector<BoneDeviceInfo> mSkelDeviceInfos, mSkelDeviceInfos_ref;

		bool mSyncArm = false, mSetDesiredTorque = false;
        bool mDevRewMultiply = false;
		bool mUseFootClear = true;
        double mSyncArmWeight;
        double mSwayCoeff = 1.0;
        double mDevRewCoeff = 5, mDevPCoeff = 20.0, mDevNCoeff = 100.0, mDevPBias = 0.0, mDevNBias = 1.0;
        double mHeadExpCoeff = 4.0;
        double mVelCoeff = 40.0;
        double mVelMargin = 0.15;
        double mSwayMarginRatio = 1.0;

		double mTargetCOMVelocity = 1.34;
		double mAnkleStiffness = -1.0;

		double mPhaseDisplacement;

		double mPhaseRatio = 1.0, mGlobalRatio = 1.0, mMassRatio = 1.0;
        double mCurrentCadence, mCurrentPhase;
		double mStrideLengthR, mStrideLengthL;

		double mDebouncerAlpha = 0.2, mActionAlpha = -1.0;

		double mMuscleForceRatio = 1.0;
        double mDeviceK = 0.0, mDeviceDelay = 0.0;

        bool mRECORD_METABOLIC = false, mRECORD_BHAR = false, 
			mRECORD_BHAR_ARM = false, mRECORD_BHAR_LEG = false, mRECORD_MINE = false, mRECORD_HOUD = false, 
			mRECORD_A = false, mRECORD_A2 = false, mRECORD_A3 = false, 
			mRECORD_M05A = false, mRECORD_M05A2 = false, mRECORD_M05A3 = false, 
			mRECORD_MA = false, mRECORD_MA2 = false, mRECORD_MA3 = false, 
			mRECORD_M2A = false, mRECORD_M2A2 = false, mRECORD_M2A3 = false, 
			mRECORD_A15 = false, mRECORD_M05A15 = false, mRECORD_MA15 = false, mRECORD_M2A15 = false,
			mRECORD_A125 = false, mRECORD_M05A125 = false, mRECORD_MA125 = false, mRECORD_M2A125 = false,
			mRECORD_ANGLE = false, 
			mRECORD_ANGLE_HIP = false, mRECORD_ANGLE_KNEE = false, mRECORD_ANGLE_ANKLE = false, mRECORD_VEL = false, 
			mRECORD_VEL_HIP = false, mRECORD_VEL_KNEE = false, mRECORD_VEL_ANKLE = false, mRECORD_FOOT = false, 
			mRECORD_MOMENT = false, mRECORD_POWER = false, mRECORD_POWER_Z_ONLY = false, 
			mRECORD_CONTACT = false, mRECORD_LEFT = false, mRECORD_ACTIVATION_LEG = false, mRECORD_ACTIVATION_ARM = false,
			mRECORD_ACTIVATION_GAIT = false, mRECORD_GRF = false;

		const bool mForceUseDevice, mIsRender;
    };
}; 
