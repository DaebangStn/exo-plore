#ifndef __MASS_DEVICE_H__
#define __MASS_DEVICE_H__

#include "core/Character.h"
#include "core/Record.h"
#include "util/struct.h"
#include "core/DARTHelper.h"
#include "yaml-cpp/yaml.h"

using namespace dart::dynamics;
using namespace dart::simulation;
namespace MASS
{
    struct ModifyDeviceInfo
	{
		ModifyDeviceInfo(double lx, double ly, double lz, double sc, double to)
        {
			value[0] = lx;
			value[1] = ly;
			value[2] = lz;
			value[3] = sc;
			value[4] = to;
		};
        ModifyDeviceInfo() : ModifyDeviceInfo(1.0, 1.0, 1.0, 1.0, 0.0) {};

        double value[6]{};
		double& operator[](int idx){ return value[idx]; }
		double operator[](int idx) const { return value[idx]; }
	};

	using BoneDeviceInfo = std::tuple<std::string, ModifyDeviceInfo>;
    class Character;
    enum class AngleType { Hip, HipSlope, Dev, Super, Count };
    class Device
    {
    public:
        Device();
        ~Device();

        void Reset();
        void Step(bool update_torque = true);
        void Initialize();

        void SetWorld(WorldPtr& wPtr){ mWorld = wPtr; }
        void SetCharacter(Character* character){mCharacter = character; }
        void SetHz(int sim_hz, int con_hz) { mSimulationHz = sim_hz; mControlHz = con_hz; }
        void SetOffsetModify(double offset){ mOffsetModify = offset;}
        void SetAngleType(int t){ mAngleType = static_cast<AngleType>(t);}
        void SetParam(double kl, double kr, double t);
        void SetWeightEnabled(bool enabled){mWeightEnabled = enabled; UpdateWeight();}
        void SetWeightForced(bool forced){mWeightForced = forced; UpdateWeight();}
        void SetWeightScaler(float scaler){mWeightScaler = scaler; UpdateWeight();}
        void SetZeroState(bool zero){mZeroState = zero;}
        void SetConstraintCoeff(double stiff, double damp) { mStiff = stiff; mDamp = damp; }
        void SetVirtualCoupling(bool virtual_coupling) { mVirtualCoupling = virtual_coupling; }
        void SetFakeAssist(bool fake_assist) { mFakeAssist = fake_assist; }
        void ToggleZeroState();
        void AddConstraint();
        void RemoveConstraint();
        void LoadYaml(const YAML::Node& node);

        void LoadSkeleton(const std::string& path, bool load_obj, double damping = 0.4);
        std::vector<BoneDeviceInfo> LoadSkelParamFile(const std::string &filename);
        void ModifySkeletonBodyNode(const std::vector<BoneDeviceInfo> &info, SkeletonPtr skel);
        void ModifySkeletonLength(const std::vector<BoneDeviceInfo> &info, bool b);  

        void AddAngle(unsigned int mEpisodeSubStepCount, Record* pRecord, CBufferData* pGraphData = nullptr);
        void AddVelocity(unsigned int mEpisodeSubStepCount, Record* pRecord, CBufferData* pGraphData = nullptr);
		void AddMoment(unsigned int mEpisodeSubStepCount, Record* pRecord, CBufferData* pGraphData = nullptr);
        void AddPower(unsigned int mEpisodeSubStepCount, Record* pRecord, CBufferData* pGraphData = nullptr);

        int GetAngleType() const {return static_cast<int>(mAngleType);}
        int GetFrontIdx() const {return mTorqueSignChangeIdx.front();}
        int GetBackIdx() const {return mTorqueSignChangeIdx.back();}
        double GetK() const {return mKr_;}
        double GetDelay() const {return mDelta_t;}
        double GetTorqueR() const {return mDesiredTorquesR.front();}
        double GetTorqueL() const {return mDesiredTorquesL.front();}
        double GetPosPowerR() const {return mPowerPosR.get();}
        double GetNegPowerR() const {return mPowerNegR.get();}
        double GetPosPowerL() const {return mPowerPosL.get();}
        double GetNegPowerL() const {return mPowerNegL.get();}
        double GetPowerR() const {return mPowerR;}
        double GetPowerL() const {return mPowerL;}
        double GetRMSMomentR() const;
        float GetWeightScaler() const {return mWeightScaler;}
        bool IsWeightEnabled() const {return mWeightEnabled;}
        bool IsWeightForced() const {return mWeightForced;}
        bool IsZeroState() const {return mZeroState;}
        bool IsUseState() const {return mUseState;}
        bool IsOn() const {return !(Utils::close(mKl_, 0.0f) && Utils::close(mKr_, 0.0f));}
        bool IsVirtualCoupling() const {return mVirtualCoupling;}
        bool IsFakeAssist() const {return mFakeAssist;}
        Eigen::VectorXd SuperTorque(const Eigen::VectorXd& spd_output);
        Eigen::VectorXd GetState(bool mirror = true) const;
        std::pair<double, double> GetTorques() const { return std::make_pair(mDesiredTorquesR.front(), mDesiredTorquesL.front()); }
        const SkeletonPtr& GetSkeleton() const { return mSkeleton; }

    private:
        void InitializeMass();
        void SetMassZero();
        void SetMassDefault();
        void UpdateWeight();
        void UpdateDesiredTorques();
        Eigen::VectorXd BuildState(bool mirror, int offset, int history) const;
        double GetHipAngle(const std::string& name) const;
        double GetHipSlopeAngle(const std::string& name) const;
        double GetDevAngle(const std::string& name) const;

        WorldPtr mWorld;
        SkeletonPtr mSkeleton, mStdSkeleton;
        Character* mCharacter;
        AngleType mAngleType = AngleType::Dev;

        double mDelta_t = 0.0;
        double mK_scaler = 30.0;
        float mWeightScaler = 1.0;
        double mK_, mKl_, mKr_;
        double mStiff = 100.0, mDamp = 10.0;
        double mOffsetModify;
        const int mRightTauDofIdx = 6, mLeftTauDofIdx = 9, mRightCharDofIdx = 6, mLeftCharDofIdx = 15;
        int mDelta_t_idx;
        const int mTorqueBufferSize = 200;
        int counter = 0;
        int mSimulationHz, mControlHz;
        int mStateOfs = 1, mStateHist = 16;
        int mTorqueDir = 2;
        bool mZeroState = false, mWeightEnabled = true, mWeightForced = true, mUseState = true, mStateParam = false, 
            mLongState = true, mVirtualCoupling = true, mFakeAssist = false, mPosAssist = false;

        std::deque<int> mTorqueSignChangeIdx;
        std::map<std::string, double> mDefaultMass, mZeroMass;
        std::map<std::string, Eigen::Matrix3d> mDefaultMoment, mZeroMoment;
        double mTotalMass = 0.0, mTotalZeroMass = 0.0;

        std::deque<double> mControlSignals, mDesiredTorquesR, mDesiredTorquesL;

        Utils::MAFilter mPowerPosR, mPowerNegR, mPowerPosL, mPowerNegL, mQr, mQl, mTr, mTl;
        std::shared_ptr<dart::constraint::WeldJointConstraint> mCoupRoot = nullptr, mCoupR = nullptr, mCoupL = nullptr;
        double mQrRaw, mQlRaw, mPowerR, mPowerL;
    };

}
#endif