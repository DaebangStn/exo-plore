#ifndef __MASS_CHARACTER_H__
#define __MASS_CHARACTER_H__

#include "core/Muscle.h"
#include "core/MuscleLimitConstraint.h"
#include "Utils.h"
#include "dart/dart.hpp"
#include <utility>
#include <tinyxml2.h>
#include "yaml-cpp/yaml.h"


namespace MASS
{
	class BVH;
	class Device;
	struct ModifyInfo
	{
		ModifyInfo() : ModifyInfo(1.0, 1.0, 1.0, 1.0, 0.0, 1.0){}
		ModifyInfo(double lx, double ly, double lz, double sc, double to, double m){
			value[0] = lx;
			value[1] = ly;
			value[2] = lz;
			value[3] = sc;
			value[4] = to;
			value[5] = m;
		};
		double value[6];
		double& operator[](int idx){ return value[idx]; }
		double operator[](int idx) const { return value[idx]; }
	};
	
	typedef tinyxml2::XMLElement TiXmlElement;
	typedef tinyxml2::XMLDocument TiXmlDocument;

	using BoneInfo = std::tuple<std::string, ModifyInfo>;
	using namespace dart::dynamics;
	using namespace dart::simulation;
	class Character
	{
	public:
		Character();
		~Character();

		void SetWorld(WorldPtr& wPtr){ mWorld = wPtr; }
		void LoadSkeleton(const std::string &path, bool create_obj = false, double damping = 0.4);
		void LoadMuscleYaml(const YAML::Node& node);
		void LoadBVH(const std::string &path, bool cyclic = true);

		void Reset();		
		void AddEndEffector(const std::string &body_name) { mEndEffectors.push_back(mSkeleton->getBodyNode(body_name)); }

		void setActivation(const Eigen::VectorXd &a);
		void SetAnchor(bool FixUpperOnly = true);
		void ResetAnchor();
		void SetHz(int sHz, int cHz){ mSimulationHz = sHz; mControlHz = cHz; }
		void SetUseMuscle(bool b){ mUseMuscle = b;}
		void SetMirrorMotion();
		void SetCharacterInformation();
		void SetSelfCollisionCheck(bool b) {mSkeleton->setSelfCollisionCheck(b);};
		void SetMuscleMassType(MassType mass_type);
		void SetMuscleMassType(int mass_type);
		int GetMuscleMassType() const {return static_cast<int>(mMuscleMassType);}
		
		double GetHeight() const {return mCharacterHeight;}
		double GetWeight() const {return mCharacterWeight;}
		double GetDefaultHeight() const {return mDefaultHeight;}
		double GetDefaultMass() const {return mDefaultMass;}
		double GetOffsetModify() const {return mOffsetModify;}
		Eigen::VectorXd GetSPDForces(const Eigen::VectorXd &p_desired) const;
		Eigen::VectorXd GetMuscleMass() const { return mMuscleMass; }
		const SkeletonPtr &GetSkeleton() { return mSkeleton; }
		const std::vector<Muscle*> &GetMuscles() { return mMuscles; }
		std::vector<Muscle *> GetChangedMuscles() { return mChangedMuscles; }
		Eigen::VectorXd GetTargetPositions(double t, double dt);
		Eigen::VectorXd GetTargetPositions(double t, double dt, SkeletonPtr skeleton);
		Eigen::VectorXd GetMirrorMomentum(const Eigen::VectorXd& Momentum);
		Eigen::VectorXd GetMirrorPosition(Eigen::VectorXd pos);
		Eigen::VectorXd GetKp(){return mKp;}
		Eigen::VectorXd GetDamping(){return mDamping;}
		std::pair<Eigen::VectorXd, Eigen::VectorXd> GetTargetPosAndVel(double t, double dt);
		std::pair<Eigen::VectorXd, Eigen::VectorXd> GetTargetPosAndVel(double t, double dt, SkeletonPtr skeleton);
		const std::map<std::string, Muscle*> &GetMusclesMap() { return mMusclesMap; }
		Muscle* GetMusclesInMap(std::string name) { return mMusclesMap[name]; }
		const std::vector<BodyNode *> &GetEndEffectors() { return mEndEffectors; }
		BVH *GetBVH() { return mBVH; }
		const std::vector<std::shared_ptr<MuscleLimitConstraint>> &GetMuscleLimitConstraints() { return mMuscleLimitConstraints; };
		std::map<std::string, std::vector<std::string>> GetMuscleGroupInCharacter() const {return mMuscleGroupsInCharacter;}

		void AddMuscleLimitConstraint(const std::shared_ptr<MuscleLimitConstraint> &cst) { mMuscleLimitConstraints.push_back(cst); }
		void AddChangedMuscle(Muscle *muscle_elem) { mChangedMuscles.push_back(muscle_elem); }
		static std::vector<BoneInfo> LoadSkelParamFile(const std::string &filename);
		void ModifySkeletonBodyNode(const std::vector<BoneInfo> &info, SkeletonPtr skel);
		void ModifySkeletonLengthAndMass(const std::vector<BoneInfo> &info, double mass_ratio = 1.0);
	    std::vector<Eigen::Matrix3d> getBodyNodeTransform() { return mBodyNodeTransform; }
		std::vector<std::pair<Joint *, Joint *>> getPairs() { return mPairs; }

	private:
		void _fixUpper();
		void _fixWhole();

		SkeletonPtr mSkeleton, mStdSkeleton;
		WorldPtr mWorld;
		BVH *mBVH;
		Device* mDevice;
		Eigen::Isometry3d mTc;
		Eigen::VectorXd mKp, mKv, mDamping, mMuscleMass, mArmMask, mLegMask, mTorsoMask;
		MassType mMuscleMassType;

		std::map<std::string, std::vector<std::string>> mMuscleGroupsInCharacter;
		std::map<std::string, Muscle*> mMusclesMap;
		std::map<const BodyNode*, ModifyInfo> modifyLog;
		std::vector<Muscle*> mMuscles, mStdMuscles;
		std::vector<BodyNode*> mEndEffectors;
		std::vector<std::shared_ptr<MuscleLimitConstraint>> mMuscleLimitConstraints;
		std::vector<Muscle*> mChangedMuscles;
		std::vector<std::pair<Joint *, Joint *>> mPairs;
		std::vector<Eigen::Matrix3d> mBodyNodeTransform;

		bool mUseOBJ;
		bool mUseMuscle;
		int mSimulationHz, mControlHz;

		double mCharacterHeight, mCharacterWeight;
		double mOffsetModify, mDefaultMass, mDefaultHeight;
		double mTalusSize, mLenRatio, motionScale, yOffset, rootDefaultHeight, footDifference;

        void categorizeMuscles();
    };
}; // namespace MASS

#endif