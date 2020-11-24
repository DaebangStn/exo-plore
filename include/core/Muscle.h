#ifndef __MASS_MUSCLE_H__
#define __MASS_MUSCLE_H__
#include "core/Utils.h"
#include "dart/dart.hpp"

using namespace dart::dynamics;
namespace MASS
{
	struct Anchor
	{
		std::vector<BodyNode*> bodynodes;
		std::vector<Eigen::Vector3d> local_positions;
		std::vector<double> weights;
		bool is_lbs = false;

		Anchor(std::vector<BodyNode*> bns, std::vector<Eigen::Vector3d> lps, std::vector<double> ws);
		void GetPoint(Eigen::Ref<Eigen::Vector3d> point) const;
		void GetPoint(Eigen::Ref<Eigen::Vector3d> local_pos, Eigen::Ref<Eigen::Vector3d> global_pos) const;
		void ComputeJacobian(const SkeletonPtr& skel, Eigen::MatrixXd& Jref) const;

	};
    enum MuscleType
    {
        none    = 0,
        leg     = 1 << 0,
        arm     = 1 << 1,
        torso   = 1 << 2,
        quadriceps = 1 << 3,
        hamstrings = 1 << 4,
		he = 1 << 5,
		hf = 1 << 6,
		ke = 1 << 7,
		kf = 1 << 8,
		ae = 1 << 9,
		af = 1 << 10,
		fl = 1 << 11,
    };
	enum class MassType { MT0, M0, M_Handsfield14 };
	static const std::unordered_map<std::string, MuscleType> StringToMuscleType = {
		{"leg", MuscleType::leg},
		{"arm", MuscleType::arm},
		{"torso", MuscleType::torso},
		{"quadriceps", MuscleType::quadriceps},
		{"hamstrings", MuscleType::hamstrings},
		{"he", MuscleType::he},
		{"hf", MuscleType::hf},
		{"ke", MuscleType::ke},
		{"kf", MuscleType::kf},
		{"ae", MuscleType::ae},
		{"af", MuscleType::af},
		{"fl", MuscleType::fl},
	};

	class Muscle
	{
	public:
		Muscle(std::string _name,double f0,double lm0,double lt0,double pen_angle,double lmax, double type1_fraction,
               bool vel, double len_ratio, int type, MassType mass_type, bool fast_mode);
		void AddMeshLbsAnchor(const SkeletonPtr &skel, const Eigen::Vector3d& glob_pos); // Mesh-distance based lbs weight
		void AddPrevMeshLbsAnchor(const SkeletonPtr &skel, const Eigen::Vector3d& glob_pos); // Mesh-distance based lbs weight
        void AddJointLbsAnchor(const SkeletonPtr &skel, const Eigen::Vector3d &glob_pos); // Joint-distance based lbs weight
		void AddSingleBodyAnchor(BodyNode *bn, const Eigen::Vector3d& glob_pos);

		const std::vector<Anchor*>& GetAnchors(){return mAnchors;}
		const std::vector<Anchor*>& GetAnchors() const {return mAnchors;}

		void UpdateGeometry();
		void Reset();
		
		void ApplyForceToBody();
		void ComputeJacobians();

		void SetHz(int sHz, int cHz){ mSimulationHz = sHz; mControlHz = cHz; };
		void SetActivation(double a);
		void SetMuscle();
		void SetMassType(MassType mass_type);
		void SetF0Original(double f0_original) {f0_original = f0_original;}
		void SetF0(double f0) {f0 = f0;}
		void set_l_mt_max(double l_max) { l_mt_max = l_max;}
		void set_mass_ratio(double mass_ratio) { mMassRatio = mass_ratio;}
		void SetShorteningMultiplier(double multiplier) { mShortening_multiplier = multiplier;}
		double GetShorteningMultiplier() const { return mShortening_multiplier;}
        void setMuscleBunchType(MuscleType _muscleBunchType);

		std::string GetName() const { return name;}
		double GetForce() const {return mForce;};
		double GetPassiveForce() const {return mPassiveForce;};
		double GetExcitation(int type);
		double GetLengthRatio() const {return length / l_mt0; }
		double GetActivation() const {return activation;}
		double GetF0() const { return f0;}
		double GetMT0() const {return l_mt0;}
		double GetLMTmax() const {return l_mt_max;}
		double GetLMT() const {return l_mt;}
		double GetTendonSlackLength() const {return l_t0;}
		int GetRelDofs(){return mRelatedDofs;};
        double GetMass() const {return mMass;};

		void GetJacobianTranspose(Eigen::MatrixXd& Jt) const;
		void GetReducedJacobianTranspose(Eigen::MatrixXd& Jt_reduced);
		void GetForceJacobianAndPassive(Eigen::VectorXd& Fa, Eigen::VectorXd& Fp) const;
		std::vector<Joint*> GetRelatedJoints();
		std::vector<BodyNode*> GetRelatedBodyNodes();
		Eigen::VectorXd Getdl_dtheta();
		const Eigen::Matrix3Xd& GetAnchorPos() const {return mAnchorPos;}
        Utils::FlagSet<MuscleType> getType() const {return mMuscleBunchType;}

        bool NameContains(const std::string &str) const {return name.find(str) != std::string::npos;};

        double RateUmberger03();
        double RateUmberger03_Activation();
        double RateUmberger03_Maintenance();
        double RateUmberger03_Shortening();
        double RateUmberger03_MechWork();
        double RateBhar04();
        double RateBhar04_Activation();
        double RateBhar04_Maintenance();
        double RateBhar04_Shortening();
        double RateBhar04_MechWork();


		double RateHoud06();
        double RateMine97();
        double RateJaed() const {return pow(GetMass(),0.5)/mMassMax * pow(activation,3);};
        
		double RateA() const {return activation;};
        double RateA2() const { return pow(activation, 2); };
        double RateA3() const {return pow(activation, 3);};

		double RateM05A() const {return pow(GetMass(), 0.5) * activation;};
        double RateM05A2() const { return pow(GetMass(), 0.5) * pow(activation, 2); };
        double RateM05A3() const {return pow(GetMass(), 0.5) * pow(activation, 3);};

        double RateMA() const {return GetMass() * activation;};
        double RateMA2() const {return GetMass() * pow(activation, 2);};
        double RateMA3() const {return GetMass() * pow(activation, 3);};

        double RateM2A() const {return pow(GetMass(), 2) * activation;};
        double RateM2A2() const {return pow(GetMass(), 2) * pow(activation, 2);};
        double RateM2A3() const {return pow(GetMass(), 2) * pow(activation, 3);};

		double RateA15() const {return pow(activation, 1.5);};
		double RateM05A15() const {return pow(GetMass(), 0.5) * pow(activation, 1.5);};
		double RateMA15() const {return GetMass() * pow(activation, 1.5);};
		double RateM2A15() const {return pow(GetMass(), 2) * pow(activation, 1.5);};

		double RateA125() const {return pow(activation, 1.25);};
		double RateM05A125() const {return pow(GetMass(), 0.5) * pow(activation, 1.25);};
		double RateMA125() const {return GetMass() * pow(activation, 1.25);};
		double RateM2A125() const {return pow(GetMass(), 2) * pow(activation, 1.25);};
			
        double RateFA() const {return f0 * activation;};
        double RateFA2() const {return f0 * pow(activation, 2);};
        double RateFA3() const {return f0 * pow(activation, 3);};

        double RateF2A() const {return f0 * pow(activation, 2);};
        double RateF2A2() const {return pow(f0, 2) * pow(activation, 2);};
        double RateF2A3() const {return pow(f0, 2) * pow(activation, 3);};

		void change_f(double ratio) {f0 = f0_original * ratio; }
		void change_l(double ratio) {l_mt0 = l_mt0_original * ratio;}
        double GetType1_Fraction() {return type1_fraction;}

        std::string name;
		std::vector<Anchor*> mAnchors;
		std::vector<int> related_dof_indices;
        Utils::FlagSet<MuscleType> mMuscleBunchType = MuscleType::none;

		BodyNodePtr given_bn;
		bool selected = false;
		double Getl_mt() const {return length / l_mt0;};

	private:
		std::vector<Eigen::VectorXd> mAnchorWeight;
		Eigen::Matrix3Xd mAnchorPos;
		Eigen::Matrix3Xd mAnchorPosLocal;
		Eigen::Matrix3Xd mAnchorVel;
		Eigen::Matrix3Xd mAnchorPosDif;
		Eigen::Matrix3Xd mAnchorPosDir;
		Eigen::Matrix3Xd mAnchorVelDif;
		Eigen::Matrix3Xd mAnchorForce;
		Eigen::Matrix3Xd mAnchorForceDir;
		Eigen::Matrix3Xd mJRelatedLinear;
		Eigen::MatrixXd mJRelated;
		std::vector<Eigen::MatrixXd> mCachedJs;

		static double GetDistanceFromMesh(BodyNode* bn, const Eigen::Vector3d& glob_pos);
		double Getf_A();
		double Getf_p();
		double Getl_mt0() const {return l_mt0;}
		double Getdl_m() const {return dl_m; }
		double GetV_m() const {return v_m;}
		double Getl_ratio();

		//Dynamics
		double g(double _l_m);
		double g_t(double e_t);
		double g_pl(double _l_m);
		double g_al(double _l_m);
		double g_av(double _v_m);

        double _computeMass() const;

		int mSimulationHz, mControlHz;
		int mNumAnchors, mRelatedDofs;
		int vType = 0;
		
		double dl_m = 0;
		double l_mt = 1, l_mt_max, l_m = 0, v_m = 0;
		double mV_max;
		double activation = 0, excitation = 0;

		inline static double mMassMultiplierInit = 0.0005;
		double mMassMultiplier = mMassMultiplierInit;
		double mMassRatio = 1.0;

		double f0,f0_original;
		double l_mt0_original;
		double l_mt0 = 0, l_m0, l_t0, length;
		double mForce = 0, mFa = 0, mPassiveForce = 0;

		double f_toe = 0.33, e_toe = 0.02, k_toe = 3.0, k_lin = 51.878788, e_t0 = 0.033; //For g_t
		double k_pe = 5.0, e_mo = 0.3; //For g_pl
		double gamma = 0.45; //For g_al

		double mMass;
		double mShortening_multiplier = 1.0;
		
        double type1_fraction;
		double Getdl_velocity();

        const bool mFastMode, mUseMuscleVelocity;
		const double mLenRatio, mMassMax = 12.80158; // pow(3.57793,2);
		const int mType;
		MassType mMassType;

		std::vector<std::unordered_map<int, int>> mDofMap; // anchor index -> global related dof index -> jacobian col index
	};
}
#endif
