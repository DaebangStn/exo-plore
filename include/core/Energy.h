#pragma once

#include "Muscle.h"
#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include "util/struct.h"


namespace MASS
{
class Energy
{
public:
	Energy(JntType jntType);
	~Energy() = default;

	void LoadYaml(const YAML::Node& node);
	void Reset();
	void AccumActivation(const Eigen::VectorXd& actLevels, double accum_divisor, CBufferData* pGraphData = nullptr);
	void AccumBHAR(double accum_divisor, CBufferData* pGraphData = nullptr);
	void AccumTorque(Eigen::VectorXd torque, const Eigen::VectorXd& vels, double accum_divisor, CBufferData* pGraphData = nullptr);
	// void HeelStrikeCb();
	double MetabolicReward();
	// double TargetEnergy() const;

	JntType GetJntType() const { return mJntType; }
	double GetActRewCoeff() const { return mActRewCoeff; }
	double GetTorqueRewCoeff() const { return mTorqueRewCoeff; }
	double GetActReward() const { return mActReward; }
	double GetTorqueReward() const { return mTorqueReward; }
	double GetTargetCoeff() const { return mTargetCoeff; }
	EnergyMode GetEnergyMode() const { return mMode; }
	EnergyRewardCurve GetEnergyRewardCurve() const { return mEnergyRewardCurve; }

    void SetJntType(JntType jntType);
	void SetEnergyRewardCurve(EnergyRewardCurve curve) { mEnergyRewardCurve = curve; }
    void SetActRewCoeff(double coeff) { mActRewCoeff = coeff; }
    void SetTorqueRewCoeff(double coeff) { mTorqueRewCoeff = coeff; }
    void SetEnergyMode(EnergyMode mode) { mMode = mode; }
	void SetTargetCoeff(double coeff) { mTargetCoeff = coeff; _update_weights(); }
	void LoadMuscles(const std::vector<Muscle*> muscles);
private:
	double _reward_from_energy(double energy) const;
	void _reset_accumulators();
	void _update_weights();
	
	Eigen::VectorXd mMuscleMass, mMuscleMass05, mEnergyWeight, mTargetWeight;
	std::vector<Muscle*> mMuscles;
	EnergyMode mMode = EnergyMode::POWER;
	EnergyRewardCurve mEnergyRewardCurve = EnergyRewardCurve::EXP;
	double mActRewCoeff = 25.0, mTorqueRewCoeff = 0.005, mTargetCoeff = 4.0;
	double mActAccum = 0.0, mTorqueAccum = 0.0, mTargetAccum = 0.0, mAccumDivisor = 0.0;
	double mActReward = 0.0, mTorqueReward = 0.0;
	bool mUseMuscle = true;
	JntType mJntType = JntType::Torque;
	Utils::MAFilter mEnergyAvg;
};

}
