 #include <iostream>
#include "core/Energy.h"
#include "core/Utils.h"

using namespace std;

namespace MASS
{

Energy::Energy(JntType jntType) : mJntType(jntType), mEnergyAvg(0.0005) {
	if (mJntType >= JntType::FullMuscle) mUseMuscle = true;
}

void Energy::SetJntType(JntType jntType) {
	mJntType = jntType;
	if (mJntType >= JntType::FullMuscle) mUseMuscle = true;
}

void Energy::AccumActivation(const Eigen::VectorXd& actLevels, double accum_divisor, CBufferData* pGraphData) {
	if (mJntType == JntType::Torque) {
		cout << "Accumulating activation for on Energy::mJntType = " << static_cast<int>(mJntType) << endl;
		exit(-1);
	}

	mAccumDivisor += accum_divisor;
	double energy_step = 0.0;
	if (mMode == EnergyMode::A2) energy_step += (actLevels.array() * actLevels.array()).sum();
	else if (mMode == EnergyMode::MA2) energy_step += (mMuscleMass.array() * actLevels.array() * actLevels.array()).sum();
	else if (mMode == EnergyMode::A3) energy_step += (actLevels.array() * actLevels.array() * actLevels.array()).sum();
	else if (mMode == EnergyMode::MA3) energy_step += (mMuscleMass.array() * actLevels.array() * actLevels.array() * actLevels.array()).sum();
	else if (mMode == EnergyMode::A) energy_step += (actLevels.array()).sum();
	else if (mMode == EnergyMode::MA) energy_step += (mMuscleMass.array() * actLevels.array()).sum();
	else if (mMode == EnergyMode::M2A) energy_step += (mMuscleMass.array() * mMuscleMass.array() * actLevels.array()).sum();
	else if (mMode == EnergyMode::M2A2) energy_step += (mMuscleMass.array() * mMuscleMass.array() * actLevels.array() * actLevels.array()).sum();
	else if (mMode == EnergyMode::M2A3) energy_step += (mMuscleMass.array() * mMuscleMass.array() * actLevels.array() * actLevels.array() * actLevels.array()).sum();
	else if (mMode == EnergyMode::M05A) energy_step += (mMuscleMass05.array() * actLevels.array()).sum();
	else if (mMode == EnergyMode::M05A2) energy_step += (mMuscleMass05.array() * actLevels.array() * actLevels.array()).sum();
	else if (mMode == EnergyMode::M05A3) energy_step += (mMuscleMass05.array() * actLevels.array() * actLevels.array() * actLevels.array()).sum();
	else if (mMode == EnergyMode::A15) energy_step += (actLevels.array() * actLevels.array().sqrt()).sum();
	else if (mMode == EnergyMode::M05A15) energy_step += (mMuscleMass05.array() * actLevels.array() * actLevels.array().sqrt()).sum();
	else if (mMode == EnergyMode::MA15) energy_step += (mMuscleMass.array() * actLevels.array() * actLevels.array().sqrt()).sum();
	else if (mMode == EnergyMode::M2A15) energy_step += (mMuscleMass.array() * mMuscleMass.array() * actLevels.array() * actLevels.array().sqrt()).sum();
	else if (mMode == EnergyMode::A125) energy_step += (actLevels.array() * actLevels.array().pow(0.25)).sum();
	else if (mMode == EnergyMode::M05A125) energy_step += (mMuscleMass05.array() * actLevels.array() * actLevels.array().pow(0.25)).sum();
	else if (mMode == EnergyMode::MA125) energy_step += (mMuscleMass.array() * actLevels.array() * actLevels.array().pow(0.25)).sum();
	else if (mMode == EnergyMode::M2A125) energy_step += (mMuscleMass.array() * mMuscleMass.array() * actLevels.array() * actLevels.array().pow(0.25)).sum();
	// else if (mMode == EnergyMode::MA2 || mMode == EnergyMode::MA2COT || mMode == EnergyMode::MA2COT2) energy_step += (mMuscleMass.array() * mEnergyWeight.array() * actLevels.array() * actLevels.array()).sum();
	// else if (mMode == EnergyMode::A2ANK) {
		// energy_step += (mEnergyWeight.array() * actLevels.array() * actLevels.array()).sum();
		// energy_step += mTargetCoeff * (mTargetWeight.array() * actLevels.array() * actLevels.array()).sum();
	// }
	// else if (mMode == EnergyMode::MA2ANK) {
		// energy_step += (mMuscleMass.array() * mEnergyWeight.array() * actLevels.array() * actLevels.array()).sum();
		// energy_step += mTargetCoeff * (mMuscleMass.array() * mTargetWeight.array() * actLevels.array() * actLevels.array()).sum();
	// }
	// else if (mMode == EnergyMode::A4) energy_step += (mEnergyWeight.array() * actLevels.array() * actLevels.array() * actLevels.array() * actLevels.array()).sum();
	else if (mMode == EnergyMode::BHAR) {
		cout << "BHAR mode uses AccumBHAR method instead of activation-based calculation" << endl;
		return;
	}
	else cout << "[Warning] Invalid energy mode " << static_cast<int>(mMode) << endl;

	double current_energy = energy_step * mActRewCoeff;
	mEnergyAvg.update(current_energy);
	mActAccum += current_energy;
	if (pGraphData != nullptr && pGraphData->key_exists("energy_muscle")) pGraphData->push("energy_muscle", current_energy);
	if (pGraphData != nullptr && pGraphData->key_exists("energy_avg")) pGraphData->push("energy_avg", mEnergyAvg.get());
}

void Energy::AccumTorque(Eigen::VectorXd torque, const Eigen::VectorXd& vels, double accum_divisor, CBufferData* pGraphData) {
	if (mTorqueRewCoeff < 0.0 && pGraphData == nullptr) return; // if torque reward is negative, don't accumulate torque

	mAccumDivisor += accum_divisor;
	
	if (mJntType == JntType::Torque) {
		torque.head(6).setZero();
	} else if (mJntType == JntType::FullMuscle) {
		cout << "Accumulating torque for on Energy::mJntType = " << static_cast<int>(mJntType) << endl;
		exit(-1);
	} else if (mJntType == JntType::LowerMuscle) {
		torque.head(24).setZero();
	}

	double energy_step = 0.0;
	if (mMode == EnergyMode::TAU) energy_step += torque.array().abs().sum();
	else if (mMode == EnergyMode::POWER) energy_step += abs(torque.dot(vels));
	else if (mJntType == JntType::LowerMuscle) {
		// energy_step += torque.array().abs().sum() / 100;
		energy_step += abs(torque.dot(vels));
	} else {
		cerr << "[Energy] Invalid energy mode for torque skeleton" << static_cast<int>(mMode) << endl;
		exit(-1);
	}

	if (mTorqueRewCoeff > 0.0) mTorqueAccum += energy_step * mTorqueRewCoeff;
	if (pGraphData != nullptr && pGraphData->key_exists("energy_tau")) pGraphData->push("energy_tau", energy_step);
}

void Energy::AccumBHAR(double accum_divisor, CBufferData* pGraphData) {
	if (mJntType == JntType::Torque) {
		cout << "Accumulating BHAR for on Energy::mJntType = " << static_cast<int>(mJntType) << endl;
		exit(-1);
	}

	mAccumDivisor += accum_divisor;
	double energy_step = 0.0;
	
	for (auto muscle : mMuscles) {
		double rate = muscle->RateBhar04();
		energy_step += rate;
		// cout << "muscle: " << muscle->GetName() << " rate: " << rate << endl;
	}

	double current_energy = energy_step * mActRewCoeff;
	mEnergyAvg.update(current_energy);
	mActAccum += current_energy;
	if (pGraphData != nullptr && pGraphData->key_exists("energy_muscle")) pGraphData->push("energy_muscle", current_energy);
	if (pGraphData != nullptr && pGraphData->key_exists("energy_avg")) pGraphData->push("energy_avg", mEnergyAvg.get());
}

// void Energy::HeelStrikeCb() {
// 	if (mMode == EnergyMode::MA2COT2) {
// 		cout << "Since act and torque energy are separated, this type of energy is deprecated." << endl;
// 		exit(-1);

// 		// mEnergyCurrent = mEnergyAccum / mAccumDivisor;
// 		// _reset_accumulators();
// 	}
// }

void Energy::Reset() {
	_reset_accumulators();
	mActReward = 0.0;
	mTorqueReward = 0.0;
	mEnergyAvg.reset();
}

void Energy::_reset_accumulators() {
	mActAccum = 0.0;
	mTorqueAccum = 0.0;
	mTargetAccum = 0.0;
	mAccumDivisor = 0.0;
}

double Energy::MetabolicReward() {
	double reward;
	// if (mMode != EnergyMode::MA2COT2) {
		if (mAccumDivisor < 1e-6) { cout << "accum_divisor is too small. got: " << mAccumDivisor << endl; return 0.0;}
		double act_current = mActAccum / mAccumDivisor;
		double torque_current = mTorqueAccum / mAccumDivisor;
		mActReward = _reward_from_energy(act_current);
		mTorqueReward = _reward_from_energy(torque_current);
		reward = mActReward * mTorqueReward;
		if (mActReward < 0.0 && mTorqueReward < 0.0) reward = - reward;
		_reset_accumulators();
	// }
	return reward;
}

double Energy::_reward_from_energy(double energy) const{
	double reward = 0.0;
	if (mEnergyRewardCurve == EnergyRewardCurve::EXP) reward = exp(- energy);
	else if (mEnergyRewardCurve == EnergyRewardCurve::LINEAR1) reward = 1 - energy / 5; // A single line segment which passes through (0, 1) and (5, 0)
	else if (mEnergyRewardCurve == EnergyRewardCurve::LINEAR2) {
		// Two line segments which pass through (0, 1), (4, 0.2) and (8, 0)
		if (energy < 4) reward = 1 - energy / 5;
		else reward = 0.4 - energy / 20;
	}else if (mEnergyRewardCurve == EnergyRewardCurve::LINEAR3) {
		// Two line segments which pass through (0, 1), (4, 0.5) and (20, 0)
		if (energy < 4) reward = 1 - 0.125 * energy;
		else reward = 0.625 - 0.03125 * energy;
	}else if (mEnergyRewardCurve == EnergyRewardCurve::LINEAR4) {
		// Two line segments which pass through (0, 1), (1.0, 0.1) and (4, 0)
		if (energy < 1.0) reward = 1 - energy;
		else reward = (4 - energy) / 30;
  }else {
		cerr << "[Energy] Invalid energy reward curve " << static_cast<int>(mEnergyRewardCurve) << endl;
	}
	if (energy < 0.0) reward = 0.1 * energy;
	return reward;
}

void Energy::LoadMuscles(const std::vector<Muscle*> muscles) {
	mMuscles = muscles;
	mMuscleMass = Eigen::VectorXd::Zero(mMuscles.size());
	mMuscleMass05 = Eigen::VectorXd::Zero(mMuscles.size());
	for(int i = 0; i < mMuscles.size(); i++) {
		mMuscleMass[i] = mMuscles[i]->GetMass();
		mMuscleMass05[i] = sqrt(mMuscles[i]->GetMass());
	}
	_update_weights();
}

void Energy::_update_weights() {
	mEnergyWeight = Eigen::VectorXd::Ones(mMuscles.size());
	mTargetWeight = Eigen::VectorXd::Zero(mMuscles.size());

	// if(mMode == EnergyMode::A2ANK || mMode == EnergyMode::MA2ANK) {
		for(int i = 0; i < mMuscles.size(); i++) {
			const auto muscle_type = mMuscles[i]->getType();
			if(muscle_type.has(MuscleType::ae)) {mEnergyWeight[i] = 0.0; mTargetWeight[i] = 1.0;}
		}
	// }
}

void Energy::LoadYaml(const YAML::Node& node) {
	try {
		mTargetCoeff = node["target_coeff"] ? node["target_coeff"].as<double>() : mTargetCoeff;
		mActRewCoeff = node["act_coeff"].as<double>();
		mTorqueRewCoeff = node["torque_coeff"].as<double>();
		mMode = static_cast<EnergyMode>(node["mode"].as<int>());
		mEnergyRewardCurve = static_cast<EnergyRewardCurve>(node["curve"].as<int>());
	}
	catch(const std::exception& e) {
		std::cerr << "[Energy] Error loading energy yaml: " << e.what() << std::endl;
	}
}

}
