
#include <iostream>
#include "core/Device.h"
#include "core/Utils.h"
#include "core/Record.h"
using namespace std;
using namespace dart;

namespace MASS
{

Device::Device()
: 
// mPowerPosR(1.0), mPowerNegR(1.0), mPowerPosL(1.0), mPowerNegL(1.0),
mPowerPosR(0.001), mPowerNegR(0.001), mPowerPosL(0.001), mPowerNegL(0.001),
  mQr(0.05), mQl(0.05), mTr(1.0), mTl(1.0){
    mK_ = (mK_scaler / 2);
    mKl_ = (mK_scaler / 2);
    mKr_ = (mK_scaler / 2);
}

Device::~Device() {}

void Device::LoadYaml(const YAML::Node& node){
    try {
        YAML::Node state = node["state"];
        mUseState = state["include"].as<bool>();
        mStateOfs = state["offset"].as<int>();
        mStateHist = state["history"].as<int>();
        mLongState = state["append_long_horizon"].as<bool>();
        mStateParam = state["include_param"] ? state["include_param"].as<bool>() : mStateParam;
        mZeroState = state["zero"] ? state["zero"].as<bool>() : mZeroState;

        YAML::Node direction = state["direction"];
        mAngleType = static_cast<AngleType>(direction["angle"].as<int>());
        mTorqueDir = static_cast<int>(direction["torque"].as<int>());

        YAML::Node coupling = node["coupling"];
        mStiff = coupling["stiff"].as<double>();
        mDamp = coupling["damp"].as<double>();
//        mVirtualCoupling = coupling["virtual"].as<bool>();
    }
    catch(const std::exception& e) {
        std::cerr << "[Device] Error loading device yaml: " << e.what() << std::endl;
    }
}

void Device::AddConstraint(){
    const auto skel_char = mCharacter->GetSkeleton();
    const double ts = mSkeleton->getTimeStep();
    // Special constant related to the LCP solver https://www.ode.org/ode-latest-userguide.html#sec_3_8_0
    const double erp = ts * mStiff / (ts * mStiff + mDamp);
    const double cfm = 1 / (ts * mStiff + mDamp);
    mCoupRoot = make_shared<constraint::WeldJointConstraint>(skel_char->getBodyNode("Pelvis"), mSkeleton->getBodyNode("Controller"));
    mCoupR = make_shared<constraint::WeldJointConstraint>(skel_char->getBodyNode("FemurR"), mSkeleton->getBodyNode("FastenerRightBack"));
    mCoupL = make_shared<constraint::WeldJointConstraint>(skel_char->getBodyNode("FemurL"), mSkeleton->getBodyNode("FastenerLeftBack"));
    for (auto& c : {mCoupRoot, mCoupR, mCoupL}) {
        if (c != nullptr) {
            c->setConstraintForceMixing(cfm);
            c->setErrorReductionParameter(erp);
            c->setErrorAllowance(10.0);
            c->setMaxErrorReductionVelocity(10.0);
            mWorld->getConstraintSolver()->addConstraint(c);
        }
    }

    for (auto& bn: mSkeleton->getBodyNodes()) bn->setCollidable(false);
}

void Device::RemoveConstraint(){
    for (auto& c : {mCoupRoot, mCoupR, mCoupL}) {
        if (c != nullptr) mWorld->getConstraintSolver()->removeConstraint(c);
    }
}

void Device::LoadSkeleton(const std::string& path, bool load_obj, double damping)
{
	mSkeleton = BuildFromFile(path, load_obj, damping);
	mStdSkeleton = BuildFromFile(path, load_obj, damping);
    mSkeleton->setSelfCollisionCheck(false);
}

void Device::Initialize()
{
    if(mSkeleton == nullptr){
        cerr << "Device Skeleton Null" << endl;
        exit(1);
    }
    mWorld->addSkeleton(mSkeleton);
    mSkeleton->setSelfCollisionCheck(false);
    InitializeMass();
    mDelta_t_idx = (int)(mDelta_t * mControlHz);
    Reset();
}

void Device::InitializeMass(){
    Eigen::Vector3d totalCOM = Eigen::Vector3d::Zero();
    Eigen::Matrix3d totalMoment = Eigen::Matrix3d::Zero();
    auto bds = mSkeleton->getBodyNodes();
    for (const auto& b : bds) {
        // Get mass
        double mass = b->getMass();
        double zero_mass = mass * 1e-4;
        mDefaultMass[b->getName()] = mass;
        mZeroMass[b->getName()] = zero_mass;
        mTotalMass += mass;
        mTotalZeroMass += zero_mass;

        // Get COM and inertia matrix of the body
        Eigen::Vector3d com = b->getLocalCOM();
        Eigen::Matrix3d mmt = b->getInertia().getMoment();
        mDefaultMoment[b->getName()] = mmt;
        b->eachShapeNodeWith<DynamicsAspect>([&](ShapeNode* sn) {
            mZeroMoment[b->getName()] = sn->getShape()->computeInertia(zero_mass);
        });

        // Weighted average for total COM
        totalCOM = (totalCOM * (mTotalMass - mass) + mass * com) / mTotalMass;

        // Parallel Axis Theorem adjustment for moment of inertia
        Eigen::Matrix3d comOffset = mass * (com * com.transpose() - com.squaredNorm() * Eigen::Matrix3d::Identity());
        totalMoment += mmt + comOffset;
    }
    // Set the total inertia
    Inertia totalInertia = Inertia(mTotalMass, totalCOM, totalMoment);
}

void Device::Reset()
{
    mSkeleton->clearConstraintImpulses();
    mSkeleton->clearInternalForces();
    mSkeleton->clearExternalForces();

    SkeletonPtr skel_char = mCharacter->GetSkeleton();
    const int dof_n = mSkeleton->getNumDofs();
    Eigen::VectorXd p(dof_n);
    Eigen::VectorXd v(dof_n);
    p.head(6) = skel_char->getPositions().head(6);
    v.head(6) = skel_char->getVelocities().head(6);
    p.segment<3>(6) = skel_char->getJoint("FemurR")->getPositions();
    p.segment<3>(9) = skel_char->getJoint("FemurL")->getPositions();
    v.segment<3>(6) = skel_char->getJoint("FemurR")->getVelocities();
    v.segment<3>(9) = skel_char->getJoint("FemurL")->getVelocities();
    Utils::setSkelPosAndVel(mSkeleton, p, v);
    UpdateWeight();

    counter = 0; mPosAssist = false;
	mControlSignals = deque<double>(1200, 0); mDesiredTorquesR = deque<double>(1200, 0); mDesiredTorquesL = deque<double>(1200, 0);
    mTorqueSignChangeIdx.clear();

    mPowerPosR.reset(); mPowerNegR.reset(); mPowerPosL.reset(); mPowerNegL.reset();
    mQr.reset(); mQl.reset(); mTr.reset(); mTl.reset();
    mQrRaw = 0.0; mQlRaw = 0.0; mPowerR = 0.0; mPowerL = 0.0;
}

double Device::GetRMSMomentR() const {
    double rms = 0.0;
    if (mTorqueSignChangeIdx.size() < 2) return 0.0;
    const int lower_idx = mTorqueSignChangeIdx.front();
    const int upper_idx = mTorqueSignChangeIdx.back();
    for (int i = lower_idx; i <= upper_idx; i++) {
        rms += mDesiredTorquesR[i] * mDesiredTorquesR[i];
    }
    rms /= (upper_idx - lower_idx + 1);
    return sqrt(rms);
}

void Device::Step(bool update_torque)
{
    if (update_torque) {
        if (counter == 0) { UpdateDesiredTorques(); counter+=1; }
        else counter = (counter + 1) % (mSimulationHz / mControlHz);
    }

    double tr = mDesiredTorquesR.front(); double tl = mDesiredTorquesL.front();
    if (mVirtualCoupling) {
        Eigen::VectorXd tau = Eigen::VectorXd::Zero(mCharacter->GetSkeleton()->getNumDofs());
        tau[mRightCharDofIdx] = tr; tau[mLeftCharDofIdx] = tl;
        if (!mFakeAssist) mCharacter->GetSkeleton()->setForces(tau);
    } else {
        Eigen::VectorXd tau = Eigen::VectorXd::Zero(mSkeleton->getNumDofs());
        tau[mRightTauDofIdx] = tr; tau[mLeftTauDofIdx] = tl;
        if (!mFakeAssist) mSkeleton->setForces(tau);
    }

    Eigen::VectorXd vel = mSkeleton->getVelocities();
    mPowerR = tr * vel[mRightTauDofIdx];
    mPowerL = tl * vel[mLeftTauDofIdx];
    if (mPowerR > 0) mPowerPosR.update(mPowerR);
    else mPowerNegR.update(mPowerR);
    if (mPowerL > 0) mPowerPosL.update(mPowerL);
    else mPowerNegL.update(mPowerL);
}

void Device::AddAngle(unsigned int mEpisodeSubStepCount, Record* pRecord, CBufferData* pGraphData) {
    if (pRecord != nullptr) {
        queue<pair<string, double>> record_buf;
        record_buf.emplace("dev_angleR", mQr.get());
        pRecord->add(mEpisodeSubStepCount, record_buf);
    }
    if (pGraphData != nullptr) {
        pGraphData->push("dev_angleR", mQr.get() * 180 / M_PI);
        pGraphData->push("dev_angleR_raw", mQrRaw * 180 / M_PI);
    }
}

void Device::AddVelocity(unsigned int mEpisodeSubStepCount, Record* pRecord, CBufferData* pGraphData)
{
    Eigen::VectorXd vel = mSkeleton->getVelocities();
    if (pRecord != nullptr) {
        queue<pair<string, double>> record_buf;
        record_buf.emplace("dev_velocity_HipR", vel[mRightTauDofIdx]);
        pRecord->add(mEpisodeSubStepCount, record_buf);
    }
    if (pGraphData != nullptr) pGraphData->push("dev_velocity", vel[mRightTauDofIdx]);
}


void Device::AddMoment(unsigned int mEpisodeSubStepCount, Record* pRecord, CBufferData* pGraphData)
{
    double tr = mDesiredTorquesR.front();
    if (pRecord != nullptr) {
        queue<pair<string, double>> record_buf;
        record_buf.emplace("dev_moment_HipR", tr);
        pRecord->add(mEpisodeSubStepCount, record_buf);
    }
    if(pGraphData != nullptr) pGraphData->push("dev_assist_raw", tr);
    if(pGraphData != nullptr) pGraphData->push("dev_assist_bw", -tr / mCharacter->GetSkeleton()->getMass());
}


void Device::AddPower(unsigned int mEpisodeSubStepCount, Record* pRecord, CBufferData* pGraphData)
{
    if (pRecord == nullptr && pGraphData == nullptr) return;

    if (pRecord != nullptr) {
        queue<pair<string, double>> record_buf;
        record_buf.emplace("dev_power_HipR", mPowerR);
        pRecord->add(mEpisodeSubStepCount, record_buf);
    }

    if(pGraphData != nullptr) {
        const double weightInv = 1.0 / mCharacter->GetSkeleton()->getMass();
        pGraphData->push("dev_power", mPowerR);
        pGraphData->push("power_dev_bw", mPowerR * weightInv);
        // pGraphData->push("power_pos", GetPosPowerR());
        // pGraphData->push("power_neg", GetNegPowerR());
        const double pos_power = mPowerR > 0 ? mPowerR : 0;
        const double neg_power = mPowerR < 0 ? mPowerR : 0;
        pGraphData->push("power_pos", pos_power);
        pGraphData->push("power_neg", neg_power);
    }
}


Eigen::VectorXd Device::GetState(bool mirror) const
{
    Eigen::VectorXd state = BuildState(mirror, mStateOfs, mStateHist);
    if (mStateParam && !mZeroState) {
        state[2 * mStateHist - 1] = mK_;
        state[2 * mStateHist - 2] = mDelta_t;
    }
    if (mLongState) {
        Eigen::VectorXd long_state = BuildState(mirror, mStateOfs * 16, mStateHist);
        state.conservativeResize(state.rows() + long_state.rows());
        state.tail(long_state.rows()) = long_state;
    }
	return state;
}

Eigen::VectorXd Device::BuildState(bool mirror, int offset, int history) const
{
    Eigen::VectorXd state = Eigen::VectorXd::Zero(2 * history);
    for(int i=0; i<history; i++){
        const double devTorqueR = mDesiredTorquesR.at(mDelta_t_idx + i*offset)/(mK_scaler / 2);
        const double devTorqueL = mDesiredTorquesL.at(mDelta_t_idx + i*offset)/(mK_scaler / 2);
        if (mZeroState) {
            state[0*history+i] = 0;
            state[1*history+i] = 0;
        } else if (mirror) {
            state[0*history+i] = devTorqueL;
            state[1*history+i] = devTorqueR;
        } else {
            state[0*history+i] = devTorqueR;
            state[1*history+i] = devTorqueL;
        }
    }
    return state;
}

void Device::ToggleZeroState(){
	if(mZeroState) mZeroState = false;
	else mZeroState = true;
}

Eigen::VectorXd Device::SuperTorque(const Eigen::VectorXd& spd_output){
    Eigen::VectorXd super_torque = Eigen::VectorXd::Zero(spd_output.size());
    mTr.update(spd_output[mRightCharDofIdx] * mKr_ / mK_scaler); mTl.update(spd_output[mLeftCharDofIdx] * mKl_ / mK_scaler);
    double tr = mTr.get(); double tl = mTl.get();
    super_torque[mRightCharDofIdx] = tr; super_torque[mLeftCharDofIdx] = tl;
    mDesiredTorquesR.push_front(tr); mDesiredTorquesL.push_front(tl);
    return super_torque;
}

void Device::UpdateDesiredTorques()
{
    if (mAngleType==AngleType::Hip) {mQrRaw = GetHipAngle("FemurR"); mQlRaw = GetHipAngle("FemurL");}
    else if (mAngleType==AngleType::HipSlope) {mQrRaw = GetHipSlopeAngle("FemurR"); mQlRaw = GetHipSlopeAngle("FemurL");}
    else if (mAngleType==AngleType::Dev) {mQrRaw = GetDevAngle("FemurR"); mQlRaw = GetDevAngle("FemurL");}
    else if (mAngleType==AngleType::Super) return;
    else {
        cout << "[Error] UpdateDesiredTorques: Invalid angle type " << static_cast<int>(mAngleType) << endl;
        exit(-1);
    }

    mQr.update(mQrRaw); mQl.update(mQlRaw);
	double y = sin(mQr.get()) - sin(mQl.get());
    mControlSignals.pop_back();
    mControlSignals.push_front(y);
    double tr = - mKr_ * mControlSignals.at(mDelta_t_idx);
    double tl = mKl_ * mControlSignals.at(mDelta_t_idx);
    mDesiredTorquesR.push_front(tr); mDesiredTorquesL.push_front(tl);
    mDesiredTorquesR.pop_back(); mDesiredTorquesL.pop_back();

    for (auto& idx : mTorqueSignChangeIdx) idx++;
    if (mTorqueSignChangeIdx.size() > 0 && mTorqueSignChangeIdx.back() >= mTorqueBufferSize) mTorqueSignChangeIdx.pop_back();
    if (tr > 0) {
        if (!mPosAssist) {
            mPosAssist = true;
            mTorqueSignChangeIdx.push_front(0);
        }
    } else if (tr < 0) {
        if (mPosAssist) {
            mPosAssist = false;
            mTorqueSignChangeIdx.push_front(0);
        }
    }
}

double Device::GetHipAngle(const std::string& name) const
{
    // // hip angle with respect to the pelvis (hip joint angle)
	// SkeletonPtr skel_char = mCharacter->GetSkeleton();
	// BodyNode* root = skel_char->getBodyNode("Pelvis");
    // BodyNode* femur = skel_char->getBodyNode(name);
             
    // Eigen::Isometry3d root_ = root->getTransform();
    // Eigen::Isometry3d root_inv = root_.inverse();
    // Eigen::Isometry3d trans_femur = femur->getTransform();
    
    // Eigen::Vector3d pos = (root_inv * trans_femur).translation();
	// double q = atan2(pos[2], -pos[1]);
	// q *= -1;    
    // return q;

    double angle = mCharacter->GetSkeleton()->getJoint(name)->getPositions()[0];
    return angle;
}

double Device::GetHipSlopeAngle(const std::string& name) const
{
    // hip angle with respect to the horizontal line
	SkeletonPtr skel_char = mCharacter->GetSkeleton();
	BodyNode* root = skel_char->getBodyNode("Pelvis");
    BodyNode* femur = skel_char->getBodyNode(name);

	Eigen::Isometry3d trans_femur = femur->getTransform();
    Eigen::Vector3d pos = trans_femur.translation();
    Eigen::Vector3d rootCOM = root->getCOM();
	
	double delta_y = pos[1] - rootCOM[1];
	double delta_z = pos[2] - rootCOM[2];
	double delta_norm = sqrt(delta_y*delta_y + delta_z*delta_z);
	double new_q = atan2(delta_z/delta_norm, -delta_y/delta_norm);
	new_q *= -1;

    return new_q;
}

double Device::GetDevAngle(const std::string& name) const
{
    // device hip angle
    double angle = 0.0;
    if (name == "FemurR") angle = mSkeleton->getJoint("RodRight")->getPositions()[0];
    else if (name == "FemurL") angle = mSkeleton->getJoint("RodLeft")->getPositions()[0];
    else {
        cout << "[Warning] GetAngleQ2: Invalid body name " << name << endl;
        angle = 0;
    }
    return angle;
}

void Device::SetParam(double kl, double kr, double t)
{
	mKl_ = kl * mK_scaler;
	mKr_ = kr * mK_scaler;
    mK_ = (mKl_ + mKr_) / 2;
	mDelta_t = t;
	mDelta_t_idx = (int)(mDelta_t * mControlHz);
    UpdateWeight();
}

void Device::UpdateWeight()
{
    if((IsOn() && mWeightEnabled) || mWeightForced) SetMassDefault();
    else SetMassZero();
}

void Device::SetMassZero()
{
    if (!Utils::close(mTotalZeroMass, mSkeleton->getMass())){
        for(auto& b : mSkeleton->getBodyNodes()){
            Inertia inertia(mZeroMass[b->getName()], Eigen::Vector3d::Zero(), mZeroMoment[b->getName()]);
            b->setInertia(inertia);
        }
    }
}

void Device::SetMassDefault()
{
    if (!Utils::close(mTotalMass * mWeightScaler, mSkeleton->getMass())){
        for(auto& b : mSkeleton->getBodyNodes()){
            Inertia inertia(mDefaultMass[b->getName()] * mWeightScaler, Eigen::Vector3d::Zero(), mDefaultMoment[b->getName()] * mWeightScaler);
            b->setInertia(inertia);
        }
    }
}

static std::tuple<Eigen::Vector3d, double, double, double> UnfoldModifyInfo(const ModifyDeviceInfo &info){
	return std::make_tuple(Eigen::Vector3d(info[0], info[1], info[2]), info[3], info[4], info[5]);
}

static Eigen::Isometry3d modifyIsometry3d(const Eigen::Isometry3d &iso, const ModifyDeviceInfo &info, int axis, bool rotate = true)
{
	Eigen::Vector3d l; 
	double s, t, m;
	std::tie(l, s, t, m) = UnfoldModifyInfo(info);
	double l0 = l[0];
	double l1 = l[1];
	double l2 = l[2];
	l[axis]       = l0;
	l[(axis+1)%3] = l1;
	l[(axis+2)%3] = l2;
	Eigen::Vector3d translation = iso.translation();
	translation = translation.cwiseProduct(l); 
	translation *= s;
	auto tmp = Eigen::Isometry3d(Eigen::Translation3d(translation));
	tmp.linear() = iso.linear();
	return tmp;
}

static void modifyShapeNode_(BodyNode* rtgBody, BodyNode* stdBody, const ModifyDeviceInfo &info, int axis){
	Eigen::Vector3d l; 
	double s, t, m;
	std::tie(l, s, t, m) = UnfoldModifyInfo(info);
	double l0 = l[0];
	double l1 = l[1];
	double l2 = l[2];
	l[axis]       = l0;
	l[(axis+1)%3] = l1;
	l[(axis+2)%3] = l2;
	double la = l[axis], lb = l[(axis+1)%3], lc = l[(axis+2)%3];

    for(int i = 0; i < rtgBody->getNumShapeNodes(); i++){
		ShapeNode* rtgShape = rtgBody->getShapeNode(i);
		ShapeNode* stdShape = stdBody->getShapeNode(i);
		ShapePtr newShape;
		if(auto rtg = std::dynamic_pointer_cast<CapsuleShape>(rtgShape->getShape()))
		{
			auto std = std::dynamic_pointer_cast<CapsuleShape>(stdShape->getShape());
			double radius = std->getRadius() * s * (lb+lc)/2, height = std->getHeight() * s * la;
			newShape = ShapePtr(new CapsuleShape(radius, height));
		}
		else if(auto rtg = std::dynamic_pointer_cast<SphereShape>(rtgShape->getShape()))
		{
			auto std = std::dynamic_pointer_cast<SphereShape>(stdShape->getShape());
			double radius = std->getRadius() * s * (la+lb+lc)/3;
			newShape = ShapePtr(new SphereShape(radius));
		}
		else if(auto rtg = std::dynamic_pointer_cast<CylinderShape>(rtgShape->getShape()))
		{
			auto std = std::dynamic_pointer_cast<CylinderShape>(stdShape->getShape());
			double radius = std->getRadius() * s * (lb+lc) / 2, height = std->getHeight() * s * la;
			newShape = ShapePtr(new CylinderShape(radius, height));
		}
		else if(std::dynamic_pointer_cast<BoxShape>(rtgShape->getShape()))
		{
			auto std = std::dynamic_pointer_cast<BoxShape>(stdShape->getShape());
			Eigen::Vector3d size = std->getSize() * s;
			size = size.cwiseProduct(l);
			newShape = ShapePtr(new BoxShape(size));
		}
		else if(auto rtg = std::dynamic_pointer_cast<MeshShape>(rtgShape->getShape()))
		{
			auto std = std::dynamic_pointer_cast<MeshShape>(stdShape->getShape());
			Eigen::Vector3d scale = std->getScale(); 
			scale *= s; 
			scale = scale.cwiseProduct(l);
			rtg->setScale(scale);
			Eigen::Isometry3d s = stdShape->getRelativeTransform(), r = modifyIsometry3d(s.inverse(), info, axis).inverse();
			rtgShape->setRelativeTransform(r);
			newShape = rtg;
		}
		rtgShape->setShape(newShape);
	}
	// double mass = stdBody->getMass() * l[0] * l[1] * l[2] * s * s * s;
	double mass = stdBody->getMass();
	Inertia inertia;
	inertia.setMass(mass);
	rtgBody->eachShapeNodeWith<DynamicsAspect>([&inertia, mass](ShapeNode* sn) {
		inertia.setMoment(sn->getShape()->computeInertia(mass));
		return false;
	});
	rtgBody->setInertia(inertia);
}

std::vector<BoneDeviceInfo> Device::LoadSkelParamFile(const std::string &filename)
{
	std::vector<BoneDeviceInfo> infos;
	tinyxml2::XMLDocument doc;
	if(doc.LoadFile(filename.c_str())){
		std::cout << "Can't open/parse file : " << filename << std::endl;
		throw std::invalid_argument("In func ModifyInfo");
	}
	
	tinyxml2::XMLElement *infoxml = doc.FirstChildElement("ModifyDeviceInfo");
	for(tinyxml2::XMLElement *boneXML = infoxml->FirstChildElement("Bone"); boneXML; boneXML = boneXML->NextSiblingElement("Bone")){
		std::string body = std::string(boneXML->Attribute("body"));
		std::stringstream ss(boneXML->Attribute("info"));
		ModifyDeviceInfo info; 
		for(int i = 0; i < 5; i++)  ss >> info[i];
		infos.emplace_back(body, info);
	}
	
	return infos;
}

std::map<std::string, int> skeletonDeviceAxis =
    {
            {"Controller", 1},
            {"ControllerRight", 1},
            {"ControllerLeft", 1},
            {"RodRight", 1},
            {"RodLeft", 1},
            {"FastenerRightOut", 2},
            {"FastenerRightFront", 1},
            {"FastenerRightBack", 1},
            {"FastenerLeftOut", 2},
            {"FastenerLeftFront", 1},
            {"FastenerLeftBack", 1},
    };

void Device::ModifySkeletonBodyNode(const std::vector<BoneDeviceInfo> &info, SkeletonPtr skel)
{
	for(auto bone : info) 
	{
		std::string name; 
		ModifyDeviceInfo info;
		std::tie(name, info) = bone;
		int axis = skeletonDeviceAxis[name];
		BodyNode* rtgBody = skel->getBodyNode(name);
		BodyNode* stdBody = mStdSkeleton->getBodyNode(name);

		if(rtgBody == NULL) continue;
        modifyShapeNode_(rtgBody, stdBody, info, axis);

		if(Joint* rtgParent = rtgBody->getParentJoint())
		{
			Joint* stdParent = stdBody->getParentJoint();
			Eigen::Isometry3d up = stdParent->getTransformFromChildBodyNode();
			rtgParent->setTransformFromChildBodyNode(modifyIsometry3d(up, info, axis));
		}

		for(int i = 0; i < rtgBody->getNumChildJoints(); i++)
		{
			Joint* rtgJoint = rtgBody->getChildJoint(i);
			Joint* stdJoint = stdBody->getChildJoint(i);
			Eigen::Isometry3d down = stdJoint->getTransformFromParentBodyNode();
			rtgJoint->setTransformFromParentBodyNode(modifyIsometry3d(down, info, axis, false));
		}
	}
}

void Device::ModifySkeletonLength(const std::vector<BoneDeviceInfo> &info, bool b)
{
	for(auto bone : info)
	{
		std::string name; 
		ModifyDeviceInfo modifyDeviceInfo;
		std::tie(name, modifyDeviceInfo) = bone;
	}
	ModifySkeletonBodyNode(info, mSkeleton);

	Eigen::VectorXd positions = mSkeleton->getPositions();
	Utils::setSkelPos(mSkeleton,Eigen::VectorXd::Zero(mSkeleton->getNumDofs()));
	Utils::setSkelPos(mStdSkeleton,Eigen::VectorXd::Zero(mSkeleton->getNumDofs()));

	positions[4] = mOffsetModify;

	Utils::setSkelPos(mSkeleton,positions);
	Utils::setSkelPos(mStdSkeleton,Eigen::VectorXd::Zero(mSkeleton->getNumDofs()));
}
}