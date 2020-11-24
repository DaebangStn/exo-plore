#include "core/Muscle.h"
#include "core/Utils.h"
#include <cmath>


using namespace std;

namespace MASS
{

const unordered_map<string, double> mass_multiplier_handsfield14 = 
{
    {"Gluteus_Maximus", 0.0006877582},
    {"Gluteus_Medius", 0.0003116271},
    {"Gluteus_Minimus", 0.0003322722},
    {"Bicep_Femoris_Longus", 0.0006000161},
    {"Semimembranosus", 0.0003829245},
    {"Semitendinosus", 0.0010971862},
    {"Rectus_Femoris", 0.0007841537},
    {"Vastus_Intermedius", 0.0009564668},
    {"Vastus_Lateralis", 0.0008635717},
    {"Vastus_Medialis", 0.0008802249},
    {"Soleus", 0.0003887523},
    {"Gastrocnemius_Lateral_Head", 0.0007591916},
    {"Gastrocnemius_Medial_Head", 0.0006260364},
    {"Tibialis_Posterior", 0.0003681490},
    {"Flexor_Hallucis", 0.0014364119},
    {"Flexor_Digitorum_Longus", 0.0006764145},
    {"Peroneus", 0.0003580696},
    {"iliacus", 0.0006779782},
    {"Psoas", 0.0006395856},
    {"Bicep_Femoris_Short", 0.0023078833},
    {"Tibialis_Anterior", 0.0006384236},
    {"Adductor_Magnus", 0.0006223110},
    {"Sartorius", 0.0033607449},
    {"Adductor_Longus", 0.0007935943},
    {"Adductor_Brevis", 0.0005371407},
    {"Gracilis", 0.0015576406},
    {"Pectineus", 0.0006019082},
    {"Tensor_Fascia_Lata", 0.0007580213},
    {"Obturator_Externus", 0.0021493792},
    {"Obturator_Internus", 0.0010336390},
    {"Piriformis", 0.0002992004},
    {"Quadratus_Femoris", 0.0001399649},
    {"Popliteus", 0.0013105995},
    {"Extensor_Digitorum_Longus", 0.0004052377},
    {"Extensor_Hallucis_Longus", 0.0004052377},
};


Anchor::Anchor(vector<BodyNode *> bns, vector<Eigen::Vector3d> lps, vector<double> ws)
	: bodynodes(bns), local_positions(lps), weights(ws) {
		if (bns.size() == 2) is_lbs = true;
		else if (bns.size() > 2) {
			cerr << "Anchor: " << bodynodes[0]->getName() << " has more than 2 body nodes" << endl;
			exit(1);
		}
	}

void Anchor::GetPoint(Eigen::Ref<Eigen::Vector3d> point) const
{
	if (is_lbs)
		point = weights[0] * (bodynodes[0]->getTransform() * local_positions[0]) + 
			weights[1] * (bodynodes[1]->getTransform() * local_positions[1]);
	else point = weights[0] * (bodynodes[0]->getTransform() * local_positions[0]);
}

void Anchor::GetPoint(Eigen::Ref<Eigen::Vector3d> local_pos, Eigen::Ref<Eigen::Vector3d> global_pos) const
{
	if (is_lbs) {
		global_pos = weights[0] * (bodynodes[0]->getTransform() * local_positions[0]) + 
			weights[1] * (bodynodes[1]->getTransform() * local_positions[1]);
		local_pos = bodynodes[0]->getTransform().inverse() * global_pos;
	}
	else {
		global_pos = bodynodes[0]->getTransform() * local_positions[0];
		local_pos = local_positions[0];
	}
}

void Muscle::setMuscleBunchType(MuscleType _muscleBunchType) {
	mMuscleBunchType.add(_muscleBunchType);
}

void Muscle::SetMassType(MassType mass_type) {
	mMassType = mass_type;

	if (mMassType == MassType::M_Handsfield14) {
		for (auto& muscle : mass_multiplier_handsfield14) {
			if (name.find(muscle.first) != string::npos) {
				mMassMultiplier = muscle.second;
				break;
			}
		}
	}else mMassMultiplier = mMassMultiplierInit;
}

Muscle::Muscle(string _name, double _f0, double _lm0, double _lt0, double _pen_angle, double lmax,
               double _type1_fraction, bool vel, double _len_ratio, int type, MassType mass_type, bool fast_mode)
	: mUseMuscleVelocity(vel), name(_name), f0_original(_f0), f0(_f0), l_m0(_lm0), l_t0(_lt0),
	  l_mt_max(lmax), type1_fraction(_type1_fraction), mLenRatio(_len_ratio), mType(type), mMassType(mass_type), mFastMode(fast_mode)
{
	l_m = l_mt - l_t0;

	SetMassType(mass_type);
	if(_name == "L_Bicep_Femoris_Longus" || _name == "R_Bicep_Femoris_Longus" || _name == "L_Bicep_Femoris_Short" || _name == "R_Bicep_Femoris_Short" || _name == "L_Bicep_Femoris_Short1" || _name == "R_Bicep_Femoris_Short1")
	{
		e_mo = 0.6;
	}
	else if (_name == "L_Gastrocnemius_Lateral_Head" || _name == "R_Gastrocnemius_Lateral_Head" || _name == "L_Gastrocnemius_Medial_Head" || _name == "R_Gastrocnemius_Medial_Head")
	{
		e_mo = 0.6;
	}
	else if (_name == "L_Gluteus_Maximus" || _name == "R_Gluteus_Maximus" || _name == "L_Gluteus_Maximus1" || _name == "R_Gluteus_Maximus1"
	      || _name == "L_Gluteus_Maximus2" || _name == "R_Gluteus_Maximus2" || _name == "L_Gluteus_Maximus3" || _name == "R_Gluteus_Maximus3"
	      || _name == "L_Gluteus_Maximus4" || _name == "R_Gluteus_Maximus4" || _name == "L_Gluteus_Medius" || _name == "R_Gluteus_Medius"
	      || _name == "L_Gluteus_Medius1" || _name == "R_Gluteus_Medius1" || _name == "L_Gluteus_Medius2" || _name == "R_Gluteus_Medius2"
	      || _name == "L_Gluteus_Medius3" || _name == "R_Gluteus_Medius3" || _name == "L_Gluteus_Minimus" || _name == "R_Gluteus_Minimus"
	      || _name == "L_Gluteus_Minimus1" || _name == "R_Gluteus_Minimus1" || _name == "L_Gluteus_Minimus2" || _name == "R_Gluteus_Minimus2")
	{
		e_mo = 0.6;
		type1_fraction = 0.45;
	}
	else if (_name == "L_Psoas_Major" || _name == "R_Psoas_Major" || _name == "L_Psoas_Major1" || _name == "R_Psoas_Major1")
	{
		e_mo = 0.6;
	}
	else if (_name == "L_Rectus_Femoris" || _name == "R_Rectus_Femoris" || _name == "L_Rectus_Femoris1" || _name == "R_Rectus_Femoris1")
	{
		e_mo = 1.0;
		k_pe = 9.0;
		type1_fraction = 0.35;
	}
	else if (_name == "L_Semimembranosus" || _name == "R_Semimembranosus" || _name == "L_Semimembranosus1" || _name == "R_Semimembranosus1")
	{
		e_mo = 0.8;
		type1_fraction = 0.65;
	}
	else if (_name == "L_Soleus" || _name == "R_Soleus" || _name == "L_Soleus1" || _name == "R_Soleus1")
	{
		e_mo = 0.6;
		type1_fraction = 0.8;
	}
	else if (_name == "L_Tibialis_Anterior" || _name == "R_Tibialis_Anterior" || _name == "L_Tibialis_Posterior" || _name == "R_Tibialis_Posterior")
	{
		e_mo = 0.6;
		type1_fraction = 0.75;
	}
	else if (_name == "L_Vastus_Intermedius" || _name == "R_Vastus_Intermedius" || _name == "L_Vastus_Intermedius1" || _name == "R_Vastus_Intermedius1"
	      || _name == "L_Vastus_Lateralis" || _name == "R_Vastus_Lateralis" || _name == "L_Vastus_Lateralis1" || _name == "R_Vastus_Lateralis1"
	      || _name == "L_Vastus_Medialis" || _name == "R_Vastus_Medialis" || _name == "L_Vastus_Medialis1" || _name == "R_Vastus_Medialis1"
	      || _name == "L_Vastus_Medialis2" || _name == "R_Vastus_Medialis2")
	{
		e_mo = 1.0;
		k_pe = 9.0;		
	}
}

double Muscle::_computeMass() const {
	double mass;
	if(mMassType == MassType::MT0) mass = f0 * l_mt0 * mMassMultiplier;
	else if (mMassType == MassType::M0 || mMassType == MassType::M_Handsfield14) mass = f0 * l_m0 * mMassMultiplier;
	else {
		cout << "Invalid mass type for muscle " << name << " : " << static_cast<int>(mMassType) << endl;
		mass = 0.0;
	}
	return mass * mMassRatio;
}

void Muscle::AddJointLbsAnchor(const SkeletonPtr &skel, const Eigen::Vector3d &glob_pos){
    vector<BodyNode*> lbs_body_nodes;
    vector<Eigen::Vector3d> lbs_local_positions;
    vector<double> lbs_weights;
    double total_weight = 0.0;

	// 1. Find the nearest joint
    vector<double> joint_distance;
    joint_distance.resize(skel->getNumBodyNodes(), 0.0);

    for (int i = 0; i < skel->getNumBodyNodes(); i++)
    {
		BodyNode* bn = skel->getBodyNode(i);
        Eigen::Isometry3d T = bn->getTransform() * bn->getParentJoint()->getTransformFromChildBodyNode();
        joint_distance[i] = (glob_pos - T.translation()).norm();
    }

    vector<int> index_sort_by_distance = Utils::sort_indices(joint_distance);

    int min_idx = index_sort_by_distance[0];
    double min_dist = joint_distance[min_idx];
	BodyNode* nearest_bn = skel->getBodyNode(min_idx);
	Eigen::Vector3d local_pos = nearest_bn->getTransform().inverse() * glob_pos;
	
	// 2. If the anchor is close to the joint, use the lbs weight of child and parent body node
    if (min_dist < 0.08 && nearest_bn->getParentBodyNode() != nullptr)
    {
		double body_distance = local_pos.norm();
		double lbs_weight = 1.0 / pow(body_distance + 1E-6, 2);

        total_weight += lbs_weight;
        lbs_weights.push_back(lbs_weight);
        lbs_body_nodes.push_back(nearest_bn);
        lbs_local_positions.push_back(local_pos);

		BodyNode* parent_bn = nearest_bn->getParentBodyNode();
		Eigen::Vector3d parent_local_pos = parent_bn->getTransform().inverse() * glob_pos;
		double parent_distance = parent_local_pos.norm();
		double parent_lbs_weight = 1.0 / pow(parent_distance + 1E-6, 2);

		total_weight += parent_lbs_weight;
		lbs_weights.push_back(parent_lbs_weight);
		lbs_body_nodes.push_back(parent_bn);
		lbs_local_positions.push_back(parent_local_pos);
	// 3. If the anchor is far from the joint, do not use the lbs
    } else {
        total_weight = 1.0;
        lbs_weights.push_back(1.0);
        lbs_body_nodes.push_back(nearest_bn);
        lbs_local_positions.push_back(local_pos);
    }

    for(int i=0; i<lbs_body_nodes.size(); i++) lbs_weights[i] /= total_weight;
    mAnchors.push_back(new Anchor(lbs_body_nodes, lbs_local_positions, lbs_weights));
}

void Muscle::AddPrevMeshLbsAnchor(const dart::dynamics::SkeletonPtr &skel, const Eigen::Vector3d &glob_pos)
{
	// 1. Find the nearest body node
	BodyNode* bn;

	// Weight Calculation Part
	vector<double> distance;
	vector<Eigen::Vector3d> local_positions;
	distance.resize(skel->getNumBodyNodes(), 0.0);
	local_positions.resize(skel->getNumBodyNodes());

	for (int i = 0; i < skel->getNumBodyNodes(); i++) { // Todo: compute distance with mesh after the joint distance is computed (do not compute whole mesh distance, only compute the nearest body)
		double min_distance = 1000000000.0;
		bn = skel->getBodyNodes()[i];
		bn->eachShapeNode([&min_distance, glob_pos](ShapeNode* sn) {
			if (sn->getShape()->is<MeshShape>()) {
				auto mesh = dynamic_pointer_cast<MeshShape>(sn->getShape())->getMesh()->mMeshes[0];
				for(int idx = 0; idx < mesh->mNumVertices; idx++) {
					Eigen::Vector3d pos = 0.01 * Eigen::Vector3d(
						mesh->mVertices[idx][0],
						mesh->mVertices[idx][1],
						mesh->mVertices[idx][2]);

					if((glob_pos - pos).norm() < min_distance) min_distance = (glob_pos - pos).norm();
				}
			}
		});
		local_positions[i] = skel->getBodyNode(i)->getTransform().inverse() * glob_pos;
		distance[i] = min_distance;
	}

	const auto point1 = chrono::high_resolution_clock::now();
	
	//Weight Calculation 
	vector<int> index_sort_by_distance = Utils::sort_indices(distance);
	vector<dart::dynamics::BodyNode*> lbs_body_nodes;
	vector<Eigen::Vector3d> lbs_local_positions;
	vector<double> lbs_weights;
    double total_weight = 0.0;
	bool isMult = false;
	int b1, b2;
	double min_dist = 1000.0;
	for(auto jn : skel->getJoints())
	{
		if(jn->getParentBodyNode() == nullptr) continue;
		Eigen::Vector3d joint_pos = jn->getChildBodyNode()->getTransform() * jn->getTransformFromChildBodyNode() * Eigen::Vector3d::Zero();
		if ((joint_pos - glob_pos).norm() < min_dist)
		{
			min_dist = (joint_pos - glob_pos).norm();
			b1 = jn->getChildBodyNode()->getIndexInSkeleton();
			b2 = jn->getParentBodyNode()->getIndexInSkeleton();
		}
	}
	
	if(min_dist < 0.08)
	{
		isMult = true;
		if (distance[b1] > distance[b2])
		{
			int tmp = b1;
			b1 = b2;
			b2 = tmp;
		}
	}	

	if(isMult){
		lbs_weights.push_back(1.0 / pow(distance[b1],2));
		lbs_weights.push_back(1.0 / pow(distance[b2],2));

		total_weight = lbs_weights[0] + lbs_weights[1];

		lbs_body_nodes.push_back(skel->getBodyNode(b1));
		lbs_body_nodes.push_back(skel->getBodyNode(b2));

		lbs_local_positions.push_back(local_positions[b1]);
		lbs_local_positions.push_back(local_positions[b2]);
		
	}else{ //Obviously Close
		bn = skel->getBodyNode(index_sort_by_distance[0]);
		// cout << "[DEBUG] Body Node : " << bn->getName() << " "<< endl;
		total_weight = 1.0;
		lbs_weights.push_back(1.0);
		lbs_body_nodes.push_back(bn);
		lbs_local_positions.push_back(bn->getTransform().inverse() * glob_pos);
	}

	for(int i=0; i<lbs_body_nodes.size(); i++) lbs_weights[i] /= total_weight;
	Anchor* anchor = new Anchor(lbs_body_nodes, lbs_local_positions, lbs_weights);
	mAnchors.push_back(anchor);
}

void Muscle::AddMeshLbsAnchor(const dart::dynamics::SkeletonPtr &skel, const Eigen::Vector3d &glob_pos)
{
	vector<BodyNode*> lbs_body_nodes;
    vector<Eigen::Vector3d> lbs_local_positions;
    vector<double> lbs_weights;
    double total_weight = 0.0;

    if(mFastMode){
        AddJointLbsAnchor(skel, glob_pos);
        return;
    }

	// 1. Find the nearest joint
    vector<double> joint_distance;
    joint_distance.resize(skel->getNumBodyNodes(), 0.0);

    for (int i = 0; i < skel->getNumBodyNodes(); i++)
    {
		BodyNode* bn = skel->getBodyNode(i);
        Eigen::Isometry3d T = bn->getTransform() * bn->getParentJoint()->getTransformFromChildBodyNode();
        joint_distance[i] = (glob_pos - T.translation()).norm();
    }

    vector<int> index_sort_by_distance = Utils::sort_indices(joint_distance);

    int min_idx = index_sort_by_distance[0];
    double min_dist = joint_distance[min_idx];
	// Todo: fix bn[0] points the nearest body node
	BodyNode* nearest_bn = skel->getBodyNode(min_idx);
	Eigen::Vector3d local_pos = nearest_bn->getTransform().inverse() * glob_pos;

	// 2. If the anchor is close to the joint, use the lbs weight of child and parent body node
	if (min_dist < 0.08 && nearest_bn->getParentBodyNode() != nullptr)
	{
		double body_distance = GetDistanceFromMesh(nearest_bn, glob_pos);
		double lbs_weight = 1.0 / pow(body_distance + 1E-6, 2);

		total_weight += lbs_weight;
		lbs_weights.push_back(lbs_weight);
		lbs_body_nodes.push_back(nearest_bn);
		lbs_local_positions.push_back(local_pos);

		BodyNode* parent_bn = nearest_bn->getParentBodyNode();
		Eigen::Vector3d parent_local_pos = parent_bn->getTransform().inverse() * glob_pos;
		double parent_distance = GetDistanceFromMesh(parent_bn, glob_pos);
		double parent_lbs_weight = 1.0 / pow(parent_distance + 1E-6, 2);

		total_weight += parent_lbs_weight;
		lbs_weights.push_back(parent_lbs_weight);
		lbs_body_nodes.push_back(parent_bn);
		lbs_local_positions.push_back(parent_local_pos);
	// 3. If the anchor is far from the joint, do not use the lbs
	} else {
		total_weight = 1.0;
		lbs_weights.push_back(1.0);
		lbs_body_nodes.push_back(nearest_bn);
		lbs_local_positions.push_back(local_pos);
	}

	for(int i=0; i<lbs_body_nodes.size(); i++) lbs_weights[i] /= total_weight;	
	mAnchors.push_back(new Anchor(lbs_body_nodes, lbs_local_positions, lbs_weights));
}

void Muscle::AddSingleBodyAnchor(BodyNode *bn, const Eigen::Vector3d &glob_pos)
{
	vector<BodyNode *> lbs_body_nodes;
	vector<Eigen::Vector3d> lbs_local_positions;
	vector<double> lbs_weights;

	lbs_body_nodes.push_back(bn);
	lbs_local_positions.push_back(bn->getTransform().inverse() * glob_pos);
	lbs_weights.push_back(1.0);

	mAnchors.push_back(new Anchor(lbs_body_nodes, lbs_local_positions, lbs_weights));
}

void Muscle::SetMuscle()
{
	mNumAnchors = mAnchors.size();
	mAnchorPos.resize(3, mNumAnchors);
	mAnchorPosLocal.resize(3, mNumAnchors);
	mAnchorVel.resize(3, mNumAnchors);
	mAnchorPosDif.resize(3, mNumAnchors - 1);
	mAnchorPosDir.resize(3, mNumAnchors - 1);
	mAnchorVelDif.resize(3, mNumAnchors - 1);
	mAnchorForceDir.resize(3, mNumAnchors);
	mAnchorForce.resize(3, mNumAnchors);
	
	mDofMap = std::vector<std::unordered_map<int, int>>(mNumAnchors);
	for (int i = 0; i < mNumAnchors; i++) {
		auto& indices = mAnchors[i]->bodynodes[0]->getDependentGenCoordIndices();
		for (int j = 0; j < indices.size(); j++) {
			mDofMap[i][indices[j]] = j;
		}
	}

	l_mt0 = 0;
	for(int i = 0; i < mNumAnchors; i++) {
        if (i > 0) {
			Eigen::Vector3d pos1, pos2;
			mAnchors[i]->GetPoint(pos1);
			mAnchors[i - 1]->GetPoint(pos2);
			l_mt0 += (pos1 - pos2).norm();
		}
		Eigen::Map<Eigen::VectorXd> weights(mAnchors[i]->weights.data(), mAnchors[i]->weights.size());
		mAnchorWeight.push_back(weights);
	}
	l_mt0_original = l_mt0;

	UpdateGeometry();
	SetActivation(0);
	mMass = _computeMass();

	Eigen::MatrixXd Jt;
	Eigen::VectorXd Fa, Fp;
	GetJacobianTranspose(Jt);
	GetForceJacobianAndPassive(Fa, Fp);
	Eigen::VectorXd Jta = Jt * Fa;
    mRelatedDofs = 0;
	related_dof_indices.clear();
	for (int i = 0; i < Jta.rows(); i++)
	{
		if (abs(Jta[i]) > 1E-6)
		{
			mRelatedDofs++;
			related_dof_indices.push_back(i);
		}
	}
	if(mRelatedDofs == 0) cout << "Muscle: " << name << " has no related dofs" << endl;
	mJRelated.resize(6, mRelatedDofs);
	mJRelatedLinear.resize(3, mRelatedDofs);
}

double Muscle::GetDistanceFromMesh(BodyNode* bn, const Eigen::Vector3d& glob_pos)
{
	double min_distance = 1000000000.0;
	bn->eachShapeNode([&min_distance, glob_pos](ShapeNode* sn) {
		if (sn->getShape()->is<MeshShape>()) {
			auto mesh = dynamic_pointer_cast<MeshShape>(sn->getShape())->getMesh()->mMeshes[0];
			for(int idx = 0; idx < mesh->mNumVertices; idx++) {
				Eigen::Vector3d pos = 0.01 * Eigen::Vector3d(
					mesh->mVertices[idx][0],
					mesh->mVertices[idx][1],
					mesh->mVertices[idx][2]);

				if((glob_pos - pos).norm() < min_distance) min_distance = (glob_pos - pos).norm();
			}
		}
	});
	return min_distance;
}

void Muscle::SetActivation(double a)
{
	activation = a;
	mFa = Getf_A();
	mPassiveForce = Getf_p();
	mForce = mFa * activation + mPassiveForce;
	mAnchorForce = mForce * mAnchorForceDir;
}

void Muscle::Reset()
{
	UpdateGeometry();
	SetActivation(0);
}

void Muscle::ApplyForceToBody()
{
	for(int i = 0; i < mNumAnchors; i++) {
		mAnchors[i]->bodynodes[0]->addExtForce(mAnchorForce.col(i), mAnchorPos.col(i), false, false);
	}
}


void Muscle::UpdateGeometry()
{
	// Anchor positions
	for(int i = 0; i < mNumAnchors; i++) {
		auto&& global_ref = mAnchorPos.col(i);
		auto&& local_ref = mAnchorPosLocal.col(i);
		mAnchors[i]->GetPoint(local_ref, global_ref);
	}
	mAnchorPosDif = mAnchorPos.rightCols(mNumAnchors - 1) - mAnchorPos.leftCols(mNumAnchors - 1);
	mAnchorPosDir = mAnchorPosDif.colwise().normalized();
	mAnchorForceDir.leftCols(1) = mAnchorPosDir.leftCols(1);
    mAnchorForceDir.rightCols(1) = -mAnchorPosDir.rightCols(1);
    if(mNumAnchors > 2) {
        mAnchorForceDir.middleCols(1, mNumAnchors-2) = mAnchorPosDir.middleCols(1, mNumAnchors-2) - mAnchorPosDir.leftCols(mNumAnchors-2);
    }

    Eigen::VectorXd lengths = mAnchorPosDif.colwise().norm();
	length = lengths.sum();
	l_mt = length / l_mt0;
	l_m = l_mt - l_t0;

	// Anchor velocities
    for(int i = 0; i < mNumAnchors; i++) {
        const int num_bn = mAnchors[i]->bodynodes.size();
        Eigen::Matrix3Xd velocities(3, num_bn);
        for(int j = 0; j < num_bn; j++) velocities.col(j) = mAnchors[i]->bodynodes[j]->getLinearVelocity(mAnchors[i]->local_positions[j]);
        mAnchorVel.col(i) = velocities * mAnchorWeight[i];
    }
    mAnchorVelDif = mAnchorVel.rightCols(mNumAnchors - 1) - mAnchorVel.leftCols(mNumAnchors - 1);

    v_m = mAnchorPosDif.cwiseProduct(mAnchorVelDif).colwise().sum().transpose().cwiseQuotient(lengths).sum();
    v_m = clamp(v_m, -0.15, 0.15);
}

double Muscle::Getl_ratio()
{
	double ratio = l_m/l_m0;
	if(ratio > mLenRatio) ratio = mLenRatio;
	return ratio;
}

double Muscle::Getf_A()
{
	double _g_av = 1.0;
	if(mUseMuscleVelocity){
		// A comparison among different Hill-type contractiondynamics formulations for muscle force estimation
		double v_max = (l_mt0*l_m0) * 10.0; 
		// double v_CE = v_m;
		// if(vType==0) v_CE = GetV_m();
		// else v_CE = Getdl_m();
		_g_av = g_av(v_m/v_max);			
	}	
	return f0 * g_al(l_m/l_m0) * _g_av;
}

double Muscle::Getf_p()
{
	double len_ratio = l_m/l_m0;
	// if(len_ratio > mLenRatio)
	// 	len_ratio = mLenRatio;
	return f0 * g_pl(len_ratio);
}

double Muscle::g(double _l_m)
{
	double e_t = (l_mt - _l_m - l_t0) / l_t0;
	_l_m = _l_m / l_m0;
	double f = g_t(e_t) - (g_pl(_l_m) + activation * g_al(_l_m));
	return f;
}

double Muscle::g_t(double e_t)
{
	double f_t;
	if (e_t <= e_t0) f_t = f_toe / (exp(k_toe) - 1) * (exp(k_toe * e_t / e_toe) - 1);
	else f_t = k_lin * (e_t - e_toe) + f_toe;
	return f_t;
}

double Muscle::g_pl(double _l_m)
{
	double f_pl = (exp(k_pe * (_l_m - 1) / e_mo) - 1.0) / (exp(k_pe) - 1.0);
	if (_l_m < 1.0) return 0.0;
	return f_pl;
}

double Muscle::g_al(double _l_m)
{
	return exp(-(_l_m - 1.0) * (_l_m - 1.0) / gamma);
}

double Muscle::g_av(double _v_m)
{
	double f_av = 0;

	double k_CE_1 = 0.25;
	double k_CE_2 = 0.06;
	double const_fv_max = 1.6;
	
	if(_v_m <= -1) f_av = 0;
	else if(-1<_v_m && _v_m<= 0) f_av = (1+_v_m) / (1-_v_m/k_CE_1);
	else f_av = (1+_v_m*const_fv_max/k_CE_2) / (1+_v_m/k_CE_2);

	return f_av;
}

void Muscle::GetReducedJacobianTranspose(Eigen::MatrixXd& Jt_reduced)
{
	const auto &skel = mAnchors[0]->bodynodes[0]->getSkeleton();
	Jt_reduced.resize(mRelatedDofs, 3 * mNumAnchors);

	for (int i = 0; i < mNumAnchors; i++)
	{
		const auto& bn = mAnchors[i]->bodynodes[0];
		const auto& jac = bn->getJacobian();
		mJRelated.setZero();
		for (int j = 0; j < mRelatedDofs; j++)
		{
			auto it = mDofMap[i].find(related_dof_indices[j]);
			if (it != mDofMap[i].end()) mJRelated.col(j) = jac.col(it->second);
		}
        mJRelatedLinear = mJRelated.bottomRows<3>() + mJRelated.topRows<3>().colwise().cross(mAnchorPosLocal.col(i));        
		const auto& R = bn->getTransform().linear();
        Jt_reduced.block(0, i * 3, mRelatedDofs, 3).noalias() = (R * mJRelatedLinear).transpose();
	}
}

void Muscle::GetJacobianTranspose(Eigen::MatrixXd& Jt) const
{
	const auto &skel = mAnchors[0]->bodynodes[0]->getSkeleton();
	int dof = skel->getNumDofs();
	Jt.resize(dof, 3 * mNumAnchors);
	for (int i = 0; i < mNumAnchors; i++) 
		Jt.block(0, i * 3, dof, 3) = skel->getLinearJacobian(mAnchors[i]->bodynodes[0], mAnchorPosLocal.col(i)).transpose();
}

void Muscle::GetForceJacobianAndPassive(Eigen::VectorXd& Fa, Eigen::VectorXd& Fp) const
{
    Eigen::Map<const Eigen::VectorXd> forceDirVec(mAnchorForceDir.data(), 3 * mNumAnchors);    
    Fa = forceDirVec * mFa;
    Fp = forceDirVec * mPassiveForce;
}

vector<Joint *> Muscle::GetRelatedJoints()
{
	auto skel = mAnchors[0]->bodynodes[0]->getSkeleton();
	map<Joint *, int> jns;
	vector<Joint *> jns_related;
	for (int i = 0; i < skel->getNumJoints(); i++)
		jns.insert(make_pair(skel->getJoint(i), 0));

	Eigen::VectorXd dl_dtheta = Getdl_dtheta();

	for (int i = 0; i < dl_dtheta.rows(); i++) if (abs(dl_dtheta[i]) > 1E-6) jns[skel->getDof(i)->getJoint()] += 1;
	for (auto jn : jns) if (jn.second > 0) jns_related.push_back(jn.first);
	return jns_related;
}

vector<BodyNode *> Muscle::GetRelatedBodyNodes()
{
	vector<BodyNode *> bns_related;
	auto rjs = GetRelatedJoints();
	for (auto joint : rjs) bns_related.push_back(joint->getChildBodyNode());

	return bns_related;
}

void Anchor::ComputeJacobian(const SkeletonPtr& skel, Eigen::MatrixXd& Jref) const
{
	// Jref.resize(3, skel->getNumDofs());
	// Jref.setZero();

	if (is_lbs) {
		Jref = skel->getLinearJacobian(bodynodes[0], local_positions[0]) * weights[0] + 
			skel->getLinearJacobian(bodynodes[1], local_positions[1]) * weights[1];
	}
	else Jref = skel->getLinearJacobian(bodynodes[0], local_positions[0]);
}

void Muscle::ComputeJacobians()
{
	const SkeletonPtr& skel = mAnchors[0]->bodynodes[0]->getSkeleton();
	mCachedJs.resize(mNumAnchors);
	for (int i = 0; i < mNumAnchors; i++) mAnchors[i]->ComputeJacobian(skel, mCachedJs[i]);
}

Eigen::VectorXd Muscle::Getdl_dtheta()
{
	ComputeJacobians();
	const auto &skel = mAnchors[0]->bodynodes[0]->getSkeleton();
	Eigen::VectorXd dl_dtheta(skel->getNumDofs());
	dl_dtheta.setZero();
	for (int i = 0; i < mNumAnchors - 1; i++)
	{
		Eigen::Vector3d pi = mAnchorPos.col(i + 1) - mAnchorPos.col(i);
		Eigen::MatrixXd dpi_dtheta = mCachedJs[i + 1] - mCachedJs[i];
		Eigen::VectorXd dli_d_theta = (dpi_dtheta.transpose() * pi) / (l_mt0 * pi.norm());
		dl_dtheta += dli_d_theta;
	}

	for (int i = 0; i < dl_dtheta.rows(); i++) if (abs(dl_dtheta[i]) < 1E-6) dl_dtheta[i] = 0.0;

	return dl_dtheta;
}

double Muscle::Getdl_velocity()
{
	ComputeJacobians();
	const auto &skel = mAnchors[0]->bodynodes[0]->getSkeleton();
	double dl_velocity = 0.0;
	for(int i = 0; i < mNumAnchors - 1; i++)
	{
		Eigen::Vector3d dist = mAnchorPos.col(i+1) - mAnchorPos.col(i);
		Eigen::Vector3d d_dist = (mCachedJs[i + 1] - mCachedJs[i]) * skel->getVelocities();
		dl_velocity += dist.dot(d_dist) / dist.norm();// ((mCachedJs[i + 1] - mCachedJs[i]) * skel->getVelocities()); 
	}
	return dl_velocity;
}

double Muscle::GetExcitation(int type)
{
	// double excitation = 0.0;
	// double tau_act = 0.015;
	// double tau_deact = 0.050;
	// double da_dt;
	
	// da_dt = (activation - activation_prev) / (1.0/(double)mSimulationHz);
	// if(da_dt > 0) excitation = da_dt*(tau_act*(0.5 + 1.5*activation)) + activation;
	// else excitation = da_dt*(tau_deact/(0.5 + 1.5*activation)) + activation;
	// excitation = Utils::clamp(excitation, 0, 1);
	return activation;
}

double Muscle::RateBhar04()
{
	// It assume that No Basal Metabolic Rate (W) (based on whole body mass, not muscle mass)
	double h_A, h_M, h_SL, w;
	h_A = RateBhar04_Activation();
	h_M = RateBhar04_Maintenance();
	h_SL = RateBhar04_Shortening();
	w = RateBhar04_MechWork();

	//Added code from Opensim BHAR 
	double totalHeatRate = 0;
    double Edot_W_beforeClamp = h_A + h_M + h_SL + w;
    if (Edot_W_beforeClamp < 0) h_SL -= Edot_W_beforeClamp;

    // This check is adapted from Umberger(2003), page 104: the total heat rate
    // (i.e., Adot + Mdot + Sdot) for a given muscle cannot fall below 1.0 W/kg.
    // -----------------------------------------------------------------------
    totalHeatRate = h_A + h_M + h_SL; // (W)
    if(totalHeatRate < mMass) {
		// cout << name << ": Total heat rate (" << totalHeatRate << ") is less than muscle mass (" << mMass << ")." << endl;
		// cout << "Activation: " << h_A << ", Maintenance: " << h_M << ", Shortening: " << h_SL << ", Mechanical Work: " << w << endl;
		totalHeatRate =  mMass; // not allowed to fall below 1.0 W.kg-1
	} else {
		// cout << "===================>" << name << ": Total heat rate (" << totalHeatRate << ") is greater than muscle mass (" << mMass << ")." << endl;
	}

	return totalHeatRate + w;
}

double Muscle::RateBhar04_Activation()
{
	double h_A;
	double u = activation; // excitation == stimulation
	// if(type == 0) u = this->GetExcitation(0);
	// else if(type == 1) u = this->GetExcitation(1);

	double A_f = 133; // activation heat rate. fast twitch muscle
	double A_s = 40; // activation heat rate. slow twitch muscle
	double u_f = 1-cos(M_PI/2.0 * u); // 1 - cos(PI/2 * u_t)
	double u_s = sin(M_PI/2.0 * u); // sin(PI/2 * u_t)
	double f_FT = 1-type1_fraction; // percentages of fast twitch muscles, default 0.5
	double f_ST = type1_fraction; // percentages of slow twitch muscles, default 0.5

	double coeff = (A_s*u_s*f_ST + A_f*u_f*f_FT);
	h_A = mMass * coeff;
	// cout << name << ": Coefficient: " << coeff << endl;
	return h_A;
}

double Muscle::RateBhar04_Maintenance()
{
	double h_M;
	double l_M = 0.0;
	double l_ce = l_m/l_m0;

	if(l_ce <= 0.5) l_M = 0.5;
	else if(l_ce <= 1.0) l_M = l_ce;
	else if(l_ce <= 1.5) l_M = -2.0 * l_ce + 3.0;
	else l_M = 0.0;

	double M_f = 111;
	double M_s = 74;
	double u = activation;
	double u_f = 1-cos(M_PI/2.0 * u);
	double u_s = sin(M_PI/2.0 * u);
	double f_FT = 1-type1_fraction;
	double f_ST = type1_fraction;

	double coeff = l_M * (M_s*u_s*f_ST + M_f*u_f*f_FT);
	h_M = mMass * coeff;
	// cout << name << ": Coefficient: " << coeff << endl;
	return h_M;
}

double Muscle::RateBhar04_Shortening()
{
	double h_SL;
	double alpha = 0.0;
	
	double v_CE = v_m;
	// if(vType==0) v_CE = this->GetV_m();
	// else v_CE = this->Getdl_m();
	
	double len_ratio = l_m/l_m0;
	// if(len_ratio > mLenRatio)
	// 	len_ratio = mLenRatio;

	double force = this->GetForce();
	if(v_CE <= 0) alpha = 0.16*activation*f0*g_al(len_ratio) + 0.18*force;
	else alpha = 0.157*force;
	
	v_CE = Utils::clamp(v_CE, -0.15, 0.15);

	h_SL = - alpha*v_CE;
	// cout << name << ": h_SL: " << h_SL << ", alpha: " << alpha << ", v_CE: " << v_CE << endl;
	return h_SL * mShortening_multiplier;
}

double Muscle::RateBhar04_MechWork()
{
	double w;
	double v_CE = v_m;
	double len_ratio = l_m/l_m0;
	if(v_CE <= 0) w = -1*v_CE*(activation*f0*g_al(len_ratio));
	else w = 0;
	// cout << name << ": w: " << w << ", v_CE: " << v_CE << endl;
	return w;
}

double Muscle::RateUmberger03_Activation()
{
	double hA;
    // Fiber type distribution
    double f_ST = type1_fraction;
    double f_FT = 1.0 - f_ST;

	// === 1. Activation + Maintenance heat ===
    // From Bolstad & Ersland, Eq: h_AM = 1.28 * %FT + 25
    double hAM = (1.28 * (f_FT * 100.0) + 25.0); // W/kg at full activation
    double h_A = 0.4 * hAM;
    double h_M = 0.6 * hAM;

    // Activation scaling, 
	// Comment in the opensim, "If excitation > activation, use excitation; otherwise use average of excitation and activation"
    double A = activation;
    // if (u > a)
        // A = u;
    // else
        // A = (u + a) / 2.0;

    double A_AM = pow(A, 0.4); // scaling for hA/hM
    hA = h_A * A_AM;
    double hM = h_M * A_AM;

    return hA;
}

double Muscle::RateUmberger03_Maintenance()
{
	double hM;
    // Fiber type distribution
    double f_ST = type1_fraction;
    double f_FT = 1.0 - f_ST;

	// === 1. Activation + Maintenance heat ===
    // From Bolstad & Ersland, Eq: h_AM = 1.28 * %FT + 25
    double hAM = (1.28 * (f_FT * 100.0) + 25.0); // W/kg at full activation
    double h_M = 0.6 * hAM;

    // Activation scaling, 
	// Comment in the opensim, "If excitation > activation, use excitation; otherwise use average of excitation and activation"
    double A = activation;
    // if (u > a)
        // A = u;
    // else
        // A = (u + a) / 2.0;

    double A_AM = pow(A, 0.4); // scaling for hA/hM
    hM = h_M * A_AM;

	// Fiber length & velocity
	double l_CE = l_m / l_m0;
	double v_CE = Utils::clamp(v_m, -0.5, 0.5); // Normalized velocity

	// Muscle force
	double F_iso = f0 * g_al(l_CE); // isometric force-length curve
    // Length scaling
    double F_iso_norm = F_iso / f0;
    if (l_CE > 1.0)
    {
        hM *= F_iso_norm;
    }

    return hM;
}

double Muscle::RateUmberger03_Shortening()
{
	double hs;
    double F_CE = GetForce(); // force from contractile element
    // === 2. Shortening / Lengthening heat ===
    hs = 0.0;
	double Vmax = 10.0;
    double as_ST = (4.0 * 25.0) / Vmax;   // W/kg per l/s for ST
    double as_FT = (1.0 * 153.0) / (2.5 * Vmax); // FT is 2.5x faster
	double v_CE = Utils::clamp(v_m, -0.5, 0.5);
	double A = activation;
	double f_FT = 1.0 - type1_fraction;

    if (v_CE <= 0) // Shortening
    {
        double As = pow(A, 2.0);
        hs = -v_CE * (as_ST * (1.0 - f_FT) + as_FT * f_FT) * As;
    }
    else // Lengthening
    {
        double al = 1.25 * (as_ST * (1.0 - f_FT) + as_FT * f_FT);
        hs = v_CE * al * A;
    }
	// cout << name << ": hs: " << hs 
	//      << ", F_CE: " << F_CE
	//      << ", v_CE: " << v_CE
	//      << ", activation: " << activation
	//      << ", type1_fraction: " << type1_fraction
	//      << ", as_ST: " << as_ST
	//      << ", as_FT: " << as_FT
	//      << ", f_FT: " << f_FT
	//      << endl;
	return hs;
}

double Muscle::RateUmberger03_MechWork()
{
    // === 3. Mechanical work ===
	double F_CE = GetForce();
	double v_CE = Utils::clamp(v_m, -0.5, 0.5);
    double wCE = -F_CE * v_CE; // Only positive work allowed
    if (wCE < 0) wCE = 0;
	return wCE;
}

double Muscle::RateUmberger03()
{
	double hA = RateUmberger03_Activation();
	double hM = RateUmberger03_Maintenance();
	double hs = RateUmberger03_Shortening();
	double wCE = RateUmberger03_MechWork();
    return hA + hM + hs + wCE;
}


double Muscle::RateHoud06()
{
	double h_A;
	double h_M;
	double h_SL;
	double w;

	double h_A_fast = 52.5; // W/kg
	double h_A_short = 10.98; // W/kg
	double h_M_fast = 97.5; // W/kg
	double h_M_short = 13.42; // W/kg
	double k1_fast = 12;
	double k1_short = 6;
	double k2_fast = 14;
	double k2_short = 8;

	double v = activation * activation;
	double v_max_fast = k1_fast + k2_fast*activation;
	double v_max_short = k1_short + k2_short*activation;

	int type;
	if((1-type1_fraction) > 0.5) type = 0; // fast
	else if((1-type1_fraction) < 0.5) type = 1; // slow
	else type = 2; // don't know

	double h_SL_fast = 0.28 * f0;
	double h_SL_short = 0.16 * f0;

	if(type==0){
		h_A = mMass*h_A_fast*v*(1-exp(-0.25-18.2/(v*v_max_fast)) / (1-exp(-0.25-18.2/(v_max_fast))));	
	}
	else if(type==1){
		h_A = mMass*h_A_short*v*(1-exp(-0.25-18.2/(v*v_max_short)) / (1-exp(-0.25-18.2/(v_max_short))));	
	}
	else{
		h_A = mMass*h_A_fast*v*(1-exp(-0.25-18.2/(v*v_max_fast)) / (1-exp(-0.25-18.2/(v_max_fast))));
		h_A += mMass*h_A_short*v*(1-exp(-0.25-18.2/(v*v_max_short)) / (1-exp(-0.25-18.2/(v_max_short))));
		h_A /= 2.0;
	}

	double len_ratio = l_m/l_m0;
	// if(len_ratio > mLenRatio)
	// 	len_ratio = mLenRatio;

	if(type==0){
		h_M = mMass*(h_A_fast + h_M_fast)*activation*(g_al(len_ratio) - h_A_fast/(h_A_fast + h_M_fast));	
	}
	else if(type==1){
		h_M = mMass*(h_A_short + h_M_short)*activation*(g_al(len_ratio) - h_A_short/(h_A_short + h_M_short));	
	}
	else{
		h_M = mMass*(h_A_fast + h_M_fast)*activation*(g_al(len_ratio) - h_A_fast/(h_A_fast + h_M_fast));
		h_M += mMass*(h_A_short + h_M_short)*activation*(g_al(len_ratio) - h_A_short/(h_A_short + h_M_short));
		h_M /= 2.0;
	}	
	
	double v_CE = v_m;
	// if(vType==0) v_CE = this->GetV_m();
	// else v_CE = this->Getdl_m();
	
	if(v_CE > 0.15) v_CE = 0.15;
	else if(v_CE < -0.15) v_CE = -0.15;

	double force = this->GetForce();
	if(v_CE < 0){
		if(type==0){
			h_SL = h_SL_fast * activation * g_al(len_ratio) - v_CE;	
		}
		else if(type==1){
			h_SL = h_SL_short * activation * g_al(len_ratio) - v_CE;	
		}
		else{
			h_SL = h_SL_fast * activation * g_al(len_ratio) - v_CE;
			h_SL += h_SL_short * activation * g_al(len_ratio) - v_CE;
			h_SL /= 2.0;
		}		
	}
	else h_SL = 0.0;

	if(v_CE < 0) w = -1*v_CE*force;
	else w = 0;

	return h_A + h_M + h_SL + w;
}


double Muscle::RateMine97()
{
	double a = activation;
	double f_max = f0;
	// double v_max = 1.6;
	double v_max = 4.8*(1 + 1.5*(1-type1_fraction));
	double v_CE = v_m;
	// if(vType==0) v_CE = this->GetV_m();
	// else v_CE = this->Getdl_m();

	if(v_CE > 0.15) v_CE = 0.15;
	else if(v_CE < -0.15) v_CE = -0.15;
	
	double phi = (0.054 + 0.506*v_CE + 2.46*v_CE*v_CE)/(1 - 1.13*v_CE + 12.8*v_CE*v_CE - 1.64*v_CE*v_CE*v_CE);

	double E_dot = a * f_max * v_max * phi;

	return E_dot;
}
}