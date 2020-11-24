#include "core/DARTHelper.h"

#include <memory>
#include "util/path.h"

using namespace std;

double MASS::GetAngleXY(BodyNode* parent_bn, BodyNode* child_bn){
    if (parent_bn == nullptr || child_bn == nullptr) {
        cout << "[Warning] Illegal BodyNode on MASS::GetAngleXY" << endl;
        return 0.0;
    }

    Eigen::Vector3d parentCOM = parent_bn->getCOM();
    Eigen::Vector3d childCOM = child_bn->getCOM();
    Eigen::Vector3d displacement = childCOM - parentCOM;

    if (displacement.norm() < 1e-6) {
        cout << "[Warning] Zero displacement on MASS::GetAngleXY" << endl;
        return 0.0;
    }

    if (sqrt(displacement.x()*displacement.x() + displacement.y()*displacement.y()) < 1e-6) {
        return 0.0;
    }
    double angle = atan2(displacement.z(), sqrt(displacement.x()*displacement.x() + displacement.y()*displacement.y()));
    return angle;
}

double MASS::GetAngleXZ(BodyNode* bn){
    if (bn == nullptr) {
        cout << "[Warning] Illegal BodyNode on MASS::GetAngleXZ" << endl;
        return 0.0;
    }

    Eigen::Matrix3d rotation = bn->getWorldTransform().linear();
    double angle = atan2(rotation(0,2), sqrt(rotation(0,0) * rotation(0,0) + rotation(0,1) * rotation(0,1)));
    return angle;
}

ShapePtr MASS::MakeSphereShape(double radius)
{
	return make_shared<SphereShape>(radius);
}

ShapePtr MASS::MakeBoxShape(const Eigen::Vector3d& size)
{
	return make_shared<BoxShape>(size);
}

ShapePtr MASS::MakeCapsuleShape(double radius, double height)
{
	return make_shared<CapsuleShape>(radius,height);
}

ShapePtr MASS::MakeCylinderShape(double radius, double height)
{
	return make_shared<CylinderShape>(radius,height);
}

Inertia MASS::MakeInertia(const dart::dynamics::ShapePtr& shape,double mass)
{
	dart::dynamics::Inertia inertia;

	inertia.setMass(mass);
	inertia.setMoment(shape->computeInertia(mass));
	return inertia;
}

FreeJoint::Properties* MASS::MakeFreeJointProperties(const string& name, const Eigen::Isometry3d& parent_to_joint, const Eigen::Isometry3d& child_to_joint, const double damping, const double stiffness, const double friction)
{
	auto* props = new FreeJoint::Properties();

	props->mName = name;
	props->mT_ParentBodyToJoint = parent_to_joint;
	props->mT_ChildBodyToJoint = child_to_joint;
	props->mIsPositionLimitEnforced = false;
	props->mVelocityLowerLimits = Eigen::Vector6d::Constant(-DART_JOINT_VEL_LIMIT);
	props->mVelocityUpperLimits = Eigen::Vector6d::Constant(DART_JOINT_VEL_LIMIT);
	props->mForceLowerLimits = Eigen::Vector6d::Constant(-DART_JOINT_FORCE_LIMIT);
	props->mForceUpperLimits = Eigen::Vector6d::Constant(DART_JOINT_FORCE_LIMIT);
	props->mDampingCoefficients = Eigen::Vector6d::Constant(damping);
	props->mSpringStiffnesses = Eigen::Vector6d::Constant(stiffness);
	props->mFrictions = Eigen::Vector6d::Constant(friction);
	
	return props;
}

PlanarJoint::Properties* MASS::MakePlanarJointProperties(const string& name,const Eigen::Isometry3d& parent_to_joint,const Eigen::Isometry3d& child_to_joint, const double damping, const double friction, const double stiffness)
{
	auto* props = new PlanarJoint::Properties();

	props->mName = name;
	props->mT_ParentBodyToJoint = parent_to_joint;
	props->mT_ChildBodyToJoint = child_to_joint;
	props->mIsPositionLimitEnforced = false;
	props->mVelocityLowerLimits = Eigen::Vector3d::Constant(-DART_JOINT_VEL_LIMIT);
	props->mVelocityUpperLimits = Eigen::Vector3d::Constant(DART_JOINT_VEL_LIMIT);
	props->mForceLowerLimits = Eigen::Vector3d::Constant(-DART_JOINT_FORCE_LIMIT); 
	props->mForceUpperLimits = Eigen::Vector3d::Constant(DART_JOINT_FORCE_LIMIT);
	props->mDampingCoefficients = Eigen::Vector3d::Constant(damping);
	props->mSpringStiffnesses = Eigen::Vector3d::Constant(stiffness);
	props->mFrictions = Eigen::Vector3d::Constant(friction);

	return props;
}

BallJoint::Properties* MASS::MakeBallJointProperties(const string& name,const Eigen::Isometry3d& parent_to_joint,const Eigen::Isometry3d& child_to_joint,const Eigen::Vector3d& lower,const Eigen::Vector3d& upper, const Eigen::Vector3d& damping, const Eigen::Vector3d& friction, const Eigen::Vector3d& stiffness)
{
	auto* props = new BallJoint::Properties();

	props->mName = name;
	props->mT_ParentBodyToJoint = parent_to_joint;
	props->mT_ChildBodyToJoint = child_to_joint;
	props->mIsPositionLimitEnforced = true;
	props->mPositionLowerLimits = lower;
	props->mPositionUpperLimits = upper;
	props->mVelocityLowerLimits = Eigen::Vector3d::Constant(-DART_JOINT_VEL_LIMIT);
	props->mVelocityUpperLimits = Eigen::Vector3d::Constant(DART_JOINT_VEL_LIMIT);
	props->mForceLowerLimits = Eigen::Vector3d::Constant(-DART_JOINT_FORCE_LIMIT); 
	props->mForceUpperLimits = Eigen::Vector3d::Constant(DART_JOINT_FORCE_LIMIT);
	props->mDampingCoefficients = damping;
	props->mSpringStiffnesses = stiffness;
	props->mFrictions = friction;
	
	return props;
}

RevoluteJoint::Properties* MASS::MakeRevoluteJointProperties(const string& name,const Eigen::Vector3d& axis,const Eigen::Isometry3d& parent_to_joint,const Eigen::Isometry3d& child_to_joint,const Eigen::Vector1d& lower,const Eigen::Vector1d& upper, const double damping, const double friction, const double stiffness)
{
	auto* props = new RevoluteJoint::Properties();

	props->mName = name;
	props->mT_ParentBodyToJoint = parent_to_joint;
	props->mT_ChildBodyToJoint = child_to_joint;
	props->mIsPositionLimitEnforced = true;
	props->mAxis = axis;
	props->mPositionLowerLimits = lower;
	props->mPositionUpperLimits = upper;
	props->mVelocityLowerLimits = Eigen::Vector1d::Constant(-10.0);
	props->mVelocityUpperLimits = Eigen::Vector1d::Constant(10.0);
	props->mForceLowerLimits = Eigen::Vector1d::Constant(-1000.0); 
	props->mForceUpperLimits = Eigen::Vector1d::Constant(1000.0);
	props->mDampingCoefficients = Eigen::Vector1d::Constant(damping);
	props->mSpringStiffnesses = Eigen::Vector1d::Constant(stiffness);
	props->mFrictions = Eigen::Vector1d::Constant(friction);	

	return props;
}

WeldJoint::Properties* MASS::MakeWeldJointProperties(const string& name,const Eigen::Isometry3d& parent_to_joint,const Eigen::Isometry3d& child_to_joint)
{
	auto* props = new WeldJoint::Properties();

	props->mName = name;
	props->mT_ParentBodyToJoint = parent_to_joint;
	props->mT_ChildBodyToJoint = child_to_joint;

	return props;
}

BodyNode* MASS::MakeBodyNode(const SkeletonPtr& skeleton,BodyNode* parent,Joint::Properties* joint_properties,const string& joint_type,dart::dynamics::Inertia inertia)
{
	BodyNode* bn;

	if(joint_type == "Free")
	{
		auto* prop = dynamic_cast<FreeJoint::Properties*>(joint_properties);
		bn = skeleton->createJointAndBodyNodePair<FreeJoint>(
			parent,(*prop),BodyNode::AspectProperties(joint_properties->mName)).second;
	}
	else if(joint_type == "Planar")
	{
		auto* prop = dynamic_cast<PlanarJoint::Properties*>(joint_properties);
		bn = skeleton->createJointAndBodyNodePair<PlanarJoint>(
			parent,(*prop),BodyNode::AspectProperties(joint_properties->mName)).second;
	}
	else if(joint_type == "Ball")
	{
		auto* prop = dynamic_cast<BallJoint::Properties*>(joint_properties);
		bn = skeleton->createJointAndBodyNodePair<BallJoint>(
			parent,(*prop),BodyNode::AspectProperties(joint_properties->mName)).second;
	}
	else if(joint_type == "Revolute")
	{
		auto* prop = dynamic_cast<RevoluteJoint::Properties*>(joint_properties);
		bn = skeleton->createJointAndBodyNodePair<RevoluteJoint>(
			parent,(*prop),BodyNode::AspectProperties(joint_properties->mName)).second;
	}
	else if(joint_type == "Weld")
	{
		auto* prop = dynamic_cast<WeldJoint::Properties*>(joint_properties);
		bn = skeleton->createJointAndBodyNodePair<WeldJoint>(
			parent,(*prop),BodyNode::AspectProperties(joint_properties->mName)).second;
	}

	bn->setInertia(inertia);
	return bn;
}

Eigen::Vector3d Proj(const Eigen::Vector3d& u,const Eigen::Vector3d& v)
{
	Eigen::Vector3d proj;
	proj = u.dot(v)/u.dot(u)*u;
	return proj;	
}

Eigen::Isometry3d Orthonormalize(const Eigen::Isometry3d& T_old)
{
	Eigen::Isometry3d T;
	T.translation() = T_old.translation();
	Eigen::Vector3d v0,v1,v2;
	Eigen::Vector3d u0,u1,u2;
	v0 = T_old.linear().col(0);
	v1 = T_old.linear().col(1);
	v2 = T_old.linear().col(2);

	u0 = v0;
	u1 = v1 - Proj(u0,v1);
	u2 = v2 - Proj(u0,v2) - Proj(u1,v2);

	u0.normalize();
	u1.normalize();
	u2.normalize();

	T.linear().col(0) = u0;
	T.linear().col(1) = u1;
	T.linear().col(2) = u2;
	return T;
}

vector<double> split_to_double(const string& input, int num)
{
    vector<double> result;
    string::size_type sz = 0, nsz = 0;
    for(int i = 0; i < num; i++){
        result.push_back(stof(input.substr(sz), &nsz));
        sz += nsz;
    }
    return result;
}

Eigen::Vector1d string_to_vector1d(const string& input){
	vector<double> v = split_to_double(input, 1);
	Eigen::Vector1d res;
	res << v[0];

	return res;
}

Eigen::Vector3d string_to_vector3d(const string& input){
	vector<double> v = split_to_double(input, 3);
	Eigen::Vector3d res;
	res << v[0], v[1], v[2];

	return res;
}

Eigen::Vector4d string_to_vector4d(const string& input)
{
	vector<double> v = split_to_double(input, 4);
	Eigen::Vector4d res;
	res << v[0], v[1], v[2],v[3];

	return res;	
}

Eigen::VectorXd string_to_vectorXd(const string& input, int n){
	vector<double> v = split_to_double(input, n);
	Eigen::VectorXd res(n);
	for(int i = 0; i < n; i++){
		res[i] = v[i];
	}

	return res;
}

Eigen::Matrix3d string_to_matrix3d(const string& input){
	vector<double> v = split_to_double(input, 9);
	Eigen::Matrix3d res;
	res << v[0], v[1], v[2],
			v[3], v[4], v[5],
			v[6], v[7], v[8];

	return res;
}

SkeletonPtr MASS::BuildGround(const Eigen::Vector3d& size)
{
    SkeletonPtr skel = Skeleton::create("Ground");
	Eigen::Isometry3d T_body = Eigen::Isometry3d::Identity();
	Eigen::Isometry3d T_joint = Eigen::Isometry3d::Identity();
	T_body.translation() = Eigen::Vector3d(0.0, -size[1]/2.0, size[2]/2.0 - 5.0);
	T_body = Orthonormalize(T_body);
	Eigen::Isometry3d child_to_joint = T_body.inverse()*T_joint;
	Eigen::Isometry3d parent_to_joint = T_joint;
	auto* jnt_props = MASS::MakeWeldJointProperties("ground", parent_to_joint, child_to_joint);
	ShapePtr shape = MASS::MakeBoxShape(size);
	dart::dynamics::Inertia inertia = MakeInertia(shape, 15.0);
	BodyNode* bn = MASS::MakeBodyNode(skel, nullptr, jnt_props, "Weld", inertia);
	bn->createShapeNodeWith<VisualAspect, CollisionAspect, DynamicsAspect>(shape);

	return skel;
}

SkeletonPtr MASS::BuildFromFile(const string& path, bool create_obj, double jointDamping, Eigen::Vector4d color_filter, bool isContact, bool isBVH)
{
	TiXmlDocument doc;
	if(doc.LoadFile(path.c_str())){
		cout << "Can't open file : " << path << endl;
		return nullptr;
	}

	TiXmlElement* skeleton_elem = doc.FirstChildElement("Skeleton");
	string skel_name = skeleton_elem->Attribute("name");
	SkeletonPtr skel = Skeleton::create(skel_name);
	
	for(TiXmlElement* node=skeleton_elem->FirstChildElement("Node"); node!=nullptr; node=node->NextSiblingElement("Node"))
	{		
		string name = node->Attribute("name");
		string parent_str = node->Attribute("parent");
		BodyNode* parent = nullptr;
		if(parent_str != "None") parent = skel->getBodyNode(parent_str);
				
		TiXmlElement* body = node->FirstChildElement("Body");
		string body_type = body->Attribute("type");
		string obj_file = "None";
		if(body->Attribute("obj") != nullptr) obj_file = body->Attribute("obj");

		ShapePtr shape;
		if(body_type == "Box")
		{
			Eigen::Vector3d size = string_to_vector3d(body->Attribute("size"));
			shape = MASS::MakeBoxShape(size);
		}
		else if(body_type=="Sphere")
		{
			double radius = stod(body->Attribute("radius"));
			shape = MASS::MakeSphereShape(radius);
		}
		else if(body_type=="Capsule")
		{
			double radius = stod(body->Attribute("radius"));
			double height = stod(body->Attribute("height"));
			shape = MASS::MakeCapsuleShape(radius,height);
		}
		else if(body_type=="Cylinder")
		{
			double radius = stod(body->Attribute("radius"));
			double height = stod(body->Attribute("height"));
			shape = MASS::MakeCylinderShape(radius,height);
		}
		double mass = stod(body->Attribute("mass"));
		dart::dynamics::Inertia inertia = MakeInertia(shape, mass);
		
		bool contact = false;
		if(body->Attribute("contact") != nullptr){
			string c = body->Attribute("contact");
			if(c == "On") contact = true & isContact;
		}

		Eigen::Vector4d color = Eigen::Vector4d::Constant(0.3);
		if(body->Attribute("color") != nullptr) color = color_filter.asDiagonal()*string_to_vector4d(body->Attribute("color"));

		Eigen::Isometry3d T_body = Eigen::Isometry3d::Identity();
		T_body.linear() = string_to_matrix3d(body->FirstChildElement("Transformation")->Attribute("linear"));
		T_body.translation() = string_to_vector3d(body->FirstChildElement("Transformation")->Attribute("translation"));
		T_body = Orthonormalize(T_body);					
		
		TiXmlElement* joint = node->FirstChildElement("Joint");
		string joint_type = joint->Attribute("type");
				
		Eigen::Isometry3d T_joint = Eigen::Isometry3d::Identity();
		T_joint.linear() = string_to_matrix3d(joint->FirstChildElement("Transformation")->Attribute("linear"));
		T_joint.translation() = string_to_vector3d(joint->FirstChildElement("Transformation")->Attribute("translation"));
		T_joint = Orthonormalize(T_joint);	
		
		Eigen::Isometry3d child_to_joint = T_body.inverse()*T_joint;
		Eigen::Isometry3d parent_to_joint;
		if(parent==nullptr) parent_to_joint = T_joint;
		else parent_to_joint = parent->getTransform().inverse()*T_joint;
		
		Joint::Properties* props;		
		if(joint_type == "Free")
		{
			double damping = jointDamping; 
			double stiffness = 0.0;

			if(joint->Attribute("damping") != NULL) damping = stod(joint->Attribute("damping"));
			if(joint->Attribute("stiffness") != NULL) stiffness = stod(joint->Attribute("stiffness"));
			props = MASS::MakeFreeJointProperties(name,parent_to_joint,child_to_joint,damping,stiffness);
			
		}
		else if(joint_type == "Planar")
		{
			props = MASS::MakePlanarJointProperties(name,parent_to_joint,child_to_joint);
		}
		else if(joint_type == "Weld")
		{			
			props = MASS::MakeWeldJointProperties(name,parent_to_joint,child_to_joint);
		}
		else if(joint_type == "Ball")
		{ 
			Eigen::Vector3d lower, upper; // = string_to_vector3d(joint->Attribute("upper"));
			Eigen::Vector3d damping = Eigen::Vector3d::Constant(jointDamping);			
			Eigen::Vector3d friction = Eigen::Vector3d::Zero();			
			Eigen::Vector3d stiffness = Eigen::Vector3d::Zero();
		
			lower = string_to_vector3d(joint->Attribute("lower"));
			upper = string_to_vector3d(joint->Attribute("upper"));
			
			if(joint->Attribute("damping") != NULL) damping = string_to_vector3d(joint->Attribute("damping"));
			if(joint->Attribute("friction") != NULL) friction = string_to_vector3d(joint->Attribute("friction"));
			if(joint->Attribute("stiffness") != NULL) stiffness = string_to_vector3d(joint->Attribute("stiffness"));				
			props = MASS::MakeBallJointProperties(name,parent_to_joint,child_to_joint,lower,upper,damping,friction,stiffness);		
		}
		else if(joint_type == "Revolute")
		{
			Eigen::Vector1d lower = string_to_vector1d(joint->Attribute("lower"));
			Eigen::Vector1d upper = string_to_vector1d(joint->Attribute("upper"));
			Eigen::Vector3d axis = string_to_vector3d(joint->Attribute("axis"));
			
			double damping = jointDamping; 
			double friction = 0.0;
			double stiffness = 0.0;			

			if(joint->Attribute("damping") != NULL) damping = stod(joint->Attribute("damping"));
			if(joint->Attribute("friction") != NULL) friction = stod(joint->Attribute("friction"));
			if(joint->Attribute("stiffness") != NULL) stiffness = stod(joint->Attribute("stiffness"));

			props = MASS::MakeRevoluteJointProperties(name, axis, parent_to_joint, child_to_joint, lower, upper, damping, friction, stiffness);
		}

		auto bn = MakeBodyNode(skel, parent, props, joint_type, inertia);
		if(contact) bn->createShapeNodeWith<VisualAspect, CollisionAspect, DynamicsAspect>(shape);
		else bn->createShapeNodeWith<VisualAspect, DynamicsAspect>(shape);
		bn->eachShapeNodeWith<VisualAspect>([color](ShapeNode* sn) {
			sn->getVisualAspect()->setColor(color);
		});

		//Set joint rendering
		double joint_radius = 0.02;
		Eigen::Vector4d joint_color(0.8,0.2,0.8,1.0);
		ShapePtr joint_shape = MASS::MakeSphereShape(joint_radius);
		bn->createShapeNodeWith<VisualAspect>(joint_shape);
		ShapeNode* target = nullptr;
		bn->eachShapeNodeWith<VisualAspect>([&target](ShapeNode* sn) {
			target = sn;
		});
		if(target) {
			target->getVisualAspect()->setColor(joint_color);
			target->setRelativeTransform(child_to_joint);
		} else {
			cerr << "[DARTHelper] No target found" << endl;
			exit(1);
		}
		if(obj_file!="None" && create_obj)
		{
			const aiScene* scene = MeshShape::loadMesh(string(path_obj(obj_file)));
			MeshShapePtr visual_shape = shared_ptr<MeshShape>(new MeshShape(Eigen::Vector3d(0.01,0.01,0.01),scene));
			visual_shape->setColorMode(MeshShape::ColorMode::SHAPE_COLOR);
			auto vsn = bn->createShapeNodeWith<VisualAspect>(visual_shape);

			Eigen::Isometry3d T_obj;
			T_obj.setIdentity();			
			T_obj = T_body.inverse();
			vsn->setRelativeTransform(T_obj);
		}		
	}
	return skel;
}

void MASS::OsimParser(const string& path)
{
	TiXmlDocument doc;
	if(doc.LoadFile(path.c_str())){
		cout << "Can't open file : " << path << endl;
		return;
	}

	TiXmlElement* muscle = doc.FirstChildElement("Muscle");
	for (TiXmlElement* unit = muscle->FirstChildElement("Unit"); unit != NULL; unit = unit->NextSiblingElement("Unit"))
	{
		string name = unit->Attribute("name");
		string force = unit->Attribute("f0");
		cout << name << " : " << force << endl;	
	}
}

void MASS::
ExportSkeleton(string path, dart::dynamics::SkeletonPtr skel, Eigen::VectorXd kp, Eigen::VectorXd damping)
{
	ofstream sfs(path);
	cout << "Save to: "<< path << endl;

	sfs << "<Skeleton name=\"Human\">" << endl;
	for(int j=0; j<skel->getNumBodyNodes(); j++)
	{
		auto bn = skel->getBodyNode(j);

		// joint type / parent name 
		string joint_type;
		string parent_name;
		
		if(j == 0) {
			joint_type = "Free";
			parent_name = "None";
		} else {
			joint_type = bn->getParentJoint()->getType();
			parent_name = bn->getParentBodyNode()->getName();
		}

		// body type
		string body_type = "Box";
		
		// size
		Eigen::Isometry3d transform;
		Eigen::Vector3d size;
		bn->eachShapeNodeWith<VisualAspect>([&transform, &size](ShapeNode* sn) {
			transform = sn->getTransform().cast<double>();
			size = dynamic_cast<const dart::dynamics::BoxShape*>(sn->getShape().get())->getSize().cast<double>();
		});

		// contact
		string contact = "On";

		// body
		Eigen::Isometry3d T_body;
		T_body = bn->getWorldTransform();

		// joint
		Eigen::Isometry3d T_joint = Eigen::Isometry3d::Identity();
		if(bn->getParentJoint()->getParentBodyNode() != nullptr) {
			T_joint = bn->getWorldTransform() * bn->getParentJoint()->getTransformFromChildBodyNode();
		} else {
			T_joint = Eigen::Isometry3d::Identity();
		}

		// file write
		sfs << "  <Node name=\"" + bn->getName() + "\" parent=\"" + parent_name + "\" >" << endl;

		/////// body ///////
		sfs << "    <Body type=\"" + body_type 
			+ "\" mass=\"" + to_string(bn->getMass()) 
			+ "\" size=\"" + Utils::VectorXd2String(size) 
			+ "\" contact=\"" + contact
			+ "\">" << endl;

		Eigen::VectorXd T_body_linear;
		T_body_linear.resize(9);
		for(int i=0; i<3; i++)
			T_body_linear.segment(3*i, 3) = T_body.linear().row(i);

		sfs << "      <Transformation linear=\"" + Utils::VectorXd2String(T_body_linear) 
			+ "\" translation=\"" + Utils::VectorXd2String(T_body.translation()) 
			+ "\"/>" << endl;

		sfs << "      <Geometry linear=\"" + Utils::VectorXd2String(T_body_linear) 
			+ "\" translation=\"" + Utils::VectorXd2String(bn->getLocalCOM()) 
			+ "\"/>" << endl;

		sfs << "    </Body>" << endl;

		/////// joint ///////
		if(j == 0) {
			sfs << "    <Joint type=\"Free\" kp=\"0 0 0 0 0 0\" damping=\"" + to_string(kp[bn->getParentJoint()->getIndexInSkeleton(0)]) + "\">" << endl; 
			sfs << "      <Transformation linear=\"1.000000 0.000000 0.000000 0.000000 1.000000 0.000000 0.000000 0.000000 1.000000" 
				<< "\" translation=\"" << bn->getParentJoint()->getTransformFromParentBodyNode().translation().transpose() 
				<< "\"/>" << endl;
		} else {
			if(joint_type.find("Ball") != string::npos) {
				Eigen::Vector3d lower = bn->getParentJoint()->getPositionLowerLimits(); 
				Eigen::Vector3d upper = bn->getParentJoint()->getPositionUpperLimits(); 

				sfs << "    <Joint type=\"Ball"; 
				if(!dart::math::isInf(lower) && !dart::math::isInf(upper)) {
					sfs << "\" lower=\"" + Utils::VectorXd2String(lower)
						+ "\" upper=\"" + Utils::VectorXd2String(upper);
				}
				sfs << "\" kp=\"" + Utils::VectorXd2String(kp.segment(bn->getParentJoint()->getIndexInSkeleton(0), 3));
				sfs << "\" damping=\"" + Utils::VectorXd2String(damping.segment(bn->getParentJoint()->getIndexInSkeleton(0), 3));				
			} else if (joint_type.find("Revolute") != string::npos) {
				Eigen::Vector1d lower = bn->getParentJoint()->getPositionLowerLimits(); 
				Eigen::Vector1d upper = bn->getParentJoint()->getPositionUpperLimits(); 

				sfs << "    <Joint type=\"Revolute"; 
				sfs << "\" axis=\"1 0 0";
				if(!dart::math::isInf(lower) && !dart::math::isInf(upper)) {
					sfs << "\" lower=\"" + Utils::VectorXd2String(lower)
						+ "\" upper=\"" + Utils::VectorXd2String(upper);
				}
				sfs << "\" kp=\"" + to_string(kp[bn->getParentJoint()->getIndexInSkeleton(0)]);
				sfs << "\" damping=\"" + to_string(damping[bn->getParentJoint()->getIndexInSkeleton(0)]);
			} else if (joint_type.find("Weld") != string::npos) {
				sfs << "    <Joint type=\"Weld\" kp=\"0"; 
			}
			
			sfs << "\">" << endl;

			Eigen::VectorXd T_joint_linear;
			T_joint_linear.resize(9);
			for(int i=0; i<3; i++)
				T_joint_linear.segment(3*i, 3) = T_joint.linear().row(i);

			sfs << "      <Transformation linear=\"" + Utils::VectorXd2String(T_joint_linear) 
				+ "\" translation=\"" + Utils::VectorXd2String(T_joint.translation()) 
				+ "\"/>" << endl;
		}
		sfs << "    </Joint>" << endl;
		sfs << "  </Node>" << endl << endl;
		
	}
	sfs << "</Skeleton>" << endl;
	sfs.close();
}
void MASS::
ExportMuscles(string path, vector<Muscle*> muscles)
{
	ofstream mfs(path);
	cout << "Save to: "<< path << endl;

	mfs << "<Muscle>" << endl;

	for(auto m: muscles)
	{
		string name = m->GetName();
		double f0 = m->GetF0();
		double l_m0 = m->GetMT0();
		double l_t0 = m->GetTendonSlackLength();

		mfs << "<Unit name=\"" << name
				<< "\" f0=\"" << f0
				<< "\" lm=\"" << l_m0
				<< "\" lt=\"" << l_t0
				<< "\" pen_angle=\"" << 0.0
				<< "\" lmax=\"" << -0.1				
				<< "\" >" << endl;

//		 From #17e6, since given_bn is removed for speed up, this part is commented
		 for(auto a: m->mAnchors)
		 {
            const auto bn = a->bodynodes[0];
		 	const auto body_name = bn->getName();
		 	Eigen::Vector3d scale, shape;

			bn->eachShapeNodeWith<VisualAspect>([&shape](ShapeNode* sn) {
				shape = dynamic_cast<const dart::dynamics::BoxShape*>(sn->getShape().get())->getSize().cast<double>()/2.0;
			});

			Eigen::Vector3d glob_position;
			a->GetPoint(glob_position);

		 	auto T = bn->getTransform();
		 	auto local_position = T.inverse() * glob_position;

		 	scale[0] = local_position[0] / shape[0];
		 	scale[1] = local_position[1] / shape[1];
		 	scale[2] = local_position[2] / shape[2];

		 	if(abs(scale[0]) < 1e-6) scale[0] = 0.0;
		 	if(abs(scale[1]) < 1e-6) scale[1] = 0.0;
		 	if(abs(scale[2]) < 1e-6) scale[2] = 0.0;

		 	mfs << "  <Waypoint body=\"" << body_name
		 		<< "\" p=\"" << glob_position[0] << " " << glob_position[1] << " " << glob_position[2]
		 		<< "\" />" << endl;
		 }

		mfs << "</Unit>" << endl;
	}

	mfs << "</Muscle>" << endl;
	mfs.close();
}

// void
// MASS::OsimParser(const string& path)
// {
// 	TiXmlDocument doc;
// 	if(doc.LoadFile(path.c_str())){
// 		cout << "Can't open file : " << path << endl;
// 		return;
// 	}

// 	TiXmlElement* elem = doc.FirstChildElement("OpenSimDocument");
// 	cout << "version : " << elem->Attribute("Version") << endl;

// 	TiXmlElement* model = elem->FirstChildElement("Model");
// 	cout << "name : " << model->Attribute("name") << endl;
						
// 	// TiXmlElement* credits = model->NextSiblingElement();
// 	// if(credits)
// 	// 	cout << credits->Value() << endl;

// 	// TiXmlElement* publications = credits->NextSiblingElement("publications");
// 	// cout << "a2" << endl;
// 	// TiXmlElement* length_units = publications->NextSiblingElement("length_units");
// 	// cout << "a3" << endl;
// 	// TiXmlElement* force_units =  length_units->FirstChildElement("force_units");
// 	// cout << "a4" << endl;
// 	// TiXmlElement* forceSet =  force_units->NextSiblingElement("gravity")->NextSiblingElement("BodySet")->NextSiblingElement("ConstraintSet")->NextSiblingElement("ForceSet");
// 	// cout << "a5" << endl;
// 	// TiXmlElement* objects = forceSet->FirstChildElement("objects");
// 	// cout << "a6" << endl;

// 	for (TiXmlElement* child = model->FirstChildElement("Thelen2003Muscle"); child != NULL; child = child->NextSiblingElement("Thelen2003Muscle"))
// 	{
// 		string name = child->Attribute("name");
		

// 		TiXmlElement* iso = child->FirstChildElement("isDisabled")->NextSiblingElement("min_control")->NextSiblingElement("max_control")->NextSiblingElement("GeometryPath")->NextSiblingElement("optimal_force")->NextSiblingElement("max_isometric_force");
// 		string force = iso->GetText();
// 		cout <<  name << " : " << force << endl;		

// 	}	
// }