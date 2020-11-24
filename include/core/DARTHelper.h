#ifndef __DART_HELPER_H__
#define __DART_HELPER_H__
#include "dart/dart.hpp"
#include "Muscle.h"
#include "Utils.h"
#include <experimental/filesystem>
#include <tinyxml2.h>

#define DART_JOINT_VEL_LIMIT    10.0
#define DART_JOINT_FORCE_LIMIT  1000.0


using namespace dart::dynamics;
using namespace dart::simulation;
namespace fs = std::experimental::filesystem;
namespace Eigen 
{
    using Vector1d = Matrix<double, 1, 1>;
    using Matrix1d = Matrix<double, 1, 1>;
}

std::vector<double> split_to_double(const std::string& input, int num);
Eigen::Vector1d string_to_vector1d(const std::string& input);
Eigen::Vector3d string_to_vector3d(const std::string& input);
Eigen::Vector4d string_to_vector4d(const std::string& input);
Eigen::VectorXd string_to_vectorXd(const std::string& input, int n);
Eigen::Matrix3d string_to_matrix3d(const std::string& input);

using namespace dart::dynamics;
namespace MASS
{
    typedef tinyxml2::XMLElement TiXmlElement;
    typedef tinyxml2::XMLDocument TiXmlDocument;

    double GetAngleXY(BodyNode* parent_bn, BodyNode* child_bn);
    double GetAngleXZ(BodyNode* bn);
    ShapePtr MakeSphereShape(double radius);
    ShapePtr MakeBoxShape(const Eigen::Vector3d& size);
    ShapePtr MakeCapsuleShape(double radius, double height);
    ShapePtr MakeCylinderShape(double radius, double height);
    Inertia MakeInertia(const dart::dynamics::ShapePtr& shape,double mass);

    FreeJoint::Properties*	MakeFreeJointProperties(const std::string& name,const Eigen::Isometry3d& parent_to_joint = Eigen::Isometry3d::Identity(),const Eigen::Isometry3d& child_to_joint = Eigen::Isometry3d::Identity(), const double damping = 0.4, const double stiffness = 0.0, const double friction = 0.0);
    PlanarJoint::Properties* MakePlanarJointProperties(const std::string& name,const Eigen::Isometry3d& parent_to_joint = Eigen::Isometry3d::Identity(),const Eigen::Isometry3d& child_to_joint = Eigen::Isometry3d::Identity(), const double damping = 0.4, const double friction = 0.0, const double stiffness = 0.0);
    WeldJoint::Properties* MakeWeldJointProperties(const std::string& name,const Eigen::Isometry3d& parent_to_joint = Eigen::Isometry3d::Identity(),const Eigen::Isometry3d& child_to_joint = Eigen::Isometry3d::Identity());
    RevoluteJoint::Properties* MakeRevoluteJointProperties(const std::string& name,const Eigen::Vector3d& axis,const Eigen::Isometry3d& parent_to_joint = Eigen::Isometry3d::Identity(),const Eigen::Isometry3d& child_to_joint = Eigen::Isometry3d::Identity(),const Eigen::Vector1d& lower = Eigen::Vector1d::Constant(-2.0),const Eigen::Vector1d& upper = Eigen::Vector1d::Constant(2.0), const double damping = 0.4, const double friction = 0.0, const double stiffness = 0.0);
    BallJoint::Properties* MakeBallJointProperties(const std::string& name,const Eigen::Isometry3d& parent_to_joint = Eigen::Isometry3d::Identity(),const Eigen::Isometry3d& child_to_joint = Eigen::Isometry3d::Identity(),const Eigen::Vector3d& lower = Eigen::Vector3d::Constant(-2.0),const Eigen::Vector3d& upper = Eigen::Vector3d::Constant(2.0), const Eigen::Vector3d& damping = Eigen::Vector3d::Constant(0.4), const Eigen::Vector3d& friction = Eigen::Vector3d::Constant(0.0), const Eigen::Vector3d& stiffness = Eigen::Vector3d::Constant(0.0));
    
    BodyNode* MakeBodyNode(const dart::dynamics::SkeletonPtr& skeleton,dart::dynamics::BodyNode* parent,dart::dynamics::Joint::Properties* joint_properties,const std::string& joint_type,dart::dynamics::Inertia inertia);
    SkeletonPtr BuildGround(const Eigen::Vector3d& size);
    SkeletonPtr BuildFromFile(const std::string& path, 
                            bool create_obj = false, 
                            double jointDamping = 0.4, 
                            Eigen::Vector4d color_filter = Eigen::Vector4d(1,1,1,1), 
                            bool isContact = true, 
                            bool isBVH = false);
    void OsimParser(const std::string& path);

    void ExportSkeleton(std::string path, dart::dynamics::SkeletonPtr skel, Eigen::VectorXd kp, Eigen::VectorXd damping);
    void ExportMuscles(std::string path, std::vector<Muscle*> muscles);
}

#endif