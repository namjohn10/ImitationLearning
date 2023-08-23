#ifndef __MS_DARTHELPER_H__
#define __MS_DARTHELPER_H__
#include "dart/dart.hpp"
#include <tinyxml2.h>

#include <experimental/filesystem>

using namespace dart::dynamics;

typedef tinyxml2::XMLElement TiXmlElement;
typedef tinyxml2::XMLDocument TiXmlDocument;

namespace fs = std::experimental::filesystem;

namespace Eigen
{
    using Vector1d = Matrix<double, 1, 1>;
    using Matrix1d = Matrix<double, 1, 1>;
}

std::vector<std::string> split_string(const std::string &input);

std::vector<double> split_to_double(const std::string &input, int num);
Eigen::Vector1d string_to_vector1d(const std::string &input);
Eigen::Vector3d string_to_vector3d(const std::string &input);
Eigen::Vector4d string_to_vector4d(const std::string &input);
Eigen::VectorXd string_to_vectorXd(const std::string &input, int n);
Eigen::Matrix3d string_to_matrix3d(const std::string &input);

dart::dynamics::ShapePtr MakeSphereShape(double radius);
dart::dynamics::ShapePtr MakeBoxShape(const Eigen::Vector3d &size);
dart::dynamics::ShapePtr MakeCapsuleShape(double radius, double height);
dart::dynamics::ShapePtr MakeCylinderShape(double radius, double height);
dart::dynamics::Inertia MakeInertia(const dart::dynamics::ShapePtr &shape, double mass);

dart::dynamics::FreeJoint::Properties *MakeFreeJointProperties(const std::string &name, const Eigen::Isometry3d &parent_to_joint = Eigen::Isometry3d::Identity(), const Eigen::Isometry3d &child_to_joint = Eigen::Isometry3d::Identity(), const double damping = 0.4);
dart::dynamics::PlanarJoint::Properties *MakePlanarJointProperties(const std::string &name, const Eigen::Isometry3d &parent_to_joint = Eigen::Isometry3d::Identity(), const Eigen::Isometry3d &child_to_joint = Eigen::Isometry3d::Identity());
dart::dynamics::BallJoint::Properties *MakeBallJointProperties(const std::string &name, const Eigen::Isometry3d &parent_to_joint = Eigen::Isometry3d::Identity(), const Eigen::Isometry3d &child_to_joint = Eigen::Isometry3d::Identity(), const Eigen::Vector3d &lower = Eigen::Vector3d::Constant(-2.0), const Eigen::Vector3d &upper = Eigen::Vector3d::Constant(2.0), const double damping = 0.4, const double friction = 0.0, const double stiffness = 0.0);
dart::dynamics::RevoluteJoint::Properties *MakeRevoluteJointProperties(const std::string &name, const Eigen::Vector3d &axis, const Eigen::Isometry3d &parent_to_joint = Eigen::Isometry3d::Identity(), const Eigen::Isometry3d &child_to_joint = Eigen::Isometry3d::Identity(), const Eigen::Vector1d &lower = Eigen::Vector1d::Constant(-2.0), const Eigen::Vector1d &upper = Eigen::Vector1d::Constant(2.0), const double damping = 0.4, const double friction = 0.0, const double stiffness = 0.0);
dart::dynamics::WeldJoint::Properties *MakeWeldJointProperties(const std::string &name, const Eigen::Isometry3d &parent_to_joint = Eigen::Isometry3d::Identity(), const Eigen::Isometry3d &child_to_joint = Eigen::Isometry3d::Identity());
dart::dynamics::BodyNode *MakeBodyNode(const dart::dynamics::SkeletonPtr &skeleton, dart::dynamics::BodyNode *parent, dart::dynamics::Joint::Properties *joint_properties, const std::string &joint_type, dart::dynamics::Inertia inertia);
dart::dynamics::SkeletonPtr BuildFromFile(const std::string &path, double defaultDamping = 0.4, Eigen::Vector4d color_filter = Eigen::Vector4d(1, 1, 1, 1), bool isContact = true, bool isBVH = false);

std::string Trim(std::string str);
#endif
