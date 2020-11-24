#ifndef __MASS_BVH_H__
#define __MASS_BVH_H__
#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include "dart/dart.hpp"

using namespace dart::dynamics;
namespace MASS
{

class BVHNode
{
public:
	enum CHANNEL
	{
		Xpos=0, Ypos=1, Zpos=2,	Xrot=3,	Yrot=4,	Zrot=5
	}; 

	std::map<std::string,CHANNEL> CHANNEL_NAME =
	{
		{"Xposition",Xpos},
		{"XPOSITION",Xpos},
		{"Yposition",Ypos},
		{"YPOSITION",Ypos},
		{"Zposition",Zpos},
		{"ZPOSITION",Zpos},
		{"Xrotation",Xrot},
		{"XROTATION",Xrot},
		{"Yrotation",Yrot},
		{"YROTATION",Yrot},
		{"Zrotation",Zrot},
		{"ZROTATION",Zrot}
	};	

public:
	BVHNode(const std::string& name,BVHNode* parent);
	void AddChild(BVHNode* child);
	void SetChannel(int c_offset,std::vector<std::string>& c_name);
	void Set(const Eigen::VectorXd& m_t);
	void Set(const Eigen::Matrix3d& R_t);
	Eigen::Matrix3d Get();	
	BVHNode* GetNode(const std::string& name);
	int GetChannelOffset() {return mChannelOffset;}
	
private:
	BVHNode* mParent;
	std::string mName;
	std::vector<BVHNode*> mChildren;
	Eigen::Matrix3d mR;
	
	int mChannelOffset;
	int mNumChannels;
	std::vector<BVHNode::CHANNEL> mChannel;
};

class BVH
{
public:
	std::vector<Eigen::VectorXd> mPureMotions;
	std::vector<Eigen::VectorXd> mMirrorMotions;
	std::vector<Eigen::VectorXd> mModifiedMotions;

public:
	BVH(const SkeletonPtr& skel, const std::map<std::string,std::string>& bvh_map);
	~BVH();

	void Parse(const std::string& file,bool cyclic=true);

	Eigen::VectorXd GetMotion(double t);
	Eigen::VectorXd GetMotion(double t, SkeletonPtr skeleton);
	Eigen::VectorXd GetModifiedMotion(double t);
	Eigen::Matrix3d Get(const std::string& bvh_node);

	double GetMaxTime(){return (mNumTotalFrames)*mTimeStep;}
	double GetTimeStep(){return mTimeStep;}
	const std::map<std::string,std::string>& GetBVHMap(){return mBVHMap;}
	const Eigen::Isometry3d& GetT0(){return T0;}
	const Eigen::Isometry3d& GetT1(){return T1;}
	bool IsCyclic(){return mCyclic;}

	//Modified Begin
	void setMirrorMotions(std::vector<Eigen::VectorXd> mirrorMotions){ mMirrorMotions = mirrorMotions; }
	void blendMirrorMotion();
	std::vector<Eigen::VectorXd>& GetModifiedMotions() {return mModifiedMotions;}
	
	//Modified End
	Eigen::VectorXd toPureMotion(Eigen::VectorXd m);
	Eigen::VectorXd toPureMotion(Eigen::VectorXd m, SkeletonPtr skeleton);

	void SetPureMotions();
	void ResetModifiedMotions();
	void print();
	int GetTotalFrames() {return mNumTotalFrames;}
	BVHNode* ReadHierarchy(BVHNode* parent,const std::string& name,int& channel_offset,std::ifstream& is);

private:
	BVHNode* mRoot;
	SkeletonPtr mSkeleton;
	
	Eigen::Isometry3d T0,T1;
	std::map<std::string, BVHNode*> mMap;
	std::map<std::string, std::string> mBVHMap;
	std::vector<Eigen::VectorXd> mMotions;
		
	int mNumTotalChannels;
	int mNumTotalFrames;

	bool mCyclic;
	double mTimeStep;
	double mScalef;
};

};

#endif