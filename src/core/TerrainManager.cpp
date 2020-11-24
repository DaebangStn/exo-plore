#include "core/TerrainManager.h"
#include "core/DARTHelper.h"


using namespace std;
using namespace MASS;
using namespace dart::dynamics;
using namespace dart::simulation;

TerrainManager::TerrainManager(const TerrainConfig& config): mConfig(config)
{
}

void TerrainManager::initialize(WorldPtr world) {
    mWorld = world;
}

void TerrainManager::reset(const Eigen::Vector3d& characterPos) {
    mTerrainEndZ = characterPos.z() - mConfig.width / 2.0;
    mTerrainEndY = characterPos.y() - 0.983;
    while (mTerrainSkeletons.size() > 0) removeLastTerrainSegment();
    update(characterPos);
}

void TerrainManager::update(const Eigen::Vector3d& characterPos) {
    double distanceToEnd = mTerrainEndZ - characterPos.z();
    
    if (distanceToEnd < mConfig.margin) {
        // cout << "Creating new terrain segment distanceToEnd: " << distanceToEnd << " terrainEndZ: " << mTerrainEndZ << " terrainEndY: " << mTerrainEndY << endl;    

        pair<double, double> newPos;
        if (mConfig.mode == TerrainMode::FLAT) {
            newPos = nextPosFlat();
        } else {
            newPos = nextPosStair();
        }
        
        auto terrain = createTerrainSegment(newPos.first, newPos.second);
        mTerrainSkeletons.push_back(terrain);
        mWorld->addSkeleton(terrain);
        
        // Optional: Remove old terrain segments that are far behind
        while (mTerrainSkeletons.size() > 3) removeLastTerrainSegment();
    }
}

SkeletonPtr TerrainManager::createTerrainSegment(double z, double y) {
    auto terrain = Skeleton::create("terrain_" + to_string(z));
    
    // Create box shape for terrain
    double height = 1.0;  // Thickness of terrain
    auto shape = MakeBoxShape(Eigen::Vector3d(mTerrainDepth, mTerrainBlockHeight, mConfig.width));
    Eigen::Isometry3d T_body = Eigen::Isometry3d::Identity();
	Eigen::Isometry3d T_pos = Eigen::Isometry3d::Identity();
	T_body.translation() = Eigen::Vector3d(0.0, y, z);
	Eigen::Isometry3d child_to_joint = T_body.inverse()*T_pos;
	Eigen::Isometry3d parent_to_joint = T_pos;
	auto* jnt_props = MakeWeldJointProperties("terrain", parent_to_joint, child_to_joint);
	dart::dynamics::Inertia inertia = MakeInertia(shape, 15.0);
	BodyNode* bn = MakeBodyNode(terrain, nullptr, jnt_props, "Weld", inertia);
	bn->createShapeNodeWith<VisualAspect, CollisionAspect, DynamicsAspect>(shape);
    bn->eachShapeNodeWith<VisualAspect>([&](ShapeNode* sn) {
        sn->getVisualAspect()->setColor(Eigen::Vector4d(0.1, 0.7, 0.5, 0.5));
    });    
    return terrain;
}

pair<double, double> TerrainManager::nextPosFlat() {
    double newZPos = mTerrainEndZ + mConfig.width / 2.0;
    double newYPos = mTerrainEndY - mTerrainBlockHeight / 2.0;
    
    mTerrainEndZ += mConfig.width;

    return make_pair(newZPos, newYPos);
}

pair<double, double> TerrainManager::nextPosStair() {
    double newZPos = mTerrainEndZ + mConfig.width / 2.0;
    double newYPos = mTerrainEndY - mTerrainBlockHeight / 2.0;
    mTerrainEndY += mConfig.stairHeight;
    mTerrainEndZ += mConfig.width;
    return make_pair(newZPos, newYPos);
}

void TerrainManager::removeLastTerrainSegment() {
    if (mTerrainSkeletons.size() > 0) {
        mWorld->removeSkeleton(mTerrainSkeletons[0]);
        mTerrainSkeletons.erase(mTerrainSkeletons.begin());
    }
    else cout << "No terrain segments to remove" << endl;
}
