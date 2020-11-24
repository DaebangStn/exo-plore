#ifndef __SIM_TERRAIN_MANAGER_H__
#define __SIM_TERRAIN_MANAGER_H__

#include <dart/dart.hpp>

enum class TerrainMode {
    FLAT,
    STAIR
};

struct TerrainConfig {
    double width = 0.5;
    double margin = 0.2;
    TerrainMode mode = TerrainMode::FLAT;
    double stairHeight = 0.2; // Only used in STAIR mode
};

class TerrainManager {
public:
    TerrainManager(const TerrainConfig& config);
    ~TerrainManager() = default;

    void initialize(dart::simulation::WorldPtr world);
    void reset(const Eigen::Vector3d& characterPos);
    void update(const Eigen::Vector3d& characterPos);
    const TerrainConfig& getConfig() const { return mConfig; }
    void setConfig(const TerrainConfig& config) { mConfig = config; }
    
    const std::vector<dart::dynamics::SkeletonPtr>& getTerrainSkeletons() const { return mTerrainSkeletons; }

private:
    dart::dynamics::SkeletonPtr createTerrainSegment(double z, double y);
    void removeLastTerrainSegment();
    
    // Returns the next position and update terrain end position
    std::pair<double, double> nextPosFlat();
    std::pair<double, double> nextPosStair();

    TerrainConfig mConfig;
    dart::simulation::WorldPtr mWorld;
    std::vector<dart::dynamics::SkeletonPtr> mTerrainSkeletons;
    
    double mTerrainEndZ = 0.0;
    double mTerrainEndY = 0.0; // terrain height
    double mTerrainBlockHeight = 1.0;
    double mTerrainDepth = 5.0;
};

#endif // __SIM_TERRAIN_MANAGER_H__ 