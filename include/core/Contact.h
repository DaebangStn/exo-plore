#ifndef __MASS_CONTACT_H__
#define __MASS_CONTACT_H__

#include "dart/dart.hpp"
#include "core/Utils.h"
#include "core/TerrainManager.h"

#define CONTACT_RIGHT 0
#define CONTACT_LEFT 1


namespace MASS
{
class Debouncer
{
public:
    Debouncer(double alpha) : mContactR(alpha), mContactL(alpha) {}
    bool updateR(bool contact)
    {
        mContactR.update(contact);
        return mContactR.get() > criteria;
    }
    bool updateL(bool contact)
    {
        mContactL.update(contact);
        return mContactL.get() > criteria;
    }
    void setAlpha(double alpha){mContactR.alpha = alpha; mContactL.alpha = alpha;}
    double getAlpha() const {return mContactR.alpha;}
    bool isContactR() const {return mContactR.get() > criteria;}
    bool isContactL() const {return mContactL.get() > criteria;}
private:
    const double criteria = 0.5;
    Utils::MAFilter mContactR, mContactL;
};

class Contact
{
public:
    Contact(const dart::simulation::WorldPtr& wPtr, dart::dynamics::SkeletonPtr cPtr);
    ~Contact() = default;

    void Initialize(dart::dynamics::BodyNode* ground);
    void Initialize(TerrainManager* terrainManager);
    void Reset();    
    void Set();
    void SetContact(bool r, bool l){mContactRight = r; mContactLeft = l;};
    void setUseDebouncer(bool use){mUseDebouncer = use;};
    void setDebouncerAlpha(double alpha){mDebouncer.setAlpha(alpha);};
    bool isContact(bool LR) const;
    bool getUseDebouncer() const {return mUseDebouncer;};
    double getDebouncerAlpha() const {return mDebouncer.getAlpha();};
    double GetGRF(bool LR) const {return LR ? mGRF_L : mGRF_R;};
    Eigen::Vector3d GetGRF_3d(bool LR) const {return LR ? mGRF_L_3d : mGRF_R_3d;};

private:
    bool isGround(const dart::dynamics::ShapeNode* shapeFrame) const;

    bool mUseTerrainManager = false;
    dart::simulation::WorldPtr mWorld;
    dart::dynamics::BodyNode* mGround;
    TerrainManager* mTerrainManager;
    dart::dynamics::SkeletonPtr mCharSkeleton;
    double mGRF_R = 0.0, mGRF_L = 0.0;
    Eigen::Vector3d mGRF_R_3d = Eigen::Vector3d::Zero(), mGRF_L_3d = Eigen::Vector3d::Zero();

    std::vector<dart::dynamics::BodyNode*> mContactBodiesR, mContactBodiesL;

    bool mContactRight, mContactLeft, isFirstAfterReset;
    bool mUseDebouncer = true;
    Debouncer mDebouncer;
};

}

#endif
