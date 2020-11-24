#include "core/Contact.h"
#include "core/Utils.h"

using namespace dart::dynamics;
using namespace dart::simulation;

namespace MASS
{

Contact::Contact(const WorldPtr& wPtr, SkeletonPtr cPtr) : mDebouncer(0.25)
{
	mWorld = wPtr;
	mCharSkeleton = cPtr;

    mContactRight = false; mContactLeft = false;
	
	mContactBodiesR.push_back(mCharSkeleton->getBodyNode("TalusR"));
	mContactBodiesR.push_back(mCharSkeleton->getBodyNode("FootThumbR"));
	mContactBodiesR.push_back(mCharSkeleton->getBodyNode("FootPinkyR"));
	
	mContactBodiesL.push_back(mCharSkeleton->getBodyNode("TalusL"));
	mContactBodiesL.push_back(mCharSkeleton->getBodyNode("FootThumbL"));
	mContactBodiesL.push_back(mCharSkeleton->getBodyNode("FootPinkyL"));
}

void Contact::Initialize(dart::dynamics::BodyNode* ground)
{
	mGround = ground;
	mUseTerrainManager = false;
	Reset();
}

void Contact::Initialize(TerrainManager* terrainManager)
{
	mTerrainManager = terrainManager;
	mUseTerrainManager = true;
	Reset();
}

void Contact::Reset()
{
	isFirstAfterReset = true;
	mGRF_R = 0.0; mGRF_L = 0.0;
	mGRF_R_3d.setZero(); mGRF_L_3d.setZero();
}

bool Contact::isContact(bool LR) const {
    if (mUseDebouncer) {
        if (LR) return mDebouncer.isContactL();
        else return mDebouncer.isContactR();
    } else {
        return LR ? mContactRight : mContactLeft;
    }
}

void Contact::Set()
{
	if(isFirstAfterReset){
		isFirstAfterReset = false;
		return;
	}
	mGRF_R_3d.setZero();
	mGRF_L_3d.setZero();	

	const dart::collision::CollisionResult& result = mWorld->getLastCollisionResult();

    mContactRight = mContactLeft = false;
    bool isTalusR = false, isToeR = false;
    bool isTalusL = false, isToeL = false;

    for(const auto& contact : result.getContacts())
    {
        const ShapeFrame* sf1 = contact.collisionObject1->getShapeFrame();
        const ShapeFrame* sf2 = contact.collisionObject2->getShapeFrame();
        
        // Check if either shape is ground
        bool contacted = false;
        const ShapeFrame* otherShape = nullptr;
        
        // Convert ShapeFrame to ShapeNode to access the parent BodyNode
        const auto* sn1 = dynamic_cast<const dart::dynamics::ShapeNode*>(sf1);
        const auto* sn2 = dynamic_cast<const dart::dynamics::ShapeNode*>(sf2);
        
        if (isGround(sn1)) {
            contacted = true;
            otherShape = sf2;
        } else if (isGround(sn2)) {
            contacted = true;
            otherShape = sf1;
        }

        if (!contacted) continue;

        // Convert otherShape to ShapeNode
        const auto* otherShapeNode = dynamic_cast<const dart::dynamics::ShapeNode*>(otherShape);
        if (!otherShapeNode) continue;

        // Check contact with right foot bodies
        for(const auto& curBody : mContactBodiesR)
        {
            if (otherShapeNode->getBodyNodePtr() == curBody) {
                mContactRight = true;
                mGRF_R_3d += contact.force;
                std::string name = curBody->getName();
                if(name == "FootPinkyR" || name == "FootThumbR") isToeR = true;
                if(name == "TalusR") isTalusR = true;
                break;
            }
        }
        // Check contact with left foot bodies
        for(const auto& curBody : mContactBodiesL)
        {
            if (otherShapeNode->getBodyNodePtr() == curBody) {
                mContactLeft = true;
                mGRF_L_3d += contact.force;
                std::string name = curBody->getName();
                if(name == "FootPinkyL" || name == "FootThumbL") isToeL = true;
                if(name == "TalusL") isTalusL = true;
                break;
            }
        }
    }
    mGRF_R = mGRF_R_3d.norm();
    mGRF_L = mGRF_L_3d.norm();
    mDebouncer.updateR(mContactRight);
    mDebouncer.updateL(mContactLeft);
}

bool Contact::isGround(const dart::dynamics::ShapeNode* sn) const {
    if (mUseTerrainManager) {
        return sn && sn->getBodyNodePtr() && sn->getBodyNodePtr()->getName().find("terrain") != std::string::npos;
    } else {
        return sn && sn->getBodyNodePtr() && sn->getBodyNodePtr() == mGround;
    }
}

}