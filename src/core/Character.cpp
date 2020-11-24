#include "core/Character.h"
#include "core/DARTHelper.h"
#include "core/Muscle.h"
#include "core/BVH.h"
#include "util/path.h"
#include <chrono>

using namespace std;
namespace MASS
{
    // const unordered_set<string_view> lengthHyperlordosisGp = {
    //     "R_Psoas_Major", "L_Psoas_Major", "R_Psoas_Major1", "L_Psoas_Major1", "R_Psoas_Major2", "L_Psoas_Major2"
    // };
    // const unordered_set<string_view> lengthEquinusGp = {
    //     "R_Soleus", "L_Soleus", "R_Soleus1", "L_Soleus1", "R_Tibialis_Posterior", "L_Tibialis_Posterior",
    //     "R_Gastrocnemius_Lateral_Head", "L_Gastrocnemius_Lateral_Head"
    // };
    // const unordered_set<string_view> forceCalcanealGp = {
    //     "R_Soleus", "L_Soleus", "R_Soleus1", "L_Soleus1", "R_Tibialis_Posterior", "L_Tibialis_Posterior",
    //     "R_Gastrocnemius_Lateral_Head", "L_Gastrocnemius_Lateral_Head", "R_Gastrocnemius_Medial_Head", "L_Gastrocnemius_Medial_Head"
    // };
    // const unordered_set<string_view> forceWaddlingGp = {
    //     "R_Gluteus_Medius", "L_Gluteus_Medius", "R_Gluteus_Medius1", "L_Gluteus_Medius1", "R_Gluteus_Medius2", 
    //     "L_Gluteus_Medius2", "R_Gluteus_Medius3", "L_Gluteus_Medius3", "R_Gluteus_Minimus", "L_Gluteus_Minimus", 
    //     "R_Gluteus_Minimus1", "L_Gluteus_Minimus1", "R_Gluteus_Minimus2", "L_Gluteus_Minimus2", "R_Tensor_Fascia_Lata", 
    //     "L_Tensor_Fascia_Lata", "R_Tensor_Fascia_Lata1", "L_Tensor_Fascia_Lata1", "R_Tensor_Fascia_Lata2", "L_Tensor_Fascia_Lata2"
    // };
    // const unordered_set<string_view> forceFootdropGp = {
    //     "R_Tibialis_Anterior", "R_Extensor_Hallucis_Longus", "R_Peroneus_Tertius", "R_Peroneus_Tertius1",
    //     "R_Extensor_Digitorum_Longus", "R_Extensor_Digitorum_Longus1", "R_Extensor_Digitorum_Longus2", "R_Extensor_Digitorum_Longus3",
    //     "L_Tibialis_Anterior", "L_Extensor_Hallucis_Longus", "L_Peroneus_Tertius", "L_Peroneus_Tertius1",
    //     "L_Extensor_Digitorum_Longus", "L_Extensor_Digitorum_Longus1", "L_Extensor_Digitorum_Longus2", "L_Extensor_Digitorum_Longus3"
    // };

    const unordered_map<string, unordered_set<string_view>> muscleGroups = {
        {"leg", {
            "Adductor_Brevis", "Adductor_Longus", "Adductor_Magnus", 
            "Bicep_Femoris", 
            "Extensor_Digitorum_Longus", "Extensor_Hallucis_Longus",
            "Flexor_Digiti_Minimi_Brevis_Foot", "Flexor_Digitorum_Longus", "Flexor_Hallucis",
            
            "Rectus_Femoris", "Vastus",
            "Sartorius", "Semimembranosus", "Semitendinosus",
            "Gastrocnemius", "Soleus",
            
            "Tibialis_Anterior", "Tibialis_Posterior",
            
            "Gluteus", 
            "iliacus", "Psoas_Major",
            
            "Gracilis", "Inferior_Gemellus", "Obturator_Externus", "Obturator_Internus", "Pectineus", "Peroneus_Brevis",
            "Peroneus_Longus", "Peroneus_Tertius", "Piriformis", "Plantaris",
            "Popliteus", "Quadratus_Femoris",
            "Superior_Gemellus", "Tensor_Fascia_Lata"
        }},
        {"arm", {
            "Abductor_Pollicis_Longus", "Anconeous", "Bicep_Brachii_Long_Head",
            "Bicep_Brachii_Short_Head", "Brachialis", "Brachioradialis", "Coracobrachialis",
            "Deltoid", "Extensor_Carpi_Radialis_Longus", "Extensor_Carpi_Ulnaris",
            "Extensor_Digiti_Minimi", "Extensor_Digitorum", "Extensor_Pollicis_Brevis",
            "Extensor_Pollicis_Longus", "Flexor_Carpi_Radialis", "Flexor_Carpi_Ulnaris",
            "Flexor_Digitorum_Profundus", "Flexor_Pollicis_Longus", "Flexor_Radialis_Longus",
            "Flexor_Radialis_Short", "Flexor_Ulnaris_Longus", "Forearm_Flexors",
            "Forearm_Extensors", "Infraspinatus", "Latissimus_Dorsi", "Levator_Scapulae",
            "Longissimus_Capitis", "Longissimus_Thoracis", "Longus_Capitis", "Multifidus",
            "Omnibus", "Orbicularis_Oris", "Pectoralis_Major", "Pectoralis_Minor",
            "Rhomboid_Major", "Rhomboid_Minor", "Serratus_Anterior", "Subclavian",
            "Subscapularis", "Supraspinatus", "Teres_Major", "Triceps_Lateral_Head",
            "Triceps_Long_Head", "Triceps_Medial_Head"
        }},
        {"torso", {
            "ab", "Internal_Oblique", "Longissimus_Capitis", "Longissimus_Thoracis", "Longus_Capitis",
            "Multifidus", "Platysma", "Quadratus_Lumborum", "Scalene_Anterior", "Scalene_Middle",
            "Semispinalis_Capitis", "Splenius_Capitis", "Splenius_Cervicis", "Sternocleidomastoid",
            "iliocostalis", "Rectus_Abdominis", "Serratus_Posterior_Inferior", "Transversus_Abdominis",
            "Transversus_Abdominis", "Trapezius"
        }},
        {"quadriceps", {
            "Rectus_Femoris", "Vastus"
        }},
        {"hamstrings", {
            "Bicep_Femoris", "Semitendinosus", "Semimembranosus"
        }},

        {"he", {
            "Gluteus_Maximus", 
            "Adductor_Magnus",
            "Bicep_Femoris", "Semitendinosus", "Semimembranosus"
        }},
        {"hf", {
            "iliacus", "Psoas_Major", "Rectus_Femoris", 
            "Adductor_Brevis", "Adductor_Longus", 
            "Sartorius", "Pectineus"
        }},
        {"ke", {
            "Rectus_Femoris", "Vastus"
        }},
        {"kf", {
            "Gastrocnemius", 
            "Bicep_Femoris", "Semitendinosus", "Semimembranosus",
            "Popliteus", "Gracilis", "Sartorius"
        }},
        {"ae", {
            "Soleus", "Gastrocnemius", "Tibialis_Posterior", 
            "Flexor_Hallucis", "Flexor_Digitorum_Longus", 
            "Plantaris", "Peroneus_Longus", "Peroneus_Brevis" 
        }},
        {"af", {
            "Tibialis_Anterior", "Extensor_Digitorum_Longus", "Extensor_Hallucis_Longus"
        }},
        {"fl", {
            "iliacus", "Psoas_Major",
            "Tibialis_Anterior", "Extensor_Digitorum_Longus", "Extensor_Hallucis_Longus", 
            "Peroneus_Tertius",
        }}
    };

    Character::Character(): mSkeleton(nullptr), mBVH(nullptr), mTc(Eigen::Isometry3d::Identity()), motionScale(1.0){}

    Character::~Character(){
        delete mBVH;

        for(auto m : mMuscles) delete m;
        for(auto sm : mStdMuscles) delete sm;
    }

    void Character::SetCharacterInformation()
    {
        Eigen::VectorXd p_bkup = mSkeleton->getPositions();
        Eigen::VectorXd v_bkup = mSkeleton->getVelocities();

        int dofs = mSkeleton->getNumDofs();
        Eigen::VectorXd p_zero = Eigen::VectorXd::Zero(dofs);
        Eigen::VectorXd v_zero = Eigen::VectorXd::Zero(dofs);

        mSkeleton->setPositions(p_zero);
        mSkeleton->setVelocities(v_zero);
        mSkeleton->computeForwardKinematics(true, true, false);

        BodyNode* head = mSkeleton->getBodyNode("Head");
        BodyNode* talusR = mSkeleton->getBodyNode("TalusR");
        double headSizeY, talusSizeY;
        head->eachShapeNodeWith<dart::dynamics::VisualAspect>([&headSizeY](ShapeNode* sn) {
            if (auto box = dynamic_cast<const BoxShape*>(sn->getShape().get())) {
                headSizeY = box->getSize()[1];
                return false;
            }
            return true;
        });
        talusR->eachShapeNodeWith<dart::dynamics::VisualAspect>([&talusSizeY](ShapeNode* sn) {
            if (auto box = dynamic_cast<const BoxShape*>(sn->getShape().get())) {
                talusSizeY = box->getSize()[1];
                return false;
            }
            return true;
        });

        mTalusSize = talusSizeY;
 
        double headHeight = head->getCOM()[1];
        double talusHeight = talusR->getCOM()[1];

        double height = 0.5*(headSizeY + talusSizeY) + (headHeight - talusHeight);
        mCharacterHeight = height;
        mCharacterWeight = mSkeleton->getMass();

        mSkeleton->setPositions(p_bkup);
        mSkeleton->setVelocities(v_bkup);
        mSkeleton->computeForwardKinematics(true, true, false);
    }

    void Character::LoadSkeleton(const string &path, bool create_obj, double damping)
    {
        mUseOBJ = create_obj;
        mSkeleton = BuildFromFile(path, create_obj, damping);
        mStdSkeleton = BuildFromFile(path, create_obj, damping);

        int dof = mSkeleton->getNumDofs();
        mKp = Eigen::VectorXd::Zero(dof);
        mKv = Eigen::VectorXd::Zero(dof);
        mDamping = Eigen::VectorXd::Zero(dof);

        for(BodyNode* bodynode : mSkeleton->getBodyNodes()) modifyLog[bodynode] = ModifyInfo();

        SetCharacterInformation();
        rootDefaultHeight = mSkeleton->getRootBodyNode()->getTransform().translation()[1];
        yOffset = -1e18;
        for(const auto& foot : {"TalusR", "TalusL"}) yOffset = max(yOffset, -mSkeleton->getBodyNode(foot)->getCOM()[1] + mTalusSize);

        map<string, string> bvh_map;
        TiXmlDocument doc;
        doc.LoadFile(path.c_str());
        TiXmlElement *skel_elem = doc.FirstChildElement("Skeleton");
        for (TiXmlElement *node = skel_elem->FirstChildElement("Node"); node != nullptr; node = node->NextSiblingElement("Node"))
        {
            if (node->Attribute("endeffector") != nullptr)
            {
                string ee = node->Attribute("endeffector");
                if (ee == "True")
                {
                    mEndEffectors.push_back(mSkeleton->getBodyNode(string(node->Attribute("name"))));
                }
            }

            TiXmlElement *joint_elem = node->FirstChildElement("Joint");
            if (joint_elem->Attribute("bvh") != nullptr)
            {
                bvh_map.insert(make_pair(node->Attribute("name"), joint_elem->Attribute("bvh")));
            }
            int dof = mSkeleton->getJoint(node->Attribute("name"))->getNumDofs();
            if(dof!=0)
            {
                int idx = mSkeleton->getJoint(node->Attribute("name"))->getIndexInSkeleton(0);
                if (joint_elem->Attribute("kp") != nullptr)
                {
                    if(dof == 3) mKp.segment(idx, dof) = string_to_vector3d(joint_elem->Attribute("kp"));
                    else if (dof == 1) mKp[idx] = stod(joint_elem->Attribute("kp"));
                }

                if (joint_elem->Attribute("damping") != nullptr)
                {
                    if(dof == 3) mDamping.segment(idx, dof) = string_to_vector3d(joint_elem->Attribute("damping"));
                    else if (dof == 1) mDamping[idx] = stod(joint_elem->Attribute("damping"));
                    else if (dof == 6)
                    {
                        double dp = stod(joint_elem->Attribute("damping"));
                        mDamping[idx] = dp; mDamping[idx+1] = dp; mDamping[idx+2] = dp;
                        mDamping[idx+3] = dp; mDamping[idx+4] = dp; mDamping[idx+5] = dp;
                    }
                }
            }
        }
        for(int i=0; i<mKp.size(); i++) mKv[i] = sqrt(mKp[i]*2);
        mBVH = new BVH(mSkeleton, bvh_map);

        // Make Symmetry Joint Pair (For Symmetry)
        for (dart::dynamics::Joint *jn : mSkeleton->getJoints())
        {
            if (jn->getName()[jn->getName().size() - 1] == 'R') continue;
            if (jn->getName()[jn->getName().size() - 1] == 'L'){
                for (dart::dynamics::Joint *jn_2 : mSkeleton->getJoints()){
                    if ((jn_2->getName().substr(0, jn_2->getName().size() - 1) == jn->getName().substr(0, jn->getName().size() - 1)) && (jn_2->getName() != jn->getName())){
                        mPairs.push_back(std::make_pair(jn, jn_2));

                        Eigen::AngleAxisd first_axis = Eigen::AngleAxisd(jn->getChildBodyNode()->getTransform().linear());
                        first_axis.axis()[1] *= -1;
                        first_axis.axis()[2] *= -1;

                        Eigen::Matrix3d first_rot = first_axis.toRotationMatrix();
                        Eigen::Matrix3d second_rot = jn_2->getChildBodyNode()->getTransform().linear();

                        mBodyNodeTransform.push_back(first_rot.transpose() * second_rot);

                        break;
                    }
                }
            } else {
                mPairs.push_back(std::make_pair(jn, jn));
                mBodyNodeTransform.push_back(Eigen::Matrix3d::Identity());
            }
        }
    }

    void Character::LoadMuscleYaml(const YAML::Node& node){
    try {
        bool fast_mode = node["fast_mode"].as<bool>();
        bool new_anchor = node["new_anchor"] ? node["new_anchor"].as<bool>() : false;
        int muscle_type = node["type"].as<int>();
        string muscle_file_path = path_rel_to_abs(node["file"].as<string>()).string();
        int mass_type_int = node["mass_type"].as<int>();
        mMuscleMassType = static_cast<MassType>(mass_type_int);
        mLenRatio = node["extension_len_ratio_clip"].as<double>();
        double shortening_multiplier = node["shortening_multiplier"] ? node["shortening_multiplier"].as<double>() : 1.0;

        TiXmlDocument doc;
        if (doc.LoadFile(muscle_file_path.c_str()))
        {
            cout << "Can't open file : " << muscle_file_path << endl;
            return;
        }

        TiXmlElement *muscledoc = doc.FirstChildElement("Muscle");
        for (TiXmlElement *unit = muscledoc->FirstChildElement("Unit"); unit != nullptr; unit = unit->NextSiblingElement("Unit"))
        {
            string name = unit->Attribute("name");
            double f0 = stod(unit->Attribute("f0"));
            double lm = stod(unit->Attribute("lm"));
            double lt = stod(unit->Attribute("lt"));
            double pa = stod(unit->Attribute("pen_angle"));
            double lmax = -0.1;
            double type1_fraction = 0.5;
            if(unit->Attribute("type1_fraction")!=nullptr) type1_fraction = stod(unit->Attribute("type1_fraction"));

            bool use_velocity_state = true;
            Muscle* muscle_elem = new Muscle(name, f0, lm, lt, pa, lmax, type1_fraction, use_velocity_state, mLenRatio, muscle_type, mMuscleMassType, fast_mode);
            Muscle* stdmuscle_elem = new Muscle(name, f0, lm, lt, pa, lmax, type1_fraction, use_velocity_state, mLenRatio, muscle_type, mMuscleMassType, fast_mode);

            bool isValid = true;
            int num_waypoints = 0;
            for (TiXmlElement *waypoint = unit->FirstChildElement("Waypoint"); waypoint != nullptr; waypoint = waypoint->NextSiblingElement("Waypoint")) num_waypoints++;
            int i = 0;
            for (TiXmlElement *waypoint = unit->FirstChildElement("Waypoint"); waypoint != nullptr; waypoint = waypoint->NextSiblingElement("Waypoint"))
            {
                string body = waypoint->Attribute("body");
                if (mSkeleton->getBodyNode(body) == nullptr) {
                    isValid = false;
                    break;
                }

                Eigen::Vector3d glob_pos = string_to_vector3d(waypoint->Attribute("p"));
                if (i == 0 || i == num_waypoints - 1){
                    muscle_elem->AddSingleBodyAnchor(mSkeleton->getBodyNode(body), glob_pos);
                    stdmuscle_elem->AddSingleBodyAnchor(mStdSkeleton->getBodyNode(body),glob_pos);
                }
                else{
                    if (new_anchor) muscle_elem->AddMeshLbsAnchor(mSkeleton, glob_pos);
                    else muscle_elem->AddPrevMeshLbsAnchor(mSkeleton, glob_pos);
                    if (new_anchor) stdmuscle_elem->AddMeshLbsAnchor(mStdSkeleton, glob_pos);
                    else stdmuscle_elem->AddPrevMeshLbsAnchor(mStdSkeleton, glob_pos);
                }

                i++;
            }

            if (isValid){
                muscle_elem->SetMuscle();
                muscle_elem->SetHz(mSimulationHz, mControlHz);
                muscle_elem->SetShorteningMultiplier(shortening_multiplier);

                if (muscle_elem->GetRelDofs() > 0)
                {
                    stdmuscle_elem->SetMuscle();
                    stdmuscle_elem->SetShorteningMultiplier(shortening_multiplier);
                    mMuscles.push_back(muscle_elem);
                    mStdMuscles.push_back(stdmuscle_elem);
                    mMusclesMap[name] = muscle_elem;
                }
            }else cout << "[Warning] Muscle " << name << " is not valid" << endl;
        }

        mMuscleMass = Eigen::VectorXd::Zero(mMuscles.size());
        mArmMask = Eigen::VectorXd::Zero(mMuscles.size());
        mLegMask = Eigen::VectorXd::Zero(mMuscles.size());
        mTorsoMask = Eigen::VectorXd::Zero(mMuscles.size());
        for(int i = 0; i < mMuscles.size(); i++) {
            mMuscleMass[i] = mMuscles[i]->GetMass();
            const auto muscle_type = mMuscles[i]->getType();
            mArmMask[i] = (muscle_type.has(MuscleType::arm)) ? 1.0 : 0.0;
            mLegMask[i] = (muscle_type.has(MuscleType::leg)) ? 1.0 : 0.0;
            mTorsoMask[i] = (muscle_type.has(MuscleType::torso)) ? 1.0 : 0.0;
        }
        categorizeMuscles();
        }
        catch(const std::exception& e) {
            std::cerr << "[Character] Error loading muscle yaml: " << e.what() << std::endl;
        }
        
        SetMuscleMassType(mMuscleMassType);
    }

    void Character::SetMuscleMassType(MassType mass_type){
        for (auto muscle : mMuscles) muscle->SetMassType(mass_type);
    }

    void Character::SetMuscleMassType(int mass_type){
        mMuscleMassType = static_cast<MassType>(mass_type);
        for (auto muscle : mMuscles) muscle->SetMassType(mMuscleMassType);
    }

    void Character::LoadBVH(const string &path, bool cyclic)
    {
        if (mBVH == nullptr)
        {
            cout << "Finalize_Initialization BVH class first" << endl;
            return;
        }
        mBVH->Parse(path, cyclic);
    }

    void Character::Reset()
    {
        mSkeleton->clearConstraintImpulses();
        mSkeleton->clearInternalForces();
        mSkeleton->clearExternalForces();

        mTc = mBVH->GetT0();
        mTc.translation()[1] = 0.0;
    }

    void Character::setActivation(const Eigen::VectorXd &a)
    {
        if(mUseMuscle) {
            for(int i = 0; i < mMuscles.size(); i++) {
                #ifdef LOG_VERBOSE
                if (i==0) {
                    auto start = chrono::high_resolution_clock::now();
                    mMuscles[i]->SetActivation(a[i]);
                    static double dur1 = 0.0;
                    auto point1 = chrono::high_resolution_clock::now();
                    dur1 = dur1 * 0.99 + chrono::duration_cast<chrono::nanoseconds>(point1 - start).count() * 0.01;
                    mMuscles[i]->ApplyForceToBody();
                    static double dur3 = 0.0;
                    auto point3 = chrono::high_resolution_clock::now();
                    dur3 = dur3 * 0.99 + chrono::duration_cast<chrono::nanoseconds>(point3 - point1).count() * 0.01;
                    cout << "        set activation: " << dur1 << " ns" << endl;
                    cout << "        apply force: " << dur3 << " ns" << endl;
                } else {
                #endif
                    mMuscles[i]->SetActivation(a[i]);
                    mMuscles[i]->ApplyForceToBody();
                #ifdef LOG_VERBOSE
                }
                #endif
            }
        }else cout << "[Warning] Character::SetActivation is called even though muscle is not used" << endl;
    }

    void Character::_fixUpper(){
        // Define all joints that need to be fixed
        const vector<string> fixed_joints = {
            "Pelvis",      // Root
            "Spine",       // Upper body
            "Torso", 
            "Neck",
            "Head",
            "ShoulderR", "ArmR", "ForeArmR", "HandR",  // Right arm
            "ShoulderL", "ArmL", "ForeArmL", "HandL"   // Left arm
        };
        // Fix all specified joints
        for (const auto& joint_name : fixed_joints) {
            if (auto joint = mSkeleton->getJoint(joint_name)) {
                Eigen::VectorXd zero_vel = Eigen::VectorXd::Zero(joint->getNumDofs());
                joint->setVelocities(zero_vel);
            }
            // if (auto joint = mSkeleton->getJoint(joint_name)) joint->setActuatorType(dart::dynamics::Joint::LOCKED);
        }
    }

    void Character::_fixWhole(){
        for (auto joint : mSkeleton->getJoints()) {
            Eigen::VectorXd zero_vel = Eigen::VectorXd::Zero(joint->getNumDofs());
            joint->setVelocities(zero_vel);
        }
        // for (auto joint : mSkeleton->getJoints()) joint->setActuatorType(dart::dynamics::Joint::LOCKED);
    }

    void Character::SetAnchor(bool FixUpperOnly) {
        if(FixUpperOnly) _fixUpper();
        else _fixWhole();

        // Store current positions
        auto positions = mSkeleton->getPositions();
        
        // Ensure root position is set correctly
        positions[4] = 0.08;  // Set y-position as before
        mSkeleton->setPositions(positions);
    }

    void Character::ResetAnchor() {
        // Reset all joints back to FORCE type
        for (auto joint : mSkeleton->getJoints()) joint->setActuatorType(dart::dynamics::Joint::FORCE);
        Reset();
    }

    Eigen::VectorXd Character::GetSPDForces(const Eigen::VectorXd &p_desired) const
    {
        Eigen::VectorXd q = mSkeleton->getPositions();
        Eigen::VectorXd dq = mSkeleton->getVelocities();
        double dt = mSkeleton->getTimeStep();

        Eigen::MatrixXd M_inv = (mSkeleton->getMassMatrix() + Eigen::MatrixXd(dt * mKv.asDiagonal())).inverse();
        Eigen::VectorXd qdqdt = q + dq * dt;

        Eigen::VectorXd p_diff = mKp.cwiseProduct(mSkeleton->getPositionDifferences(qdqdt, p_desired));
        Eigen::VectorXd v_diff = mKv.cwiseProduct(dq);

        Eigen::VectorXd coriolis = mSkeleton->getCoriolisAndGravityForces();
        Eigen::VectorXd const_forces = mSkeleton->getConstraintForces();

        Eigen::VectorXd ddq = - coriolis - p_diff - v_diff + const_forces;
        ddq = M_inv * ddq;
        Eigen::VectorXd tau = - p_diff - v_diff - dt*mKv.cwiseProduct(ddq);
        Eigen::VectorXd ext_forces = mSkeleton->getForces(); // exo assist and ankle spring forces
        tau -= ext_forces;
        return tau;
    }

    Eigen::VectorXd Character::GetTargetPositions(double t, double dt)
    {
        Eigen::VectorXd p = mBVH->GetModifiedMotion(t);
        if (mBVH->IsCyclic()) {
            int k = (int)(t / mBVH->GetMaxTime());
            Eigen::Vector3d bvh_vec = mBVH->GetT1().translation() - mBVH->GetT0().translation();
            bvh_vec[1] = 0.;
            p.segment<3>(3) += k * bvh_vec;
        }

        p.segment<3>(3) -= mSkeleton->getRootJoint()->getTransformFromParentBodyNode().translation();
        p[3] *= motionScale;
        p[4] += yOffset+0.03;
        p[5] *= motionScale;
        return p;
    }

    pair<Eigen::VectorXd, Eigen::VectorXd> Character::GetTargetPosAndVel(double t, double dt){
        Eigen::VectorXd p = GetTargetPositions(t, dt);
        Eigen::VectorXd p1 = GetTargetPositions(t + dt, dt);
        return make_pair(p, mSkeleton->getPositionDifferences(p1, p) / dt);
    }

    Eigen::VectorXd Character::GetTargetPositions(double t, double dt, dart::dynamics::SkeletonPtr skeleton)
    {
        Eigen::VectorXd p = mBVH->GetMotion(t, skeleton);
        if (mBVH->IsCyclic())
        {
            int k = (int)(t / mBVH->GetMaxTime());
            Eigen::Vector3d bvh_vec = mBVH->GetT1().translation() - mBVH->GetT0().translation();
            bvh_vec[1] = 0.;
            p.segment<3>(3) += k * bvh_vec;
        }
        p.segment<3>(3) -= mSkeleton->getRootJoint()->getTransformFromParentBodyNode().translation();
        return p;
    }

    pair<Eigen::VectorXd, Eigen::VectorXd> Character::GetTargetPosAndVel(double t, double dt, dart::dynamics::SkeletonPtr skeleton)
    {
        Eigen::VectorXd p = GetTargetPositions(t, dt, skeleton);
        Eigen::VectorXd p1 = GetTargetPositions(t + dt, dt, skeleton);

        return make_pair(p, skeleton->getPositionDifferences(p1, p) / dt);
    }

    Eigen::VectorXd Character::GetMirrorMomentum(const Eigen::VectorXd& Momentum)
    {
        int size = Momentum.size();
        Eigen::VectorXd mirrorMomentum = Eigen::VectorXd::Zero(size);

        int bodyNum = mSkeleton->getNumBodyNodes();
        for(int i=0; i<bodyNum; i++)
        {
            Eigen::Vector3d m = Momentum.segment(i*3, 3);
            if(i==0 || i>= 11)
            {
                m[0] *=  1;
                m[1] *= -1;
                m[2] *= -1;
            }
            else
            {
                m[0] *=  1;
                m[1] *=  1;
                m[2] *=  1;
            }

            mirrorMomentum.segment(i*3, 3) = m;
        }

        int r_idx = 1;
        int l_idx = 6;
        int bNum = l_idx - r_idx;
        Eigen::VectorXd tmp = mirrorMomentum.segment(3*r_idx, 3*bNum);
        mirrorMomentum.segment(3*r_idx, 3*bNum) = mirrorMomentum.segment(3*l_idx, 3*bNum);
        mirrorMomentum.segment(3*l_idx, 3*bNum) = tmp;

        return mirrorMomentum;
    }

    Eigen::VectorXd Character::GetMirrorPosition(Eigen::VectorXd pos)
    {
        for (auto p : mPairs) {
            const int dof = p.first->getNumDofs();
            if (dof == 0) continue;

            const int idx_first = p.second->getIndexInSkeleton(0);
            const int idx_second = p.first->getIndexInSkeleton(0);
            Eigen::VectorXd pos_first = pos.segment(idx_second, dof);
            Eigen::VectorXd pos_second = pos.segment(idx_first, dof);
            if (dof == 3) {
                pos_first[1] *= -1; pos_first[2] *= -1;
                pos_second[1] *= -1; pos_second[2] *= -1;
            }
            if (dof == 6) {
                pos_first[1] *= -1; pos_first[2] *= -1; pos_first[3] *= -1;
                pos_second[1] *= -1; pos_second[2] *= -1; pos_second[3] *= -1;
            }
            pos.segment(idx_first, dof) = pos_first;
            pos.segment(idx_second, dof) = pos_second;
        }
        return pos;
    }


    void Character::SetMirrorMotion()
    {
        vector<Eigen::VectorXd> pureMotions = mBVH->mPureMotions;
        vector<Eigen::VectorXd> mirrorMotions;
        int n = pureMotions.size();
        Eigen::Vector3d pos(0, 0, 0);
        for (int i = 0; i < n; i++)
        {
            int idx = (i + n / 2) % n;
            Eigen::VectorXd m = GetMirrorPosition(pureMotions[idx]);
            m.segment<3>(3) += pos;
            if (idx == (n - 1))
            {
                pos = m.segment<3>(3);
                pos[1] = 0;
            }
            mirrorMotions.push_back(m);
        }
        Eigen::Vector3d p0 = mirrorMotions[0].segment<3>(3);
        for (auto &m : mirrorMotions) m.segment<3>(3) -= Eigen::Vector3d(p0[0], 0, p0[2]);

        Eigen::Vector3d d(0, 0, 0);
        d[0] = (pureMotions.back()[3] - pureMotions[0][3]) + (mirrorMotions.back()[3] - mirrorMotions[0][3]);
        d[2] = -(pureMotions.back()[5] - pureMotions[0][5]) + (mirrorMotions.back()[5] - mirrorMotions[0][5]);

        for (int i = n / 2; i < n; i++) mirrorMotions[i].segment<3>(3) -= d;

        mBVH->setMirrorMotions(mirrorMotions);
        mBVH->blendMirrorMotion();
    }

    static tuple<Eigen::Vector3d, double, double, double> UnfoldModifyInfo(const MASS::ModifyInfo &info)
    {
        return make_tuple(Eigen::Vector3d(info[0], info[1], info[2]), info[3], info[4], info[5]);
    }

    static Eigen::Isometry3d modifyIsometry3d(const Eigen::Isometry3d &iso, const MASS::ModifyInfo &info, int axis, bool rotate = true)
    {
        Eigen::Vector3d l;
        double s, t, m;
        tie(l, s, t, m) = UnfoldModifyInfo(info);
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

    static void modifyShapeNode(BodyNode* rtgBody, BodyNode* stdBody, const MASS::ModifyInfo &info, int axis)
    {
        Eigen::Vector3d l;
        double s, t, m;
        tie(l, s, t, m) = UnfoldModifyInfo(info);
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
            if(auto rtg = dynamic_pointer_cast<CapsuleShape>(rtgShape->getShape()))
            {
                auto std = dynamic_pointer_cast<CapsuleShape>(stdShape->getShape());
                double radius = std->getRadius() * s * (lb+lc)/2, height = std->getHeight() * s * la;
                newShape = ShapePtr(new CapsuleShape(radius, height));
            }
            else if(auto rtg = dynamic_pointer_cast<SphereShape>(rtgShape->getShape()))
            {
                auto std = dynamic_pointer_cast<SphereShape>(stdShape->getShape());
                double radius = std->getRadius() * s * (la+lb+lc)/3;
                newShape = ShapePtr(new SphereShape(radius));
            }
            else if(auto rtg = dynamic_pointer_cast<CylinderShape>(rtgShape->getShape()))
            {
                auto std = dynamic_pointer_cast<CylinderShape>(stdShape->getShape());
                double radius = std->getRadius() * s * (lb+lc) / 2, height = std->getHeight() * s * la;
                newShape = ShapePtr(new CylinderShape(radius, height));
            }
            else if(dynamic_pointer_cast<BoxShape>(rtgShape->getShape()))
            {
                auto std = dynamic_pointer_cast<BoxShape>(stdShape->getShape());
                Eigen::Vector3d size = std->getSize() * s;
                size = size.cwiseProduct(l);
                newShape = ShapePtr(new BoxShape(size));
            }
            else if(auto rtg = dynamic_pointer_cast<MeshShape>(rtgShape->getShape()))
            {
                auto std = dynamic_pointer_cast<MeshShape>(stdShape->getShape());
                Eigen::Vector3d scale = std->getScale();
                scale *= s;
                scale = scale.cwiseProduct(l);
                rtg->setScale(scale);
                Eigen::Isometry3d tf = stdShape->getRelativeTransform();
                Eigen::Isometry3d r = modifyIsometry3d(tf.inverse(), info, axis).inverse();
                rtgShape->setRelativeTransform(r);
                newShape = rtg;
            }
            rtgShape->setShape(newShape);
        }
        if (m < 0.69) {
            cout << "[Warning] Mass is too small: " << m << " for body: " << rtgBody->getName() << " override to 0.7" << endl;
            m = 0.7;
        }
        double mass = stdBody->getMass() * m;
        dart::dynamics::Inertia inertia;
        inertia.setMass(mass);
        rtgBody->eachShapeNodeWith<dart::dynamics::DynamicsAspect>([&inertia, mass](ShapeNode* sn) {
            inertia.setMoment(sn->getShape()->computeInertia(mass));
            return false;
        });
        rtgBody->setInertia(inertia);
    }

    static map<string, int> skeletonAxis =
            {
                    {"Pelvis", 1},
                    {"FemurR", 1},
                    {"TibiaR", 1},
                    {"TalusR", 2},
                    {"FootThumbR", 2},
                    {"FootPinkyR", 2},
                    {"FemurL", 1},
                    {"TibiaL", 1},
                    {"TalusL", 2},
                    {"FootThumbL", 2},
                    {"FootPinkyL", 2},
                    {"Spine", 1},
                    {"Torso", 1},
                    {"Neck", 1},
                    {"Head", 1},
                    {"ShoulderR", 0},
                    {"ArmR", 0},
                    {"ForeArmR", 0},
                    {"HandR", 0},
                    {"ShoulderL", 0},
                    {"ArmL", 0},
                    {"ForeArmL", 0},
                    {"HandL", 0},
            };

    void MASS::Character::ModifySkeletonBodyNode(const vector<BoneInfo> &info, dart::dynamics::SkeletonPtr skel)
    {
        for(auto bone : info)
        {
            string name; ModifyInfo info;
            tie(name, info) = bone;
            int axis = skeletonAxis[name];
            BodyNode* rtgBody = skel->getBodyNode(name);
            BodyNode* stdBody = mStdSkeleton->getBodyNode(name);
            if(rtgBody == NULL) continue;
            if (mUseOBJ) modifyShapeNode(rtgBody, stdBody, info, axis);

            if(Joint* rtgParent = rtgBody->getParentJoint())
            {
                Joint* stdParent = stdBody->getParentJoint();
                Eigen::Isometry3d up = stdParent->getTransformFromChildBodyNode();
                rtgParent->setTransformFromChildBodyNode(modifyIsometry3d(up, info, axis));
            }

            for(int i=0; i<rtgBody->getNumChildJoints(); i++)
            {
                Joint* rtgJoint = rtgBody->getChildJoint(i);
                Joint* stdJoint = stdBody->getChildJoint(i);
                Eigen::Isometry3d down = stdJoint->getTransformFromParentBodyNode();
                rtgJoint->setTransformFromParentBodyNode(modifyIsometry3d(down, info, axis, false));
            }
        }
    }

    void MASS::Character::ModifySkeletonLengthAndMass(const vector<BoneInfo> &info, double mass_ratio)
    {
        for(auto bone : info)
        {
            string name;
            ModifyInfo info;
            tie(name, info) = bone;
            modifyLog[mSkeleton->getBodyNode(name)] = info;
        }
        ModifySkeletonBodyNode(info, mSkeleton);

        Eigen::VectorXd positions = mSkeleton->getPositions();
        Utils::setSkelPos(mSkeleton, Eigen::VectorXd::Zero(mSkeleton->getNumDofs()));
        Utils::setSkelPos(mStdSkeleton, Eigen::VectorXd::Zero(mSkeleton->getNumDofs()));

        double currentLegLength = mSkeleton->getBodyNode("Pelvis")->getCOM()[1] - mSkeleton->getBodyNode("TalusL")->getCOM()[1];
        double originalLegLength = mStdSkeleton->getBodyNode("Pelvis")->getCOM()[1] - mStdSkeleton->getBodyNode("TalusL")->getCOM()[1];
        motionScale = currentLegLength / originalLegLength;

        double prevOffset = yOffset;
        yOffset = -1e18;
        for(const auto& foot : {"TalusR", "TalusL"}) yOffset = max(yOffset, -mSkeleton->getBodyNode(foot)->getCOM()[1] + mTalusSize);

        mOffsetModify = yOffset - prevOffset;
        positions[4] += mOffsetModify;
        mOffsetModify = positions[4];
        footDifference = 0;
        for(const auto& foot : {"TalusR", "TalusL"}) footDifference = max(footDifference, yOffset - (-mSkeleton->getBodyNode(foot)->getCOM()[1] + mTalusSize));
        rootDefaultHeight = mSkeleton->getRootBodyNode()->getTransform().translation()[1] + yOffset;

        // custom muscle waypoint initial guessing
        for(int i=0; i<mMuscles.size(); i++)
        {
            Muscle* mMuscle = mMuscles[i];
            Muscle* mStdMuscle = mStdMuscles[i];
            mMuscle->set_mass_ratio(mass_ratio);
            for(int j=0; j<mMuscle->GetAnchors().size(); j++)
            {
                Anchor* mAnchor = mMuscle->GetAnchors()[j];
                Anchor* mStdAnchor = mStdMuscle->GetAnchors()[j];
                for(int k=0; k<mAnchor->bodynodes.size(); k++)
                {
                    const BodyNode* mBody = mAnchor->bodynodes[k];
                    int axis = skeletonAxis[mBody->getName()];
                    auto cur = Eigen::Isometry3d(Eigen::Translation3d(mStdAnchor->local_positions[k]));
                    Eigen::Isometry3d tmp = modifyIsometry3d(cur, modifyLog[mBody], axis);
                    mAnchor->local_positions[k] = tmp.translation();
                }
            }
            mMuscle->SetMuscle();
            double new_f0 = mStdMuscle->GetF0() * pow(mMuscle->GetMT0()/mStdMuscle->GetMT0(), 1.5);
            mMuscle->SetF0(new_f0);
            mMuscle->SetF0Original(new_f0);
        }

        for(int i=0; i<mMuscles.size(); i++)
        {
            Muscle* mMuscle = mMuscles[i];
            Muscle* mStdMuscle = mStdMuscles[i];
            mMuscle->SetMuscle();
            double new_f0 = mStdMuscle->GetF0() * pow(mMuscle->GetMT0()/mStdMuscle->GetMT0(), 1.5);
            mMuscle->SetF0(new_f0);
            mMuscle->SetF0Original(new_f0);
        }
        Utils::setSkelPos(mSkeleton, positions);
        Utils::setSkelPos(mStdSkeleton, Eigen::VectorXd::Zero(mSkeleton->getNumDofs()));

        SetCharacterInformation();
    }

    vector<BoneInfo> MASS::Character::LoadSkelParamFile(const string &filename)
    {
        vector<BoneInfo> infos;
        tinyxml2::XMLDocument doc;
        if(doc.LoadFile(filename.c_str()))
        {
            cout << "Can't open/parse file : " << filename << endl;
            throw invalid_argument("In func ModifyInfo");
        }

        tinyxml2::XMLElement *infoxml = doc.FirstChildElement("ModifyInfo");
        for(tinyxml2::XMLElement *boneXML = infoxml->FirstChildElement("Bone"); boneXML; boneXML = boneXML->NextSiblingElement("Bone"))
        {
            string body = string(boneXML->Attribute("body"));
            stringstream ss(boneXML->Attribute("info"));
            ModifyInfo info;
            for(int i = 0; i < 5; i++) ss >> info[i];
            infos.emplace_back(body, info);
        }
        return infos;
    }

    static map<string, int> readJointMap(const string &filename, dart::dynamics::SkeletonPtr skel)
    {
        FILE *in = fopen(filename.c_str(), "r");
        map<string, int> jointMap;
        char line[1005];
        while (fgets(line, 100, in) != NULL)
        {
            stringstream linestream(line);
            string name, bnName; int idx;
            linestream >> name >> bnName >> idx;

            dart::dynamics::BodyNode* bn = skel->getBodyNode(bnName);
            if(bn==NULL) continue;
            if(bn->getParentJoint()->getNumDofs() == 0) continue;

            int offset = bn->getParentJoint()->getDof(0)->getIndexInSkeleton();
            jointMap[name] = offset + idx;
        }

        return jointMap;
    }

    void Character::categorizeMuscles() {
        mMuscleGroupsInCharacter["muscle_length_Hyperlordosis"] = {"R_Psoas_Major", "L_Psoas_Major", "R_Psoas_Major1", "L_Psoas_Major1", "R_Psoas_Major2", "L_Psoas_Major2"};
        mMuscleGroupsInCharacter["muscle_length_Equinus"] = {"R_Soleus", "L_Soleus", "R_Soleus1", "L_Soleus1", "R_Tibialis_Posterior", "L_Tibialis_Posterior",
                                                             "R_Gastrocnemius_Lateral_Head", "L_Gastrocnemius_Lateral_Head"};
        mMuscleGroupsInCharacter["muscle_force_Waddling"] = {"R_Gluteus_Medius", "L_Gluteus_Medius", "R_Gluteus_Medius1", "L_Gluteus_Medius1", "R_Gluteus_Medius2", "L_Gluteus_Medius2", "R_Gluteus_Medius3", "L_Gluteus_Medius3",
                                                             "R_Gluteus_Minimus", "L_Gluteus_Minimus", "R_Gluteus_Minimus1", "L_Gluteus_Minimus1", "R_Gluteus_Minimus2", "L_Gluteus_Minimus2",
                                                             "R_Tensor_Fascia_Lata", "L_Tensor_Fascia_Lata", "R_Tensor_Fascia_Lata1", "L_Tensor_Fascia_Lata1", "R_Tensor_Fascia_Lata2", "L_Tensor_Fascia_Lata2"};
        mMuscleGroupsInCharacter["muscle_force_Calcaneal"] = {"R_Soleus", "L_Soleus", "R_Soleus1", "L_Soleus1", "R_Tibialis_Posterior", "L_Tibialis_Posterior",
                                                              "R_Gastrocnemius_Lateral_Head", "L_Gastrocnemius_Lateral_Head", "R_Gastrocnemius_Medial_Head", "L_Gastrocnemius_Medial_Head"};
        mMuscleGroupsInCharacter["muscle_force_Footdrop_R"] = {
                "R_Tibialis_Anterior", "R_Extensor_Hallucis_Longus", "R_Peroneus_Tertius", "R_Peroneus_Tertius1",
                "R_Extensor_Digitorum_Longus", "R_Extensor_Digitorum_Longus1", "R_Extensor_Digitorum_Longus2", "R_Extensor_Digitorum_Longus3"};
        mMuscleGroupsInCharacter["muscle_force_Footdrop_L"] = {
                "L_Tibialis_Anterior", "L_Extensor_Hallucis_Longus", "L_Peroneus_Tertius", "L_Peroneus_Tertius1",
                "L_Extensor_Digitorum_Longus", "L_Extensor_Digitorum_Longus1", "L_Extensor_Digitorum_Longus2", "L_Extensor_Digitorum_Longus3"};
        std::vector<std::string> muscle_force_Footdrop;
        muscle_force_Footdrop.reserve(mMuscleGroupsInCharacter["muscle_force_Footdrop_R"].size() + mMuscleGroupsInCharacter["muscle_force_Footdrop_L"].size());
        muscle_force_Footdrop.insert(muscle_force_Footdrop.end(), mMuscleGroupsInCharacter["muscle_force_Footdrop_R"].begin(), mMuscleGroupsInCharacter["muscle_force_Footdrop_R"].end());
        muscle_force_Footdrop.insert(muscle_force_Footdrop.end(), mMuscleGroupsInCharacter["muscle_force_Footdrop_L"].begin(), mMuscleGroupsInCharacter["muscle_force_Footdrop_L"].end());
        mMuscleGroupsInCharacter["muscle_force_Footdrop"] = muscle_force_Footdrop;

        // Precompute muscle-to-groups mapping for faster lookup
        std::unordered_map<std::string, std::vector<MuscleType>> muscleToTypes;
        
        for (const auto& [groupName, muscles] : muscleGroups) {
            auto it = StringToMuscleType.find(groupName);
            if (it == StringToMuscleType.end()) continue; // Skip if group has no MuscleBunchType            
            MuscleType type = it->second;
            for (const auto& muscleName : muscles) muscleToTypes[std::string(muscleName)].push_back(type);
        }

        // Assign types to each muscle
        for (auto* muscle : mMuscles) {
            string muscleName = muscle->GetName();
            for (const auto& [muscleNameKey, types] : muscleToTypes) {
                if (muscleName.find(muscleNameKey) != string::npos) {
                    for (MuscleType type : types) muscle->setMuscleBunchType(type);
                    break;
                }
            }
        }
    }
}