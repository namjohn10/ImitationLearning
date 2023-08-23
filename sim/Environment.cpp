#include "Environment.h"

Environment::
    Environment()
    : mUseRootXYZ(true), mPhaseUpdateInContolHz(false), mSimulationHz(600), mControlHz(30), mUseMuscle(false), mInferencePerSim(1), mHeightCalibration(0), mEnforceSymmetry(false), isRender(false), mIsStanceLearning(false), mLimitY(0.0), mLearningStd(false)
{
    mWorld = std::make_shared<dart::simulation::World>();
    mCyclic = true;
    mIsResidual = true;
    mSimulationConut = 0;
    mRewardMap.clear();
    mActionScale = 0.04;
    mIncludeMetabolicReward = false;
    mRewardType = deepmimic;
    mStanceOffset = 0.07;

    // GaitNet
    mRefStride = 1.34;
    mStride = 1.0;
    mCadence = 1.0;
    mPhaseDisplacementScale = -1.0;
    mPhaseDisplacement = 0.0;
    mNumActuatorAction = 0;

    mLoadedMuscleNN = false;
    mUseJointState = false;
    // Parameter
    mNumParamState = 0;
    mLearningStd = false;

    // Simulation Setting
    mSimulationStep = 0;
    mEOEType = EOEType::abstime;

    mSoftPhaseClipping = false;
    mHardPhaseClipping = false;
    mPhaseCount = 0;
    mWorldPhaseCount = 0;
    mGlobalTime = 0.0;
    mWorldTime = 0.0;

    mUseNormalizedParamState = true;
    // 0 : one foot , 1 : mid feet
    mPoseOptimizationMode = 0;
    mHorizon = 300;
}
Environment::
    ~Environment()
{
}

void Environment::
    initialize(std::string metadata)
{
    if (metadata.substr(metadata.length() - 4) == ".xml") // Path 를 입력했을 경우 변환 시켜줌.
    {
        std::ifstream file(metadata);
        if (!file.is_open())
            exit(-1);
        std::stringstream buffer;
        buffer << file.rdbuf();
        metadata = buffer.str();
    }

    mMetadata = metadata;

    TiXmlDocument doc;
    doc.Parse(mMetadata.c_str());

    // Skeleton Loading
    if (doc.FirstChildElement("skeleton") != NULL)
    {
        double defaultKp = std::stod(doc.FirstChildElement("skeleton")->Attribute("defaultKp"));
        double defaultKv = std::stod(doc.FirstChildElement("skeleton")->Attribute("defaultKv"));
        double defaultDamping = 0.4;
        if (doc.FirstChildElement("skeleton")->Attribute("damping") != NULL)
            defaultDamping = std::stod(doc.FirstChildElement("skeleton")->Attribute("damping"));

        addCharacter(Trim(std::string(doc.FirstChildElement("skeleton")->GetText())), defaultKp, defaultKv, defaultDamping);

        ActuactorType _actType;

        if (Trim(doc.FirstChildElement("skeleton")->Attribute("actuactor")) == "torque")
            _actType = tor;
        else if (Trim(doc.FirstChildElement("skeleton")->Attribute("actuactor")) == "pd")
            _actType = pd;
        else if (Trim(doc.FirstChildElement("skeleton")->Attribute("actuactor")) == "mass")
            _actType = mass;
        else if (Trim(doc.FirstChildElement("skeleton")->Attribute("actuactor")) == "muscle")
            _actType = mus;

        mCharacters.back()->setActuactorType(_actType);

        mTargetPositions = mCharacters.back()->getSkeleton()->getPositions();
        mTargetVelocities = mCharacters.back()->getSkeleton()->getVelocities();
    }

    // Muscle Loading
    if (doc.FirstChildElement("muscle") != NULL)
    {
        // Check LBS Weight Setting
        bool meshLbsWeight = false;
        bool useVelocityForce = false;

        if (doc.FirstChildElement("meshLbsWeight") != NULL)
            meshLbsWeight = doc.FirstChildElement("meshLbsWeight")->BoolText();

        if (doc.FirstChildElement("useVelocityForce") != NULL)
            useVelocityForce = doc.FirstChildElement("useVelocityForce")->BoolText();

        if (doc.FirstChildElement("useJointState") != NULL)
            mUseJointState = doc.FirstChildElement("useJointState")->BoolText();

        std::string muscle_path = Trim(std::string(doc.FirstChildElement("muscle")->GetText()));
        mCharacters[0]->setMuscles(muscle_path, useVelocityForce, meshLbsWeight);
        mUseMuscle = true;
    }

    // Learning Std
    if (doc.FirstChildElement("learningStd") != NULL)
        mLearningStd = doc.FirstChildElement("learningStd")->BoolText();

    // Phase Displacement Reward
    if (doc.FirstChildElement("timeWarping") != NULL)
        mPhaseDisplacementScale = doc.FirstChildElement("timeWarping")->DoubleText();

    // mAction Setting
    ActuactorType _actType = mCharacters.back()->getActuactorType();
    if (_actType == tor || _actType == pd || _actType == mass)
    {
        mAction = Eigen::VectorXd::Zero(mCharacters.back()->getSkeleton()->getNumDofs() - mCharacters.back()->getSkeleton()->getRootJoint()->getNumDofs() + (mPhaseDisplacementScale > 0 ? 1 : 0));
        mNumActuatorAction = mCharacters.back()->getSkeleton()->getNumDofs() - mCharacters.back()->getSkeleton()->getRootJoint()->getNumDofs();
    }
    else if (_actType == mus)
    {
        mAction = Eigen::VectorXd::Zero(mCharacters.back()->getMuscles().size() + (mPhaseDisplacementScale > 0 ? 1 : 0));
        mNumActuatorAction = mCharacters.back()->getMuscles().size();
    }
    // Ground Loading
    if (doc.FirstChildElement("ground") != NULL)
        addObject(Trim(std::string(doc.FirstChildElement("ground")->GetText())));

    // Cyclic Mode
    if (doc.FirstChildElement("cyclicbvh") != NULL)
        mCyclic = doc.FirstChildElement("cyclicbvh")->BoolText();

    // Controller Setting
    if (doc.FirstChildElement("residual") != NULL)
        mIsResidual = doc.FirstChildElement("residual")->BoolText();

    // Use XYZ
    if (doc.FirstChildElement("useRootXYZ") != NULL)
        mUseRootXYZ = doc.FirstChildElement("useRootXYZ")->BoolText();

    // Simulation Setting
    if (doc.FirstChildElement("simHz") != NULL)
        mSimulationHz = doc.FirstChildElement("simHz")->IntText();
    if (doc.FirstChildElement("controlHz") != NULL)
        mControlHz = doc.FirstChildElement("controlHz")->IntText();

    // Action Scale
    if (doc.FirstChildElement("actionScale") != NULL)
        mActionScale = doc.FirstChildElement("actionScale")->DoubleText();

    // Height
    if (doc.FirstChildElement("eoeheight") != NULL)
        mLimitY = doc.FirstChildElement("eoeheight")->DoubleText();

    // Stance Learning
    if (doc.FirstChildElement("stanceLearning") != NULL)
        mIsStanceLearning = doc.FirstChildElement("stanceLearning")->BoolText();

    // Inference Per Sim
    if (doc.FirstChildElement("inferencePerSim") != NULL)
        mInferencePerSim = doc.FirstChildElement("inferencePerSim")->IntText();

    // soft Phase Clipping
    if (doc.FirstChildElement("softPhaseClipping") != NULL)
        mSoftPhaseClipping = doc.FirstChildElement("softPhaseClipping")->BoolText();

    // hard Phase Clipping
    if (doc.FirstChildElement("hardPhaseClipping") != NULL)
        mHardPhaseClipping = doc.FirstChildElement("hardPhaseClipping")->BoolText();

    // Phase Update In Control Hz
    if (doc.FirstChildElement("phaseUpdateInControlHz") != NULL)
        mPhaseUpdateInContolHz = doc.FirstChildElement("phaseUpdateInControlHz")->BoolText();

    // Torque Clipping
    if (doc.FirstChildElement("torqueClipping") != NULL)
        mCharacters[0]->setTorqueClipping(doc.FirstChildElement("torqueClipping")->BoolText());

    // Include JtP in SPD
    if (doc.FirstChildElement("includeJtPinSPD") != NULL)
        mCharacters[0]->setIncludeJtPinSPD(doc.FirstChildElement("includeJtPinSPD")->BoolText());

    // Metabolic Reward
    if (doc.FirstChildElement("metabolicReward") != NULL)
        mIncludeMetabolicReward = doc.FirstChildElement("metabolicReward")->BoolText();

    if (doc.FirstChildElement("rewardType") != NULL)
    {
        std::string str_rewardType = doc.FirstChildElement("rewardType")->GetText();
        if (str_rewardType == "deepmimic")
            mRewardType = deepmimic;
        if (str_rewardType == "scadiver")
            mRewardType = scadiver;
    }

    if (doc.FirstChildElement("eoeType") != NULL)
    {
        std::string str_eoeType = doc.FirstChildElement("eoeType")->GetText();
        if (str_eoeType == "time")
            mEOEType = EOEType::abstime;
        else if (str_eoeType == "tuple")
            mEOEType = EOEType::tuple;
    }

    // Simulation World Wetting
    mWorld->setTimeStep(1.0 / mSimulationHz);
    // mWorld->getConstraintSolver()->setLCPSolver(dart::common::make_unique<dart::constraint::PGSLCPSolver>(mWorld->getTimeStep));
    // mWorld->setConstraintSolver(std::make_unique<dart::constraint::BoxedLcpConstraintSolver>(std::make_shared<dart::constraint::PgsBoxedLcpSolver>()));
    mWorld->getConstraintSolver()->setCollisionDetector(dart::collision::BulletCollisionDetector::create());
    mWorld->setGravity(Eigen::Vector3d(0, -9.8, 0.0));
    // Add Character
    for (auto &c : mCharacters)
        mWorld->addSkeleton(c->getSkeleton());
    // Add Objects
    for (auto o : mObjects)
        mWorld->addSkeleton(o);

    // BVH Loading
    // World Setting 후에 함. 왜냐하면 Height Calibration 을 위해서는 충돌 감지를 필요로 하기 때문.
    if (doc.FirstChildElement("bvh") != NULL)
    {
        BVH *new_bvh = new BVH(Trim(std::string(doc.FirstChildElement("bvh")->GetText())));
        new_bvh->setMode(std::string(doc.FirstChildElement("bvh")->Attribute("symmetry")) == "true");
        new_bvh->setHeightCalibration(std::string(doc.FirstChildElement("bvh")->Attribute("heightCalibration")) == "true");

        new_bvh->setRefMotion(mCharacters[0], mWorld);
        mBVHs.push_back(new_bvh);
    }

    // Advanced Option
    if (doc.FirstChildElement("heightCalibration") != NULL)
    {
        if (doc.FirstChildElement("heightCalibration")->BoolText())
        {
            mHeightCalibration++;
            if (std::string(doc.FirstChildElement("heightCalibration")->Attribute("strict")) == "true")
                mHeightCalibration++;
        }
    }

    if (doc.FirstChildElement("enforceSymmetry") != NULL)
        mEnforceSymmetry = doc.FirstChildElement("enforceSymmetry")->BoolText();

    if (isTwoLevelController())
    {
        Character *character = mCharacters[0];
        mMuscleNN = py::module::import("ray_model").attr("generating_muscle_nn")(character->getNumMuscleRelatedDof(), getNumActuatorAction(), character->getNumMuscles(), true);
    }

    if (doc.FirstChildElement("Horizon") != NULL)
        mHorizon = doc.FirstChildElement("Horizon")->IntText();

    // =================== Reward ======================
    // =================================================

    if (doc.FirstChildElement("useNormalizedParamState") != NULL)
        mUseNormalizedParamState = doc.FirstChildElement("useNormalizedParamState")->BoolText();

    if (doc.FirstChildElement("HeadLinearAccWeight") != NULL)
        mHeadLinearAccWeight = doc.FirstChildElement("HeadLinearAccWeight")->DoubleText();

    if (doc.FirstChildElement("HeadRotWeight") != NULL)
        mHeadRotWeight = doc.FirstChildElement("HeadRotWeight")->DoubleText();

    if (doc.FirstChildElement("StepWeight") != NULL)
        mStepWeight = doc.FirstChildElement("StepWeight")->DoubleText();

    if (doc.FirstChildElement("MetabolicWeight") != NULL)
        mMetabolicWeight = doc.FirstChildElement("MetabolicWeight")->DoubleText();

    if (doc.FirstChildElement("AvgVelWeight") != NULL)
        mAvgVelWeight = doc.FirstChildElement("AvgVelWeight")->DoubleText();

    // ============= For parameterization ==============
    // =================================================

    std::vector<double> minV;
    std::vector<double> maxV;
    std::vector<double> defaultV;
    if (doc.FirstChildElement("parameter") != NULL)
    {
        auto parameter = doc.FirstChildElement("parameter");
        for (TiXmlElement *group = parameter->FirstChildElement(); group != NULL; group = group->NextSiblingElement())
        {
            for (TiXmlElement *elem = group->FirstChildElement(); elem != NULL; elem = elem->NextSiblingElement())
            {
                minV.push_back(std::stod(elem->Attribute("min")));
                maxV.push_back(std::stod(elem->Attribute("max")));
                if (elem->Attribute("default") == NULL)
                    defaultV.push_back(1.0);
                else
                    defaultV.push_back(std::stod(elem->Attribute("default")));

                mParamName.push_back(std::string(group->Name()) + "_" + std::string(elem->Name()));

                if ((elem->Attribute("sampling") != NULL) && std::string(elem->Attribute("sampling")) == "uniform")
                    mSamplingStrategy.push_back(true);
                else
                    mSamplingStrategy.push_back(false);

                bool isExist = false;

                if (elem->Attribute("group") != NULL)
                {
                    std::string group_name = std::string(group->Name()) + "_" + elem->Attribute("group");
                    for (auto &p : mParamGroups)
                    {
                        if (p.name == group_name)
                        {
                            p.param_names.push_back(mParamName.back());
                            p.param_idxs.push_back(mParamName.size() - 1);
                            isExist = true;
                        }
                    }
                    if (!isExist)
                    {
                        param_group p;
                        p.name = group_name;
                        p.param_idxs.push_back(mParamName.size() - 1);
                        p.param_names.push_back(mParamName.back());
                        p.v = (defaultV.back() - minV.back()) / (maxV.back() - minV.back());
                        p.is_uniform = mSamplingStrategy.back();
                        mParamGroups.push_back(p);
                    }
                }
                else
                {
                    param_group p;
                    p.name = mParamName.back();
                    p.param_idxs.push_back(mParamName.size() - 1);
                    p.param_names.push_back(mParamName.back());
                    p.v = (defaultV.back() - minV.back()) / (maxV.back() - minV.back());
                    p.is_uniform = mSamplingStrategy.back();
                    mParamGroups.push_back(p);
                }
            }
        }
    }

    mParamMin = Eigen::VectorXd::Zero(minV.size());
    mParamMax = Eigen::VectorXd::Zero(minV.size());
    mParamDefault = Eigen::VectorXd::Zero(minV.size());

    for (int i = 0; i < minV.size(); i++)
    {
        mParamMin[i] = minV[i];
        mParamMax[i] = maxV[i];
        mParamDefault[i] = defaultV[i];
    }

    mNumParamState = minV.size();

    // set num known param which is the dof of gait parameters and skeleton parameters
    // find paramname which include "skeleton" or "stride" or "cadence"
    mNumKnownParam = 0;
    for (int i = 0; i < mParamName.size(); i++)
    {
        if (mParamName[i].find("skeleton") != std::string::npos || mParamName[i].find("stride") != std::string::npos || mParamName[i].find("cadence") != std::string::npos || mParamName[i].find("torsion") != std::string::npos)
            mNumKnownParam++;
    }
    // std::cout << "Num Known Param : " << mNumKnownParam << std::endl;
}

void Environment::
    addCharacter(std::string path, double kp, double kv, double damping)
{
    mCharacters.push_back(new Character(path, kp, kv, damping));
    // std::cout << "Skeleton Added " << mCharacters.back()->getSkeleton()->getName() << " Degree Of Freedom : " << mCharacters.back()->getSkeleton()->getNumDofs() << std::endl;
}

void Environment::
    addObject(std::string path)
{
    mObjects.push_back(BuildFromFile(path));
}

void Environment::
    setAction(Eigen::VectorXd _action)
{
    mPhaseDisplacement = 0.0;
    mAction.setZero();
    if (mAction.rows() != _action.rows())
    {
        std::cout << "[ERROR] Environment SetAction" << std::endl;
        exit(-1);
    }

    mAction = _action;
    mAction.head(mNumActuatorAction) *= mActionScale;

    if (mPhaseDisplacementScale > 0.0)
        mPhaseDisplacement += mPhaseDisplacementScale * mAction[mNumActuatorAction];
    else
        mPhaseDisplacement = 0.0;

    if (mPhaseDisplacement < (-1.0 / mControlHz))
        mPhaseDisplacement = -1.0 / mControlHz;

    Eigen::VectorXd actuactorAction = mAction.head(mNumActuatorAction);

    updateTargetPosAndVel();

    if (mCharacters[0]->getActuactorType() == pd || mCharacters[0]->getActuactorType() == mass)
    {
        Eigen::VectorXd action = Eigen::VectorXd::Zero(mCharacters[0]->getSkeleton()->getNumDofs());
        action.tail(actuactorAction.rows()) = actuactorAction;
        if (isMirror())
            action = mCharacters[0]->getMirrorPosition(action);

        if (mIsResidual)
            action = mCharacters[0]->addPositions(mTargetPositions, action);

        mCharacters[0]->setPDTarget(action);
    }
    else if (mCharacters[0]->getActuactorType() == tor)
    {
        Eigen::VectorXd torque = Eigen::VectorXd::Zero(mCharacters[0]->getSkeleton()->getNumDofs());
        torque.tail(actuactorAction.rows()) = actuactorAction;
        if (isMirror())
            torque = mCharacters[0]->getMirrorPosition(torque);
        mCharacters[0]->setTorque(torque);
    }
    else if (mCharacters[0]->getActuactorType() == mus)
    {
        Eigen::VectorXd activation = (!isMirror() ? actuactorAction : mCharacters[0]->getMirrorActivation(actuactorAction));
        // Clipping Function
        mCharacters[0]->setActivations(activation);
    }

    if (mPhaseUpdateInContolHz)
    {
        mGlobalTime += 1.0 / mControlHz;
        mWorldTime += 1.0 / mControlHz;
        mCharacters[0]->updateLocalTime(1.0 / mControlHz + mPhaseDisplacement);
    }

    mSimulationStep++;
}

void Environment::
    updateTargetPosAndVel(bool isInit)
{
    double dTime = 1.0 / mControlHz;
    double dPhase = dTime / (mBVHs[0]->getMaxTime() / (mCadence / sqrt(mCharacters[0]->getGlobalRatio())));

    if (mIsStanceLearning)
    {
        mTargetPositions.setZero();
        mTargetVelocities.setZero();
    }
    else
    {
        mTargetPositions = mBVHs[0]->getTargetPose(getLocalPhase() + (isInit ? 0.0 : dPhase));
        mTargetVelocities = mCharacters[0]->getSkeleton()->getPositionDifferences(mBVHs[0]->getTargetPose(getLocalPhase() + dPhase + (isInit ? 0.0 : dPhase)), mTargetPositions) / dTime;
        
        Eigen::VectorXd upperlimit = mCharacters[0]->getSkeleton()->getPositionUpperLimits();
        Eigen::VectorXd lowerlimit = mCharacters[0]->getSkeleton()->getPositionLowerLimits();

        for (int i = 6; i < mTargetPositions.rows(); i++)
        {
            if(mTargetPositions[i] > upperlimit[i])
            {
                mTargetPositions[i] = upperlimit[i];
                if(mTargetVelocities[i] > 0)
                    mTargetVelocities[i] = 0.0;
            }
            else if(mTargetPositions[i] < lowerlimit[i])
            {
                mTargetPositions[i] = lowerlimit[i];
                if(mTargetVelocities[i] < 0)
                    mTargetVelocities[i] = 0.0;
            }
        }
    }
}

int Environment::
    isEOE()
{
    int isEOE = 0;
    double root_y = mCharacters[0]->getSkeleton()->getCOM()[1];
    if (/*isFall() ||*/ root_y < mLimitY * mCharacters[0]->getGlobalRatio())
        isEOE = 1;
    // else if (mWorld->getTime() > 10.0)
    else if (((mEOEType == EOEType::tuple) && (mSimulationStep >= mHorizon)) || ((mEOEType == EOEType::abstime) && (mWorld->getTime() > 10.0)))
        isEOE = 3;
    return isEOE;
}

double Environment::
    getReward()
{
    double r = 0.0;
    if (mRewardType == deepmimic || mRewardType == scadiver)
    {
        // Deep Mimic Reward Setting
        double w_p = 0.65;
        double w_v = 0.1;
        double w_ee = 0.45;
        double w_com = 0.1;
        double w_metabolic = 0.2;

        auto skel = mCharacters[0]->getSkeleton();
        Eigen::VectorXd pos = skel->getPositions();
        Eigen::VectorXd vel = skel->getVelocities();

        Eigen::VectorXd pos_diff = skel->getPositionDifferences(mTargetPositions, pos);
        Eigen::VectorXd vel_diff = skel->getVelocityDifferences(mTargetVelocities, vel);

        auto ees = mCharacters[0]->getEndEffectors();
        Eigen::VectorXd ee_diff = Eigen::VectorXd::Zero(ees.size() * 3);
        Eigen::VectorXd ee_rot_diff = Eigen::VectorXd::Zero(ees.size());
        Eigen::Vector3d com_diff;

        // Rotation Diff
        std::vector<Eigen::Matrix3d> ee_rots;

        for (int i = 0; i < ees.size(); i++)
        {
            auto ee = ees[i];
            ee_diff.segment(i * 3, 3) = -ee->getCOM(skel->getRootBodyNode());
            ee_rots.push_back(ee->getTransform().linear());
        }

        com_diff = -skel->getCOM();
        skel->setPositions(mTargetPositions);
        for (int i = 0; i < ees.size(); i++)
        {
            auto ee = ees[i];
            ee_diff.segment(i * 3, 3) += ee->getCOM(skel->getRootBodyNode());
            ee_rots[i] = ee_rots[i] * ee->getTransform().linear().transpose();

            ee_rot_diff[i] = Eigen::AngleAxisd(ee_rots[i]).angle();
        }
        com_diff += skel->getCOM();
        skel->setPositions(pos);

        double r_p, r_v, r_ee, r_ee_rot, r_com, r_metabolic;
        double w_alive = 0.05;
        r_ee = exp(-40 * ee_diff.squaredNorm() / ee_diff.rows());
        r_ee_rot = exp(-4 * ee_rot_diff.squaredNorm() / ee_rot_diff.rows());
        r_p = exp(-20 * pos_diff.squaredNorm() / pos_diff.rows());
        r_v = 1.0;
        // exp(-10 * vel_diff.squaredNorm() / vel_diff.rows());
        r_com = exp(-10 * com_diff.squaredNorm() / com_diff.rows());
        r_metabolic = 0.0;

        if (mRewardType == deepmimic)
            r = w_p * r_p + w_v * r_v + w_com * r_com + w_ee * r_ee * r_ee_rot;
        else if (mRewardType == scadiver)
            r = (w_alive + (1.0 - w_alive) * r_p) * (w_alive + (1.0 - w_alive) * r_v) * (w_alive + (1.0 - w_alive) * r_com) * (w_alive + (1.0 - w_alive) * r_ee) * (w_alive + (1.0 - w_alive) * r_ee_rot);

        if (mIncludeMetabolicReward)
        {
            r_metabolic = getMetabolicReward();

            if (mRewardType == deepmimic)
                r += w_metabolic * r_metabolic;
            else if (mRewardType == scadiver)
                r *= (0.1 + 0.9 * r_metabolic);
        }

        if (isRender)
        {
            mRewardMap.clear();
            mRewardMap.insert(std::make_pair("r", r));
            mRewardMap.insert(std::make_pair("r_p", r_p));
            mRewardMap.insert(std::make_pair("r_v", r_v));
            mRewardMap.insert(std::make_pair("r_com", r_com));
            mRewardMap.insert(std::make_pair("r_ee", r_ee));
            mRewardMap.insert(std::make_pair("r_ee_rot", r_ee_rot));

            if (mIncludeMetabolicReward)
                mRewardMap.insert(std::make_pair("r_metabolic", r_metabolic));
        }
    }

    if (mCharacters[0]->getActuactorType() == mus)
    {
        // Design the reward function for musculo-skeletal system
        r = 1.0;
    }

    return r;
}

std::pair<Eigen::VectorXd, Eigen::VectorXd> Environment::
    getProjState(const Eigen::VectorXd minV, const Eigen::VectorXd maxV)
{
    if (minV.rows() != maxV.rows())
        exit(-1);

    Eigen::VectorXd curParamState = getParamState();
    Eigen::VectorXd projState = Eigen::VectorXd::Zero(mNumParamState);

    for (int i = 0; i < projState.rows(); i++)
        projState[i] = dart::math::clip(curParamState[i], minV[i], maxV[i]);

    std::vector<int> projectedParamIdx;
    for (int i = 0; i < minV.rows(); i++)
        if (abs(minV[i] - maxV[i]) > 1E-3)
            projectedParamIdx.push_back(i);

    Eigen::VectorXd p, v;
    auto skel = mCharacters[0]->getSkeleton();
    Eigen::Vector3d com = skel->getCOM();

    if (!mUseRootXYZ)
    {
        com[0] = 0.0;
        com[2] = 0.0;
    }

    int num_body_nodes = skel->getNumBodyNodes();

    p.resize(num_body_nodes * 3 + num_body_nodes * 6);
    v.resize((num_body_nodes + 1) * 3 + num_body_nodes * 3);

    p.setZero();
    v.setZero();

    if (!isMirror())
    {
        for (int i = 0; i < num_body_nodes; i++)
        {
            p.segment<3>(i * 3) = skel->getBodyNode(i)->getCOM() - skel->getCOM();
            Eigen::Isometry3d transform = skel->getBodyNode(i)->getTransform();
            p.segment<6>(num_body_nodes * 3 + 6 * i) << transform.linear()(0, 0), transform.linear()(0, 1), transform.linear()(0, 2),
                transform.linear()(1, 0), transform.linear()(1, 1), transform.linear()(1, 2);
            v.segment<3>(i * 3) = skel->getBodyNode(i)->getCOMLinearVelocity() - skel->getCOMLinearVelocity();
            v.segment<3>((num_body_nodes + 1) * 3 + i * 3) = 0.1 * skel->getBodyNode(i)->getAngularVelocity();
        }
        v.segment<3>(num_body_nodes * 3) = skel->getCOMLinearVelocity();
    }
    else
    {
        int idx = 0;
        std::vector<Eigen::Matrix3d> body_node_transforms = mCharacters[0]->getBodyNodeTransform();
        for (auto j_pair : mCharacters[0]->getPairs())
        {
            int first_idx = j_pair.first->getChildBodyNode()->getIndexInSkeleton();
            int second_idx = j_pair.second->getChildBodyNode()->getIndexInSkeleton();

            Eigen::Vector3d first_pos = j_pair.second->getChildBodyNode()->getCOM() - skel->getCOM();
            first_pos[0] *= -1;
            Eigen::Vector3d second_pos = j_pair.first->getChildBodyNode()->getCOM() - skel->getCOM();
            second_pos[0] *= -1;

            Eigen::AngleAxisd first_rot = Eigen::AngleAxisd(j_pair.second->getChildBodyNode()->getTransform().linear());
            first_rot.axis() = Eigen::Vector3d(first_rot.axis()[0], -first_rot.axis()[1], -first_rot.axis()[2]);

            Eigen::AngleAxisd second_rot = Eigen::AngleAxisd(j_pair.first->getChildBodyNode()->getTransform().linear());
            second_rot.axis() = Eigen::Vector3d(second_rot.axis()[0], -second_rot.axis()[1], -second_rot.axis()[2]);

            Eigen::Matrix3d first_rot_mat = first_rot.toRotationMatrix() * body_node_transforms[idx].transpose();
            Eigen::Matrix3d second_rot_mat = second_rot.toRotationMatrix() * body_node_transforms[idx];

            p.segment<3>(first_idx * 3) = first_pos;
            p.segment<3>(second_idx * 3) = second_pos;

            p.segment<6>(num_body_nodes * 3 + first_idx * 6) << first_rot_mat(0, 0), first_rot_mat(0, 1), first_rot_mat(0, 2), first_rot_mat(1, 0), first_rot_mat(1, 1), first_rot_mat(1, 2);
            p.segment<6>(num_body_nodes * 3 + second_idx * 6) << second_rot_mat(0, 0), second_rot_mat(0, 1), second_rot_mat(0, 2), second_rot_mat(1, 0), second_rot_mat(1, 1), second_rot_mat(1, 2);

            Eigen::Vector3d first_vel = j_pair.second->getChildBodyNode()->getCOMLinearVelocity() - skel->getCOMLinearVelocity();
            first_vel[0] *= -1;

            Eigen::Vector3d second_vel = j_pair.first->getChildBodyNode()->getCOMLinearVelocity() - skel->getCOMLinearVelocity();
            second_vel[0] *= -1;

            v.segment<3>(first_idx * 3) = first_vel;
            v.segment<3>(second_idx * 3) = second_vel;

            Eigen::Vector3d first_ang = 0.1 * j_pair.second->getChildBodyNode()->getAngularVelocity();
            first_ang[1] *= -1;
            first_ang[2] *= -1;
            v.segment<3>((num_body_nodes + 1) * 3 + first_idx * 3) = first_ang;

            Eigen::Vector3d second_ang = 0.1 * j_pair.first->getChildBodyNode()->getAngularVelocity();
            second_ang[1] *= -1;
            second_ang[2] *= -1;
            v.segment<3>((num_body_nodes + 1) * 3 + second_idx * 3) = second_ang;
            idx++;
        }
        v.segment<3>(num_body_nodes * 3) = skel->getCOMLinearVelocity();
        v.segment<3>(num_body_nodes * 3)[0] *= -1;
    }

    // Motion informcation (phase)

    Eigen::VectorXd phase = Eigen::VectorXd::Zero(1 + (mPhaseDisplacementScale > 0.0 ? 1 : 0));
    phase[0] = getNormalizedPhase();

    if (mPhaseDisplacementScale > 0.0)
        phase[1] = getLocalPhase(true);

    if (isMirror())
        for (int i = 0; i < phase.rows(); i++)
            phase[i] = (phase[i] + 0.5) - (int)(phase[i] + 0.5);

    // Gait Information (Step)
    Eigen::VectorXd step_state = Eigen::VectorXd::Zero(0);

    // Muscle State
    setParamState(projState, true);

    Eigen::VectorXd joint_state = Eigen::VectorXd::Zero(0);

    if (mUseJointState)
        joint_state = getJointState(isMirror());

    // Parameter State
    Eigen::VectorXd param_state = (mUseNormalizedParamState ? getNormalizedParamState(minV, maxV, isMirror()) : getParamState(isMirror()));
    Eigen::VectorXd proj_param_state = Eigen::VectorXd::Zero(projectedParamIdx.size());
    for (int i = 0; i < projectedParamIdx.size(); i++)
        proj_param_state[i] = param_state[projectedParamIdx[i]];

    setParamState(curParamState, true);

    // Ingration of all states

    Eigen::VectorXd state = Eigen::VectorXd::Zero(com.rows() + p.rows() + v.rows() + phase.rows() + step_state.rows() + joint_state.rows() + proj_param_state.rows());
    state << com, p, v, phase, step_state, 0.008 * joint_state, proj_param_state;

    return std::make_pair(state, joint_state);
}

Eigen::VectorXd Environment::
    getState()
{
    std::pair<Eigen::VectorXd, Eigen::VectorXd> res = getProjState(mParamMin, mParamMax);
    mState = res.first;
    mJointState = res.second;
    return mState;
}

void Environment::
    step(int _step)
{
    if (_step == 0)
        _step = mSimulationHz / mControlHz;
    else if ((mSimulationHz / mControlHz) % _step != 0)
        exit(-1);

    int rand_idx = dart::math::Random::uniform(0.0, _step - 1E-3);

    for (int i = 0; i < _step; i++)
    {
        if (mCharacters[0]->getActuactorType() == mass)
        {
            MuscleTuple mt = mCharacters[0]->getMuscleTuple(isMirror());

            Eigen::VectorXd fullJtp = Eigen::VectorXd::Zero(mCharacters[0]->getSkeleton()->getNumDofs());
            if (mCharacters[0]->getIncludeJtPinSPD())
                fullJtp.tail(fullJtp.rows() - mCharacters[0]->getSkeleton()->getRootJoint()->getNumDofs()) = mt.JtP;

            if (isMirror())
                fullJtp = mCharacters[0]->getMirrorPosition(fullJtp);

            Eigen::VectorXd fulldt = mCharacters[0]->getSPDForces(mCharacters[0]->getPDTarget(), fullJtp);

            mDesiredTorqueLogs.push_back(fulldt);

            if (isMirror())
                fulldt = mCharacters[0]->getMirrorPosition(fulldt);

            Eigen::VectorXd dt = fulldt.tail(mt.JtP.rows());

            if (!mCharacters[0]->getIncludeJtPinSPD())
                dt -= mt.JtP;

            Eigen::VectorXf activation = Eigen::VectorXf::Zero(mCharacters[0]->getMuscles().size());
            if (mLoadedMuscleNN)
                activation = mMuscleNN.attr("unnormalized_no_grad_forward")(mt.JtA_reduced, dt, py::cast<py::none>(Py_None), true, py::cast<py::none>(Py_None)).cast<Eigen::VectorXf>();

            activation = mMuscleNN.attr("forward_filter")(activation).cast<Eigen::VectorXf>();

            if (isMirror())
                activation = mCharacters[0]->getMirrorActivation(activation.cast<double>()).cast<float>();

            mCharacters[0]->setActivations(activation.cast<double>());

            if (i == rand_idx)
            {
                mRandomMuscleTuple = mt;
                mRandomDesiredTorque = dt;
            }
        }
        mCharacters[0]->step();
        mWorld->step();

        if (isRender)
            mContactLogs.push_back(getIsContact());

        if (!mPhaseUpdateInContolHz)
        {
            mGlobalTime += 1.0 / mSimulationHz;
            mWorldTime += 1.0 / mSimulationHz;
            mCharacters[0]->updateLocalTime((1.0 + mPhaseDisplacement * mControlHz) / mSimulationHz);
        }

        if (mHardPhaseClipping)
        {
            int currentGlobalCount = mGlobalTime / (mBVHs[0]->getMaxTime() / (mCadence / sqrt(mCharacters[0]->getGlobalRatio())));
            int currentLocalCount = mCharacters[0]->getLocalTime() / ((mBVHs[0]->getMaxTime() / (mCadence / sqrt(mCharacters[0]->getGlobalRatio()))));

            if (currentGlobalCount > currentLocalCount)
                mCharacters[0]->setLocalTime(mGlobalTime);
            else if (currentGlobalCount < currentLocalCount)
                mCharacters[0]->setLocalTime(currentLocalCount * ((mBVHs[0]->getMaxTime() / (mCadence / sqrt(mCharacters[0]->getGlobalRatio())))));
        }
        else if (mSoftPhaseClipping)
        {
            // FIXED LOCAL PHASE TIME
            int currentCount = mCharacters[0]->getLocalTime() / (0.5 * (mBVHs[0]->getMaxTime() / (mCadence / sqrt(mCharacters[0]->getGlobalRatio()))));
            // int currentCount = mCharacters[0]->getLocalTime() / ((mBVHs[0]->getMaxTime() / (mCadence / sqrt(mCharacters[0]->getGlobalRatio()))));
            if (mPhaseCount != currentCount)
            {
                mGlobalTime = mCharacters[0]->getLocalTime();
                mPhaseCount = currentCount;
            }
        }

        // World Time Clipping
        {
            int currentCount = mCharacters[0]->getLocalTime() / ((mBVHs[0]->getMaxTime() / (mCadence / sqrt(mCharacters[0]->getGlobalRatio()))));
            // int currentCount = mCharacters[0]->getLocalTime() / ((mBVHs[0]->getMaxTime() / (mCadence / sqrt(mCharacters[0]->getGlobalRatio()))));
            if (mWorldPhaseCount != currentCount)
            {
                mWorldTime = mCharacters[0]->getLocalTime();
                mWorldPhaseCount = currentCount;
            }
        }

        mSimulationConut++; // Should be called with mWorld Step
    }
}

void Environment::
    reset()
{
    mPhaseCount = 0;
    mWorldPhaseCount = 0;
    mSimulationConut = 0;

    // Reset Initial Time
    double time = 0.0;

    if (mRewardType == deepmimic || scadiver)
        time = dart::math::Random::uniform(0.1 * mBVHs[0]->getMaxTime(), 0.9 * mBVHs[0]->getMaxTime());

    if (mIsStanceLearning)
        time = 0.0;

    // Collision Detector Reset
    mWorld->getConstraintSolver()->setCollisionDetector(dart::collision::BulletCollisionDetector::create());
    mWorld->getConstraintSolver()->clearLastCollisionResult();

    // time = 0.0;

    mGlobalTime = time;
    mWorldTime = time;

    // time = 0.0;
    mWorld->setTime(time);

    // Reset Skeletons
    for (auto c : mCharacters)
    {
        c->getSkeleton()->setPositions(c->getSkeleton()->getPositions().setZero());
        c->getSkeleton()->setVelocities(c->getSkeleton()->getVelocities().setZero());

        c->getSkeleton()->clearConstraintImpulses();
        c->getSkeleton()->clearInternalForces();
        c->getSkeleton()->clearExternalForces();

        c->setLocalTime(time);
    }

    // Initial Pose Setting
    updateTargetPosAndVel(true);

    mCharacters[0]->getSkeleton()->setPositions(mTargetPositions);
    mCharacters[0]->getSkeleton()->setVelocities(mTargetVelocities);

    updateTargetPosAndVel();

    // Height / Pose Optimization
    if (mHeightCalibration != 0)
        mCharacters[0]->heightCalibration(mWorld, mHeightCalibration == 2);

    // Pose In ROM
    Eigen::VectorXd cur_pos = mCharacters[0]->getSkeleton()->getPositions();
    Eigen::VectorXd rom_min = mCharacters[0]->getSkeleton()->getPositionLowerLimits();
    Eigen::VectorXd rom_max = mCharacters[0]->getSkeleton()->getPositionUpperLimits();
    for (int i = 0; i < cur_pos.rows(); i++)
        cur_pos[i] = dart::math::clip(cur_pos[i], rom_min[i], rom_max[i]);
    mCharacters[0]->getSkeleton()->setPositions(cur_pos);

    mCharacters[0]->setPDTarget(mTargetPositions);
    mCharacters[0]->setTorque(mCharacters[0]->getTorque().setZero());
    if (mUseMuscle)
        mCharacters[0]->setActivations(mCharacters[0]->getActivations().setZero());

    // Initial Velocitiy Setting
    mCharacters[0]->clearLogs();

    mSimulationStep = 0;
    mContactLogs.clear();

    for (auto c : mCharacters)
    {
        c->getSkeleton()->clearInternalForces();
        c->getSkeleton()->clearExternalForces();
        c->getSkeleton()->clearConstraintImpulses();
    }
    mDesiredTorqueLogs.clear();
}

// Check whether the character falls or not
bool Environment::isFall()
{
    const auto results = mWorld->getConstraintSolver()->getLastCollisionResult();
    bool is_fall = false;
    for (int i = 0; i < results.getNumContacts(); i++)
    {

        const auto &c = results.getContact(i);

        if (c.collisionObject1->getShapeFrame()->getName().find("ground") != std::string::npos ||
            c.collisionObject2->getShapeFrame()->getName().find("ground") != std::string::npos)
        {
            if (c.collisionObject1->getShapeFrame()->getName().find("Foot") == std::string::npos &&
                c.collisionObject1->getShapeFrame()->getName().find("Talus") == std::string::npos &&

                c.collisionObject2->getShapeFrame()->getName().find("Foot") == std::string::npos &&
                c.collisionObject2->getShapeFrame()->getName().find("Talus") == std::string::npos

            )
                is_fall = true;
        }
    }

    return is_fall;
}

double
Environment::
    getMetabolicReward()
{
    double r_metabolic = 0.0;
    if (mUseMuscle)
    {
        Eigen::VectorXd activation_sum = Eigen::VectorXd::Zero(mCharacters[0]->getNumMuscles());
        const std::vector<Eigen::VectorXd> &muscleLogs = mCharacters[0]->getActivationLogs();
        int log_size = muscleLogs.size();

        if (log_size == 0)
            r_metabolic = 1.0;
        else
        {

            for (int i = 0; i < mSimulationHz / mControlHz; i++)
            {
                for (int j = 0; j < activation_sum.rows(); j++)
                    activation_sum[j] += abs(muscleLogs[log_size - 1 - i][j]);
            }
            activation_sum /= (mSimulationHz / mControlHz);
            r_metabolic = exp(-mMetabolicWeight * activation_sum.squaredNorm() / activation_sum.rows());
        }
    }
    else
    {
        Eigen::VectorXd torque_sum = Eigen::VectorXd::Zero(mCharacters[0]->getSkeleton()->getNumDofs());
        const std::vector<Eigen::VectorXd> &torqueLogs = mCharacters[0]->getTorqueLogs();
        int log_size = torqueLogs.size();
        if (log_size == 0)
            r_metabolic = 0.0;
        else
        {

            for (int i = 0; i < mSimulationHz / mControlHz; i++)
                torque_sum += torqueLogs[log_size - 1 - i].cwiseAbs();
            torque_sum /= (mSimulationHz / mControlHz);
            r_metabolic = exp(-1E-4 * mMetabolicWeight * torque_sum.squaredNorm() / torque_sum.rows());
        }
    }
    return r_metabolic;
}

double
Environment::
    getLocoPrinReward()
{
    int horizon = mSimulationHz / mControlHz;
    const std::vector<Eigen::Vector3d> &headVels = mCharacters[0]->getHeadVelLogs();
    if (headVels.size() == 0)
        return 1.0;

    Eigen::Vector3d headLinearAcc = headVels.back() - headVels[headVels.size() - horizon];

    double headRotDiff = Eigen::AngleAxisd(mCharacters[0]->getSkeleton()->getBodyNode("Head")->getTransform().linear()).angle();
    double r_head_linear_acc = exp(-mHeadLinearAccWeight * headLinearAcc.squaredNorm() / headLinearAcc.rows());
    double r_head_rot_diff = exp(-mHeadRotWeight * headRotDiff * headRotDiff);
    double r_loco = r_head_linear_acc * r_head_rot_diff;

    return r_loco;
}

double
Environment::
    getStepReward()
{
    Eigen::Vector3d foot_diff = mCurrentFoot - mCurrentTargetFoot;
    foot_diff[0] = 0; // Ignore X axis difference

    Eigen::Vector3d clipped_foot_diff = dart::math::clip(foot_diff, -0.075 * Eigen::Vector3d::Ones(), 0.075 * Eigen::Vector3d::Ones());
    foot_diff -= clipped_foot_diff;
    Eigen::Vector2i is_contact = getIsContact();
    if ((mIsLeftLegStance && is_contact[0] == 1) || (!mIsLeftLegStance && is_contact[1] == 1))
        foot_diff[1] = 0;
    foot_diff *= 8;
    double r = exp(-mStepWeight * foot_diff.squaredNorm() / foot_diff.rows());
    return r;
}

Eigen::Vector3d Environment::
    getAvgVelocity()
{
    Eigen::Vector3d avg_vel = Eigen::Vector3d::Zero();
    const std::vector<Eigen::Vector3d> &coms = mCharacters[0]->getCOMLogs();
    int horizon = (mBVHs[0]->getMaxTime() / (mCadence / sqrt(mCharacters[0]->getGlobalRatio()))) * mSimulationHz;
    if (coms.size() > horizon)
    {
        Eigen::Vector3d cur_com = coms.back();
        Eigen::Vector3d prev_com = coms[coms.size() - horizon];
        avg_vel = (cur_com - prev_com) / (mBVHs[0]->getMaxTime() / (mCadence / sqrt(mCharacters[0]->getGlobalRatio())));
    }
    else
        avg_vel[2] = getTargetCOMVelocity();

    return avg_vel;
}

double
Environment::
    getAvgVelReward()
{
    Eigen::Vector3d curAvgVel = getAvgVelocity();
    double targetCOMVel = getTargetCOMVelocity();

    Eigen::Vector3d vel_diff = curAvgVel - Eigen::Vector3d(0, 0, targetCOMVel);
    double vel_reward = exp(-mAvgVelWeight * vel_diff.squaredNorm());
    return vel_reward;
}

Eigen::VectorXd
Environment::
    getJointState(bool isMirror)
{
    Eigen::VectorXd joint_state = Eigen::VectorXd::Zero(3 * (mCharacters[0]->getSkeleton()->getNumDofs() - mCharacters[0]->getSkeleton()->getRootJoint()->getNumDofs()));
    Eigen::VectorXd min_tau = Eigen::VectorXd::Zero(mCharacters[0]->getSkeleton()->getNumDofs() - mCharacters[0]->getSkeleton()->getRootJoint()->getNumDofs());
    Eigen::VectorXd max_tau = Eigen::VectorXd::Zero(mCharacters[0]->getSkeleton()->getNumDofs() - mCharacters[0]->getSkeleton()->getRootJoint()->getNumDofs());

    auto mt = mCharacters[0]->getMuscleTuple(isMirror);

    for (int i = 0; i < mt.JtA.rows(); i++)
    {
        for (int j = 0; j < mt.JtA.cols(); j++)
        {
            if (mt.JtA(i, j) < 0)
                min_tau[i] += mt.JtA(i, j);
            else
                max_tau[i] += mt.JtA(i, j);
        }
    }
    joint_state << 0.5 * min_tau, 0.5 * max_tau, 1.0 * mt.JtP;
    return joint_state;
}

void Environment::
    updateFootStep(bool isInit)
{

    double phase = getLocalPhase(true);
    if (0.33 < phase && phase <= 0.83)
    {
        // Transition Timing
        if (!isInit)
            if (mIsLeftLegStance)
            {
                mCurrentTargetFoot = mNextTargetFoot;
                mNextTargetFoot = mCurrentFoot + Eigen::Vector3d::UnitZ() * mRefStride * mStride * mCharacters[0]->getGlobalRatio();
            }

        mIsLeftLegStance = false;
        mCurrentFoot = mCharacters[0]->getSkeleton()->getBodyNode("TalusR")->getCOM();

        if (isInit)
        {
            mCurrentTargetFoot = mCurrentFoot;
            mNextTargetFoot = mCurrentFoot + 0.5 * Eigen::Vector3d::UnitZ() * mRefStride * mStride * mCharacters[0]->getGlobalRatio();
        }
    }
    else
    {
        // Transition Timing
        if (!isInit)
            if (!mIsLeftLegStance)
            {
                mCurrentTargetFoot = mNextTargetFoot;
                mNextTargetFoot = mCurrentFoot + Eigen::Vector3d::UnitZ() * mRefStride * mStride * mCharacters[0]->getGlobalRatio();
            }

        mIsLeftLegStance = true;

        mCurrentFoot = mCharacters[0]->getSkeleton()->getBodyNode("TalusL")->getCOM();

        if (isInit)
        {
            mCurrentTargetFoot = mCurrentFoot;
            mNextTargetFoot = mCurrentFoot + 0.5 * Eigen::Vector3d::UnitZ() * mRefStride * mStride * mCharacters[0]->getGlobalRatio();
        }
    }
    mCurrentTargetFoot[1] = 0.0;
    mNextTargetFoot[1] = 0.0;
}

void Environment::
    setParamState(Eigen::VectorXd _param_state, bool onlyMuscle, bool doOptimization)
{
    int idx = 0;
    // skeleton parameter
    if (!onlyMuscle)
    {
        std::vector<std::pair<std::string, double>> skel_info;
        for (auto name : mParamName)
        {
            // gait parameter
            if (name.find("stride") != std::string::npos)
                mStride = _param_state[idx];

            if (name.find("cadence") != std::string::npos)
                mCadence = _param_state[idx];

            if (name.find("skeleton") != std::string::npos)
                skel_info.push_back(std::make_pair((name.substr(9)), _param_state[idx]));

            if (name.find("torsion") != std::string::npos)
                skel_info.push_back(std::make_pair(name, _param_state[idx]));

            idx++;
        }
        mCharacters[0]->setSkelParam(skel_info, doOptimization);
    }

    idx = 0;
    for (auto name : mParamName)
    {
        if (name.find("muscle_length") != std::string::npos)
            for (auto m : mCharacters[0]->getMuscles())
                if (name.substr(14) == m->GetName())
                {
                    m->change_l(_param_state[idx]);
                    break;
                }

        if (name.find("muscle_force") != std::string::npos)
            for (auto m : mCharacters[0]->getMuscles())
                if (name.substr(13) == m->GetName())
                {
                    m->change_f(_param_state[idx]);
                    break;
                }
        idx++;
    }
}

void Environment::
    setNormalizedParamState(Eigen::VectorXd _param_state, bool onlyMuscle, bool doOptimization)
{
    int idx = 0;
    // skeleton parameter
    if (!onlyMuscle)
    {
        std::vector<std::pair<std::string, double>> skel_info;
        for (auto name : mParamName)
        {
            // gait parameter

            if (name.find("stride") != std::string::npos)
                mStride = mParamMin[idx] + _param_state[idx] * (mParamMax[idx] - mParamMin[idx]);

            if (name.find("cadence") != std::string::npos)
                mCadence = mParamMin[idx] + _param_state[idx] * (mParamMax[idx] - mParamMin[idx]);

            if (name.find("skeleton") != std::string::npos)
                skel_info.push_back(std::make_pair((name.substr(9)), mParamMin[idx] + _param_state[idx] * (mParamMax[idx] - mParamMin[idx])));

            if (name.find("torsion") != std::string::npos)
                skel_info.push_back(std::make_pair(name, mParamMin[idx] + _param_state[idx] * (mParamMax[idx] - mParamMin[idx])));

            idx++;
        }
        mCharacters[0]->setSkelParam(skel_info, doOptimization);
    }

    idx = 0;
    for (auto name : mParamName)
    {
        if (name.find("muscle_length") != std::string::npos)
            for (auto m : mCharacters[0]->getMuscles())
                if (name.substr(14) == m->GetName())
                {
                    m->change_l(mParamMin[idx] + _param_state[idx] * (mParamMax[idx] - mParamMin[idx]));
                    break;
                }

        if (name.find("muscle_force") != std::string::npos)
            for (auto m : mCharacters[0]->getMuscles())
                if (name.substr(13) == m->GetName())
                {
                    m->change_f(mParamMin[idx] + _param_state[idx] * (mParamMax[idx] - mParamMin[idx]));
                    break;
                }
        idx++;
    }
}

Eigen::VectorXd
Environment::
    getParamState(bool isMirror)
{
    Eigen::VectorXd ParamState = Eigen::VectorXd::Zero(mNumParamState);
    int idx = 0;
    for (auto name : mParamName)
    {
        if (name.find("stride") != std::string::npos)
            ParamState[idx] = mStride;
        if (name.find("cadence") != std::string::npos)
            ParamState[idx] = mCadence;
        if (name.find("skeleton") != std::string::npos)
            ParamState[idx] = mCharacters[0]->getSkelParamValue(name.substr(9));

        if (name.find("torsion") != std::string::npos)
            ParamState[idx] = mCharacters[0]->getTorsionValue(name.substr(8));

        if (name.find("muscle_length") != std::string::npos)
            for (auto m : mCharacters[0]->getMuscles())
                if (name.substr(14) == m->GetName())
                {
                    ParamState[idx] = m->ratio_l();
                    break;
                }

        if (name.find("muscle_force") != std::string::npos)
            for (auto m : mCharacters[0]->getMuscles())
                if (name.substr(13) == m->GetName())
                {
                    ParamState[idx] = m->ratio_f();
                    break;
                }
        idx++;
    }

    if (isMirror)
    {
        int offset = 0;
        for (int i = 0; i < (int)mParamName.size() - 1; i++)
        {
            if (mParamName[i].find("skeleton") != std::string::npos)
                offset = 9;
            else if (mParamName[i].find("torsion") != std::string::npos)
                offset = 8;
            else if (mParamName[i].find("muscle_length") != std::string::npos)
                offset = 14;
            else if (mParamName[i].find("muscle_force") != std::string::npos)
                offset = 13;
            else
                continue;

            if ((mParamName[i].substr(1 + offset) == mParamName[i + 1].substr(1 + offset)) || (mParamName[i].substr(offset, mParamName[i].size() - 1 - offset) == mParamName[i + 1].substr(offset, mParamName[i + 1].size() - 1 - offset)))
            {
                double tmp = 0;
                tmp = ParamState[i];
                ParamState[i] = ParamState[i + 1];
                ParamState[i + 1] = tmp;
                i += 1;
                continue;
            }
        }
    }

    return ParamState;
}

Eigen::VectorXd
Environment::
    getParamSample()
{
    Eigen::VectorXd sampled_param = mParamMin;
    for (auto p : mParamGroups)
    {
        double w = 1;
        std::vector<double> locs;
        locs.push_back(0);
        locs.push_back(1);

        if (p.is_uniform)
        {
            w *= 0.25;
            for (int i = 1; i < 4; i++)
                locs.push_back(i * w);
            if (p.name.find("torsion") != std::string::npos)
                locs.push_back(0.5);
        }

        int sampled_c = (int)dart::math::Random::uniform(0.0, locs.size() - 0.01);
        double scale = locs[sampled_c]; // + dart::math::Random::normal(0.0, (mParamMin[p.param_idxs[0]] < 0.1? 0.1 : 0.5) * w);

        scale = dart::math::clip(scale, 0.0, 1.0);

        bool isAllSample = true; //(dart::math::Random::uniform(0, 1) < (1.0 / 10)?true:false);

        p.v = scale;

        double std_dev = dart::math::Random::normal(0.0, 0.025);
        for (auto idx : p.param_idxs)
        {
            double param_w = mParamMax[idx] - mParamMin[idx];
            if (isAllSample)
            {
                sampled_c = (int)dart::math::Random::uniform(0.0, locs.size() - 0.01);
                scale = locs[sampled_c];
                std_dev = dart::math::Random::normal(0.0, 0.025);
            }
            // std::cout << p.name << " param w " << param_w << " scale " << scale << "loc size " << locs.size() << " is uniform " << p.is_uniform << std::endl;
            sampled_param[idx] = mParamMin[idx] + param_w * scale + std_dev;
            sampled_param[idx] = dart::math::clip(sampled_param[idx], mParamMin[idx], mParamMax[idx]);
        }
    }

    return sampled_param;
}

Eigen::Vector2i Environment::getIsContact()
{
    Eigen::Vector2i result = Eigen::Vector2i(0, 0);
    const auto results = mWorld->getConstraintSolver()->getLastCollisionResult();
    for (auto bn : results.getCollidingBodyNodes())
    {
        if (bn->getName() == "TalusL" || ((bn->getName() == "FootPinkyL" || bn->getName() == "FootThumbL")))
            result[0] = 1;

        if (bn->getName() == "TalusR" || ((bn->getName() == "FootPinkyR" || bn->getName() == "FootThumbR")))
            result[1] = 1;
    }
    return result;
}

std::pair<Eigen::VectorXd, Eigen::VectorXd>
Environment::
    getSpace(std::string metadata)
{
    TiXmlDocument doc;
    Eigen::VectorXd minV = Eigen::VectorXd::Ones(mNumParamState);
    Eigen::VectorXd maxV = Eigen::VectorXd::Ones(mNumParamState);

    doc.Parse(metadata.c_str());
    if (doc.FirstChildElement("parameter") != NULL)
    {
        auto parameter = doc.FirstChildElement("parameter");
        for (TiXmlElement *group = parameter->FirstChildElement(); group != NULL; group = group->NextSiblingElement())
        {
            for (TiXmlElement *elem = group->FirstChildElement(); elem != NULL; elem = elem->NextSiblingElement())
            {
                std::string name = std::string(group->Name()) + "_" + std::string(elem->Name());
                for (int i = 0; i < mParamName.size(); i++)
                {
                    if (mParamName[i] == name)
                    {
                        minV[i] = std::stod(elem->Attribute("min"));
                        maxV[i] = std::stod(elem->Attribute("max"));
                    }
                }
            }
        }
    }

    return std::make_pair(minV, maxV);
}