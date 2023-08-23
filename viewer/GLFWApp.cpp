#include "GLFWApp.h"
#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#endif
#include "stb_image_write.h"
#include "dart/external/lodepng/lodepng.h"
const std::vector<std::string> CHANNELS =
    {
        "Xposition",
        "Yposition",
        "Zposition",
        "Xrotation",
        "Yrotation",
        "Zrotation",
};

GLFWApp::GLFWApp(int argc, char **argv, bool rendermode)
{
    mSimulation = false;
    mWidth = 1920;
    mHeight = 1080;
    mZoom = 1.0;
    mPersp = 45.0;
    mMouseDown = false;
    mRotate = false;
    mTranslate = false;
    mZooming = false;
    mTrans = Eigen::Vector3d(0.0, 0.0, 0.0);
    mEye = Eigen::Vector3d(0.0, 0.0, 1.0);
    mUp = Eigen::Vector3d(0.0, 1.0, 0.0);
    mDrawOBJ = false;

    // Screen Record
    mScreenRecord = false;
    mScreenIdx = 0;

    // Rendering Option
    mDrawReferenceSkeleton = true;
    mDrawCharacter = true;
    mDrawPDTarget = false;
    mDrawJointSphere = false;
    mStochasticPolicy = false;
    mDrawFootStep = false;
    mDrawEOE = false;

    mMuscleRenderType = activatonLevel;
    mMuscleRenderTypeInt = 2;
    mMuscleResolution = 0.0;

    // mCameraSetting
    mCameraMoving = 0;
    mFocus = 0;

    mTrackball.setTrackball(Eigen::Vector2d(mWidth * 0.5, mHeight * 0.5), mWidth * 0.5);
    mTrackball.setQuaternion(Eigen::Quaterniond::Identity());

    // GLFW Initialization
    glfwInit();
    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);

    mWindow = glfwCreateWindow(mWidth, mHeight, "render", nullptr, nullptr);
    if (mWindow == NULL)
    {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    glfwMakeContextCurrent(mWindow);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        exit(EXIT_FAILURE);
    }
    glViewport(0, 0, mWidth, mHeight);
    glfwSetWindowUserPointer(mWindow, this); // 창 사이즈 변경

    auto framebufferSizeCallback = [](GLFWwindow *window, int width, int height)
    {
        GLFWApp *app = static_cast<GLFWApp *>(glfwGetWindowUserPointer(window));
        app->mWidth = width;
        app->mHeight = height;
        glViewport(0, 0, width, height);
    };
    glfwSetFramebufferSizeCallback(mWindow, framebufferSizeCallback);

    auto keyCallback = [](GLFWwindow *window, int key, int scancode, int action, int mods)
    {
        auto &io = ImGui::GetIO();
        if (!io.WantCaptureKeyboard)
        {
            GLFWApp *app = static_cast<GLFWApp *>(glfwGetWindowUserPointer(window));
            app->keyboardPress(key, scancode, action, mods);
        }
    };
    glfwSetKeyCallback(mWindow, keyCallback);

    auto cursorPosCallback = [](GLFWwindow *window, double xpos, double ypos)
    {
        auto &io = ImGui::GetIO();
        if (!io.WantCaptureMouse)
        {
            GLFWApp *app = static_cast<GLFWApp *>(glfwGetWindowUserPointer(window));
            app->mouseMove(xpos, ypos);
        }
    };
    glfwSetCursorPosCallback(mWindow, cursorPosCallback);

    auto mouseButtonCallback = [](GLFWwindow *window, int button, int action, int mods)
    {
        auto &io = ImGui::GetIO();
        if (!io.WantCaptureMouse)
        {
            GLFWApp *app = static_cast<GLFWApp *>(glfwGetWindowUserPointer(window));
            app->mousePress(button, action, mods);
        }
    };
    glfwSetMouseButtonCallback(mWindow, mouseButtonCallback);

    auto scrollCallback = [](GLFWwindow *window, double xoffset, double yoffset)
    {
        auto &io = ImGui::GetIO();
        if (!io.WantCaptureMouse)
        {
            GLFWApp *app = static_cast<GLFWApp *>(glfwGetWindowUserPointer(window));
            app->mouseScroll(xoffset, yoffset);
        }
    };
    glfwSetScrollCallback(mWindow, scrollCallback);

    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(mWindow, true);
    ImGui_ImplOpenGL3_Init("#version 150");
    ImPlot::CreateContext();

    mns = py::module::import("__main__").attr("__dict__");
    py::module::import("sys").attr("path").attr("insert")(1, "../python");

    if (argc > 1) // Network 가 주어졌을 때
    {
        std::string path = std::string(argv[1]);

        mNetworkPaths.push_back(path);
    }

    mSelectedMuscles.clear();
    mRelatedDofs.clear();
}

GLFWApp::~GLFWApp()
{
    ImPlot::DestroyContext();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(mWindow);
    glfwTerminate();
}

void GLFWApp::writeBVH(const dart::dynamics::Joint *jn, std::ofstream &_f, const bool isPos)
{
    auto bn = jn->getParentBodyNode();

    if (!isPos) // HIERARCHY
    {

        _f << "JOINT\tCharacter_" << jn->getName() << std::endl;
        _f << "{" << std::endl;
        Eigen::Vector3d current_joint = jn->getParentBodyNode()->getTransform() * (jn->getTransformFromParentBodyNode() * Eigen::Vector3d::Zero());
        Eigen::Vector3d parent_joint = jn->getParentBodyNode()->getTransform() * ((jn->getParentBodyNode()->getParentJoint())->getTransformFromChildBodyNode() * Eigen::Vector3d::Zero());

        Eigen::Vector3d offset = current_joint - parent_joint; // jn->getTransformFromParentBodyNode() * ((bn->getParentJoint()->getTransformFromChildBodyNode()).inverse() * Eigen::Vector3d::Zero());
        offset *= 100.0;
        _f << "OFFSET\t" << offset.transpose() << std::endl;
        _f << "CHANNELS\t" << 3 << "\t" << CHANNELS[5] << "\t" << CHANNELS[3] << "\t" << CHANNELS[4] << std::endl;

        if (jn->getChildBodyNode()->getNumChildBodyNodes() == 0)
        {
            _f << "End Site" << std::endl;
            _f << "{" << std::endl;
            _f << "OFFSET\t" << (jn->getChildBodyNode()->getCOM() - current_joint).transpose() * 100.0 << std::endl;
            _f << "}" << std::endl;
        }

        else
            for (int idx = 0; idx < jn->getChildBodyNode()->getNumChildJoints(); idx++)
                writeBVH(jn->getChildBodyNode()->getChildJoint(idx), _f, false);

        _f << "}" << std::endl;
    }
    else
    {
        Eigen::Matrix3d r = Eigen::Matrix3d::Identity();
        Eigen::Vector3d pos = Eigen::Vector3d::Zero();
        int idx = jn->getJointIndexInSkeleton();

        if (jn->getNumDofs() == 1)
            pos = (mJointCalibration[idx].transpose() * Eigen::AngleAxisd(jn->getPositions()[0], ((RevoluteJoint *)(jn))->getAxis()).toRotationMatrix() * mJointCalibration[idx]).eulerAngles(2, 0, 1) * 180.0 / M_PI;
        else if (jn->getNumDofs() == 3)
            pos = (mJointCalibration[idx].transpose() * BallJoint::convertToRotation(jn->getPositions()) * mJointCalibration[idx]).eulerAngles(2, 0, 1) * 180.0 / M_PI;

        _f << pos.transpose() << " ";

        for (int idx = 0; idx < jn->getChildBodyNode()->getNumChildJoints(); idx++)
            writeBVH(jn->getChildBodyNode()->getChildJoint(idx), _f, true);
    }
}

void GLFWApp::exportBVH(const std::vector<Eigen::VectorXd> &motion, const dart::dynamics::SkeletonPtr &skel)
{

    std::ofstream bvh;
    bvh.open("motion1.bvh");
    // HIERARCHY WRITING
    Eigen::VectorXd pos_bkup = skel->getPositions();
    skel->setPositions(Eigen::VectorXd::Zero(pos_bkup.rows()));
    bvh << "HIERARCHY" << std::endl;
    dart::dynamics::Joint *jn = mEnv->getCharacter(0)->getSkeleton()->getRootJoint();
    dart::dynamics::BodyNode *bn = jn->getChildBodyNode();
    Eigen::Vector3d offset = bn->getTransform().translation();
    bvh << "ROOT\tCharacter_" << jn->getName() << std::endl;
    bvh << "{" << std::endl;
    bvh << "OFFSET\t" << offset.transpose() << std::endl;
    bvh << "CHANNELS\t" << 6 << "\t"
        << CHANNELS[0] << "\t" << CHANNELS[1] << "\t" << CHANNELS[2] << "\t"
        << CHANNELS[5] << "\t" << CHANNELS[3] << "\t" << CHANNELS[4] << std::endl;

    for (int idx = 0; idx < jn->getChildBodyNode()->getNumChildJoints(); idx++)
        writeBVH(jn->getChildBodyNode()->getChildJoint(idx), bvh, false);

    bvh << "}" << std::endl;
    bvh << "MOTION" << std::endl;
    bvh << "Frames:  " << mMotionBuffer.size() << std::endl;
    bvh << "Frame Time: " << 1.0 / 120 << std::endl;

    bvh.precision(4);
    for (Eigen::VectorXd p : mMotionBuffer)
    {
        skel->setPositions(p);
        Eigen::Vector6d root_pos;
        root_pos.head(3) = skel->getRootBodyNode()->getCOM() * 100.0;
        root_pos.tail(3) = skel->getRootBodyNode()->getTransform().linear().eulerAngles(2, 0, 1) * 180.0 / M_PI;

        // root_pos.setZero();

        bvh << root_pos.transpose() << " ";
        for (int idx = 0; idx < jn->getChildBodyNode()->getNumChildJoints(); idx++)
            writeBVH(jn->getChildBodyNode()->getChildJoint(idx), bvh, true);
        bvh << std::endl;
    }
    bvh << std::endl;
    bvh.close();
    // BVH Head Write
}

void GLFWApp::update(bool _isSave)
{
    if (mEnv->isActionTime())
    {
        // Reward Update
        mEnv->getReward();
        mRewardBuffer.push_back(mEnv->getRewardMap());

        Eigen::VectorXf action = (mNetworks.size() > 0 ? mNetworks[0].joint.attr("get_action")(mEnv->getState(), mStochasticPolicy).cast<Eigen::VectorXf>() : mEnv->getAction().cast<float>());

        mEnv->setAction(action.cast<double>());
    }
    if (_isSave)
    {
        mEnv->step(mEnv->getSimulationHz() / 120);
        mMotionBuffer.push_back(mEnv->getCharacter(0)->getSkeleton()->getPositions());
    }
    else
        mEnv->step(mEnv->getSimulationHz() / mEnv->getControlHz() / 2);
}

void GLFWApp::startLoop()
{
    while (!glfwWindowShouldClose(mWindow))
    {
        if (glfwGetKey(mWindow, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        {
            glfwSetWindowShouldClose(mWindow, true);
        }

        // Simulation Step
        if (mSimulation)
            update();

        // Rendering
        drawSimFrame();

        if (!mScreenRecord)
            drawUIFrame();
        else
        {
            int width, height;
            glfwGetFramebufferSize(mWindow, &width, &height);
            unsigned char *pixels = new unsigned char[width * height * 4];
            glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
            int stride = 4 * width;
            std::vector<unsigned char> flipped_pixels(stride * height);
            for (int y = 0; y < height; ++y)
                memcpy(&flipped_pixels[y * stride], &pixels[(height - y - 1) * stride], stride);
            lodepng::encode(("../screenshots/screenshot" + std::to_string(mScreenIdx) + ".png").c_str(), flipped_pixels.data(), width, height);
            std::cout << "Saving screenshot" << mScreenIdx << ".png ...... " << std::endl;
            delete[] pixels;
            mScreenIdx++;
        }

        glfwPollEvents();
        glfwSwapBuffers(mWindow);
    }
}

void GLFWApp::initGL()
{
    static float ambient[] = {0.2, 0.2, 0.2, 1.0};
    static float diffuse[] = {0.6, 0.6, 0.6, 1.0};
    static float front_mat_shininess[] = {60.0};
    static float front_mat_specular[] = {0.2, 0.2, 0.2, 1.0};
    static float front_mat_diffuse[] = {0.5, 0.28, 0.38, 1.0};
    static float lmodel_ambient[] = {0.2, 0.2, 0.2, 1.0};
    static float lmodel_twoside[] = {GL_FALSE};
    GLfloat position[] = {1.0, 0.0, 0.0, 0.0};
    GLfloat position1[] = {-1.0, 0.0, 0.0, 0.0};
    GLfloat position2[] = {0.0, 3.0, 0.0, 0.0};

    glClearColor(1.0, 1.0, 1.0, 1.0);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);
    glShadeModel(GL_SMOOTH);
    glPolygonMode(GL_FRONT, GL_FILL);

    glEnable(GL_LIGHT0);
    glLightfv(GL_LIGHT0, GL_AMBIENT, ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse);
    glLightfv(GL_LIGHT0, GL_POSITION, position);
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
    glLightModelfv(GL_LIGHT_MODEL_TWO_SIDE, lmodel_twoside);

    glEnable(GL_LIGHT1);
    glLightfv(GL_LIGHT1, GL_DIFFUSE, diffuse);
    glLightfv(GL_LIGHT1, GL_POSITION, position1);

    glEnable(GL_LIGHT2);
    glLightfv(GL_LIGHT2, GL_DIFFUSE, diffuse);
    glLightfv(GL_LIGHT2, GL_POSITION, position2);

    glEnable(GL_LIGHTING);

    glEnable(GL_COLOR_MATERIAL);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, front_mat_shininess);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, front_mat_specular);
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, front_mat_diffuse);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glEnable(GL_NORMALIZE);
    glEnable(GL_MULTISAMPLE);
}

void GLFWApp::setEnv(Environment *env, std::string metadata)
{
    loading_network = py::module::import("ray_model").attr("loading_network");

    if (mNetworkPaths.size() > 0)
    {
        std::string path = mNetworkPaths.back();
        if (path.substr(path.length() - 4) == ".xml")
        {
            metadata = path;
            mNetworkPaths.pop_back();
        }
        else
        {
            py::object py_metadata = py::module::import("ray_model").attr("loading_metadata")(mNetworkPaths.back());
            if (py_metadata.cast<py::none>() != Py_None)
                metadata = py_metadata.cast<std::string>();
        }
    }

    mEnv = env;
    mEnv->initialize(metadata);
    mEnv->setIsRender(true);

    mMotionSkeleton = mEnv->getCharacter(0)->getSkeleton()->cloneSkeleton();
    auto character = mEnv->getCharacter(0);
    mNetworks.clear();

    for (auto p : mNetworkPaths)
    {
        Network new_elem;
        new_elem.name = p;
        py::tuple res = loading_network(p.c_str(), mEnv->getState().rows(), mEnv->getAction().rows(), (character->getActuactorType() == mass), mEnv->getNumActuatorAction(), character->getNumMuscles(), character->getNumMuscleRelatedDof());
        new_elem.joint = res[0];
        new_elem.muscle = res[1];
        mNetworks.push_back(new_elem);
    }

    if (mNetworks.size() > 0)
        mEnv->setMuscleNetwork(mNetworks.back().muscle); // Set Main Muscle Network

    mRelatedDofs.clear();
    for (int i = 0; i < mEnv->getCharacter(0)->getSkeleton()->getNumDofs(); i++)
    {
        mRelatedDofs.push_back(false);
        mRelatedDofs.push_back(false);
    }

    // Set For BVH
    for (auto jn : mEnv->getCharacter(0)->getSkeleton()->getJoints())
    {
        if (jn == mEnv->getCharacter(0)->getSkeleton()->getRootJoint())
            mJointCalibration.push_back(Eigen::Matrix3d::Identity());
        else
            mJointCalibration.push_back((jn->getTransformFromParentBodyNode() * jn->getParentBodyNode()->getTransform()).linear().transpose());
    }
}

void GLFWApp::drawAxis()
{
    GUI::DrawLine(Eigen::Vector3d(0, 2E-3, 0), Eigen::Vector3d(0.5, 0.0, 0.0), Eigen::Vector3d(1.0, 0.0, 0.0));
    GUI::DrawLine(Eigen::Vector3d(0, 2E-3, 0), Eigen::Vector3d(0.0, 0.5, 0.0), Eigen::Vector3d(0.0, 1.0, 0.0));
    GUI::DrawLine(Eigen::Vector3d(0, 2E-3, 0), Eigen::Vector3d(0.0, 0.0, 0.5), Eigen::Vector3d(0.0, 0.0, 1.0));
}

void GLFWApp::
    drawSingleBodyNode(const BodyNode *bn, const Eigen::Vector4d &color)
{
    if (!bn)
        return;

    glPushMatrix();
    glMultMatrixd(bn->getTransform().data());

    auto sns = bn->getShapeNodesWith<VisualAspect>();
    for (const auto &sn : sns)
    {
        if (!sn)
            return;

        const auto &va = sn->getVisualAspect();

        if (!va || va->isHidden())
            return;

        glPushMatrix();
        Eigen::Affine3d tmp = sn->getRelativeTransform();
        glMultMatrixd(tmp.data());
        Eigen::Vector4d c = va->getRGBA();

        drawShape(sn->getShape().get(), color);

        glPopMatrix();
    }
    glPopMatrix();
}

void GLFWApp::drawUIDisplay()
{
    ImGui::SetNextWindowSize(ImVec2(400, 500), ImGuiCond_Once);
    ImGui::SetNextWindowPos(ImVec2(mWidth - 410, 10), ImGuiCond_Once);
    ImGui::Begin("Information");
    ImGui::Text("Elapsed     Time       :  %.3f s", mEnv->getWorld()->getTime());
    ImGui::Text("Current    Phase       :  %.3f  ", std::fmod(mEnv->getCharacter(0)->getLocalTime(), (mEnv->getBVH(0)->getMaxTime() / mEnv->getCadence())) / (mEnv->getBVH(0)->getMaxTime() / mEnv->getCadence()));
    ImGui::Text("Target     Velocity    :  %.3f m/s", mEnv->getTargetCOMVelocity());
    ImGui::Text("Average    Velocity    :  %.3f m/s", mEnv->getAvgVelocity()[2]);
    ImGui::Text("Current    Velocity    :  %.3f m/s", mEnv->getCharacter(0)->getSkeleton()->getCOMLinearVelocity()[2]);

    // Metadata
    if (ImGui::CollapsingHeader("Metadata"))
    {
        if (ImGui::Button("Print"))
            std::cout << mEnv->getMetadata() << std::endl;
        ImGui::Text(mEnv->getMetadata().c_str());
    }

    // Reward
    if (ImGui::CollapsingHeader("Reward"))
    {
        ImPlot::SetNextAxisLimits(0, -3, 0);
        ImPlot::SetNextAxisLimits(3, 0, 4);
        if (ImPlot::BeginPlot("Reward", "x", "r"))
        {
            if (mRewardBuffer.size() > 0)
            {
                double *x = new double[mRewardBuffer.size()]();
                int idx = 0;
                for (int i = mRewardBuffer.size() - 1; i > 0; i--)
                    x[idx++] = -i * (1.0 / mEnv->getControlHz());
                for (auto rs : mRewardBuffer[0])
                {

                    double *v = new double[mRewardBuffer.size()]();
                    for (int i = 0; i < mRewardBuffer.size(); i++)
                        v[i] = (mRewardBuffer[i].find(rs.first)->second);
                    ImPlot::PlotLine(rs.first.c_str(), x, v, mRewardBuffer.size());
                }
            }
            ImPlot::EndPlot();
        }
    }

    // State
    if (ImGui::CollapsingHeader("State"))
    {
        auto state = mEnv->getState();
        ImPlot::SetNextAxisLimits(0, -0.5, state.rows() + 0.5, ImGuiCond_Always);
        ImPlot::SetNextAxisLimits(3, -5, 5);

        // ImPlot::SetNextPlotLimitsY(-5,5);
        double *x = new double[state.rows()]();
        double *y = new double[state.rows()]();
        for (int i = 0; i < state.rows(); i++)
        {
            x[i] = i;
            y[i] = state[i];
        }
        if (ImPlot::BeginPlot("state"))
        {
            ImPlot::PlotBars("", x, y, state.rows(), 1.0);
            ImPlot::EndPlot();
        }
    }

    if (ImGui::CollapsingHeader("Constraint Force"))
    {
        Eigen::VectorXd cf = mEnv->getCharacter(0)->getSkeleton()->getConstraintForces();
        ImPlot::SetNextAxisLimits(0, -0.5, cf.rows() + 0.5, ImGuiCond_Always);
        ImPlot::SetNextAxisLimits(3, -5, 5);
        double *x = new double[cf.rows()]();
        double *y = cf.data();

        for (int i = 0; i < cf.rows(); i++)
            x[i] = i;

        if (ImPlot::BeginPlot("Constraint Force"))
        {
            ImPlot::PlotBars("dt", x, y, cf.rows(), 1.0);
            ImPlot::EndPlot();
        }
    }

    if (ImGui::CollapsingHeader("Torque"))
    {
        if (mEnv->getUseMuscle())
        {
            MuscleTuple tp = mEnv->getCharacter(0)->getMuscleTuple(false);

            Eigen::VectorXd fullJtp = Eigen::VectorXd::Zero(mEnv->getCharacter(0)->getSkeleton()->getNumDofs());
            if (mEnv->getCharacter(0)->getIncludeJtPinSPD())
                fullJtp.tail(fullJtp.rows() - mEnv->getCharacter(0)->getSkeleton()->getRootJoint()->getNumDofs()) = tp.JtP;
            Eigen::VectorXd dt = mEnv->getCharacter(0)->getSPDForces(mEnv->getCharacter(0)->getPDTarget(), fullJtp).tail(tp.JtP.rows());

            auto mtl = mEnv->getCharacter(0)->getMuscleTorqueLogs();

            Eigen::VectorXd min_tau = Eigen::VectorXd::Zero(tp.JtP.rows());
            Eigen::VectorXd max_tau = Eigen::VectorXd::Zero(tp.JtP.rows());

            for (int i = 0; i < tp.JtA.rows(); i++)
            {
                for (int j = 0; j < tp.JtA.cols(); j++)
                {
                    if (tp.JtA(i, j) > 0)
                        max_tau[i] += tp.JtA(i, j);
                    else
                        min_tau[i] += tp.JtA(i, j);
                }
            }

            // Drawing
            ImPlot::SetNextAxisLimits(0, -0.5, dt.rows() + 0.5, ImGuiCond_Always);
            ImPlot::SetNextAxisLimits(3, -5, 5);
            double *x = new double[dt.rows()]();
            double *y = dt.data();
            double *y_min = min_tau.data();
            double *y_max = max_tau.data();
            double *y_passive = tp.JtP.data();

            for (int i = 0; i < dt.rows(); i++)
                x[i] = i;

            if (ImPlot::BeginPlot("torque"))
            {
                ImPlot::PlotBars("min", x, y_min, dt.rows(), 1.0);
                ImPlot::PlotBars("max", x, y_max, dt.rows(), 1.0);
                ImPlot::PlotBars("dt", x, y, dt.rows(), 1.0);
                ImPlot::PlotBars("passive", x, y_passive, dt.rows(), 1.0);
                if (mtl.size() > 0)
                    ImPlot::PlotBars("exact", x, mtl.back().tail(mtl.back().rows() - 6).data(), dt.rows(), 1.0);

                ImPlot::EndPlot();
            }
        }
        else
        {
            Eigen::VectorXd dt = mEnv->getCharacter(0)->getTorque();
            ImPlot::SetNextAxisLimits(0, -0.5, dt.rows() + 0.5, ImGuiCond_Always);
            ImPlot::SetNextAxisLimits(3, -5, 5);
            double *x = new double[dt.rows()]();
            for (int i = 0; i < dt.rows(); i++)
                x[i] = i;
            if (ImPlot::BeginPlot("torque"))
            {
                ImPlot::PlotBars("dt", x, dt.data(), dt.rows(), 1.0);
                ImPlot::EndPlot();
            }
        }
    }

    if (ImGui::CollapsingHeader("Muscle Activation"))
    {
        Eigen::VectorXf activation = mEnv->getCharacter(0)->getActivations().cast<float>(); // * mEnv->getActionScale();
        int idx = 0;
        for (auto m : mEnv->getCharacter(0)->getMuscles())
        {
            ImGui::SliderFloat((m->GetName().c_str()), &activation[idx], 0.0, 1.0);
            idx++;
        }
        mEnv->getCharacter(0)->setActivations((activation.cast<double>()));
    }

    // Rendering Option
    if (ImGui::CollapsingHeader("Rendering Option"))
    {
        ImGui::SliderFloat("Muscle Resolution\t", &mMuscleResolution, 0.0, 1000.0);
        ImGui::Checkbox("Draw Reference Motion\t", &mDrawReferenceSkeleton);
        ImGui::Checkbox("Draw PD Target Motion\t", &mDrawPDTarget);
        ImGui::Checkbox("Draw Joint Sphere\t", &mDrawJointSphere);
        ImGui::Checkbox("Stochastic Policy\t", &mStochasticPolicy);
        ImGui::Checkbox("Draw Foot Step\t", &mDrawFootStep);
        ImGui::Checkbox("Draw EOE\t", &mDrawEOE);
    }
    if (ImGui::CollapsingHeader("Muscle Rendering Option"))
    {
        ImGui::RadioButton("PassiveForce", &mMuscleRenderTypeInt, 0);
        ImGui::RadioButton("ContractileForce", &mMuscleRenderTypeInt, 1);
        ImGui::RadioButton("ActivatonLevel", &mMuscleRenderTypeInt, 2);
        ImGui::RadioButton("Contracture", &mMuscleRenderTypeInt, 3);
        ImGui::RadioButton("Weakness", &mMuscleRenderTypeInt, 4);

        mMuscleRenderType = MuscleRenderingType(mMuscleRenderTypeInt);
    }
    if (mEnv->getUseMuscle())
        mEnv->getCharacter(0)->getMuscleTuple(false);
    // Related Dof Muscle Rendering
    mSelectedMuscles.clear();

    if (ImGui::CollapsingHeader("Related Dof Muscle"))
    {
        for (int i = 0; i < mRelatedDofs.size(); i += 2)
        {
            bool dof_plus, dof_minus;
            dof_plus = mRelatedDofs[i];
            dof_minus = mRelatedDofs[i + 1];
            ImGui::Checkbox((std::to_string(i / 2) + " +").c_str(), &dof_plus);
            ImGui::SameLine();
            ImGui::Checkbox((std::to_string(i / 2) + " -").c_str(), &dof_minus);
            mRelatedDofs[i] = dof_plus;
            mRelatedDofs[i + 1] = dof_minus;
        }

        // Check related dof
        for (auto m : mEnv->getCharacter(0)->getMuscles())
        {
            Eigen::VectorXd related_vec = m->GetRelatedVec();
            for (int i = 0; i < related_vec.rows(); i++)
            {
                if (related_vec[i] > 0 && mRelatedDofs[i * 2])
                {
                    mSelectedMuscles.push_back(m);
                    break;
                }
                else if (related_vec[i] < 0 && mRelatedDofs[i * 2 + 1])
                {
                    mSelectedMuscles.push_back(m);
                    break;
                }
            }
        }
    }

    if (ImGui::CollapsingHeader("Joint Position"))
    {
        Eigen::VectorXd pos_lower_limit = mEnv->getCharacter(0)->getSkeleton()->getPositionLowerLimits();
        Eigen::VectorXd pos_upper_limit = mEnv->getCharacter(0)->getSkeleton()->getPositionUpperLimits();
        Eigen::VectorXf pos = mEnv->getCharacter(0)->getSkeleton()->getPositions().cast<float>();

        for (int i = 0; i < pos.rows(); i++)
        {
            if (i < 6)
            {
                pos_lower_limit[i] = -10;
                pos_upper_limit[i] = 10;
            }
            ImGui::SliderFloat((std::to_string(i) + " Joint ").c_str(), &pos[i], pos_lower_limit[i], pos_upper_limit[i]);
        }
        mEnv->getCharacter(0)->getSkeleton()->setPositions(pos.cast<double>());
    }

    if (ImGui::CollapsingHeader("Joint Velocity"))
    {
        Eigen::VectorXd vel_lower_limit = mEnv->getCharacter(0)->getSkeleton()->getVelocities().setOnes() * -5;
        Eigen::VectorXd vel_upper_limit = mEnv->getCharacter(0)->getSkeleton()->getVelocities().setOnes() * 5;
        Eigen::VectorXf vel = mEnv->getCharacter(0)->getSkeleton()->getVelocities().cast<float>();

        for (int i = 0; i < vel.rows(); i++)
            ImGui::SliderFloat((std::to_string(i) + " Joint Velocity").c_str(), &vel[i], vel_lower_limit[i], vel_upper_limit[i]);

        mEnv->getCharacter(0)->getSkeleton()->setVelocities(vel.cast<double>());
    }

    // Parameters
    if (ImGui::CollapsingHeader("Parameters"))
    {
        Eigen::VectorXf ParamState = mEnv->getParamState().cast<float>();
        Eigen::VectorXf ParamMin = mEnv->getParamMin().cast<float>();
        Eigen::VectorXf ParamMax = mEnv->getParamMax().cast<float>();

        int idx = 0;
        for (auto c : mEnv->getParamName())
        {
            ImGui::SliderFloat(c.c_str(), &ParamState[idx], ParamMin[idx], ParamMax[idx] + 1E-10);
            idx++;
        }
        mEnv->setParamState(ParamState.cast<double>(), false, true);
        mEnv->getCharacter(0)->updateRefSkelParam(mMotionSkeleton);
    }

    // Parameters (Group)
    if (ImGui::CollapsingHeader("Parameters (Group)"))
    {
        Eigen::VectorXf group_v = Eigen::VectorXf::Ones(mEnv->getGroupParam().size());
        int idx = 0;

        for (auto p_g : mEnv->getGroupParam())
            group_v[idx++] = p_g.v;

        idx = 0;
        for (auto p_g : mEnv->getGroupParam())
        {
            ImGui::SliderFloat(p_g.name.c_str(), &group_v[idx], 0.0, 1.0);
            idx++;
        }
        mEnv->setGroupParam(group_v.cast<double>());
        mEnv->getCharacter(0)->updateRefSkelParam(mMotionSkeleton);
    }

    if (ImGui::CollapsingHeader("Gait Analysis"))
    {
        int horizon = (mEnv->getBVH(0)->getMaxTime() / (mEnv->getCadence() / sqrt(mEnv->getCharacter(0)->getGlobalRatio()))) * mEnv->getSimulationHz();

        ImPlot::SetNextAxisLimits(0, 0, horizon * 0.01, ImGuiCond_Always);
        ImPlot::SetNextAxisLimits(3, 0, 1.5);

        if (ImPlot::BeginPlot("Contact Graph"))
        {

            // std::vector<std::vector<double>> p; // = m->GetGraphData();
            std::vector<double> px_l;
            std::vector<double> px_r;

            std::vector<double> py_l;
            std::vector<double> py_r;

            px_l.clear();
            px_r.clear();

            py_l.clear();
            py_r.clear();
            std::vector<Eigen::Vector2i> c_logs = mEnv->getContactLogs();
            for (int i = 0; i < c_logs.size(); i++)
            {
                px_l.push_back(0.01 * i - c_logs.size() * 0.01 + horizon * 0.01);
                px_r.push_back(0.01 * i - (c_logs.size() + horizon * 0.5) * 0.01 + horizon * 0.01);

                py_l.push_back(c_logs[i][0]);
                py_r.push_back(c_logs[i][1]);
            }
            ImPlot::PlotLine("Left_Contact", px_l.data(), py_l.data(), px_l.size());
            ImPlot::PlotLine("Right_Contact", px_r.data(), py_r.data(), px_r.size());
            ImPlot::EndPlot();
        }
    }

    ImGui::End();
}

void GLFWApp::drawUIFrame()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    drawUIDisplay();
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void GLFWApp::drawPhase(double phase, double normalized_phase)
{
    glDisable(GL_LIGHTING);
    glDisable(GL_TEXTURE_2D);

    glPushMatrix();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glViewport(0, 0, mWidth, mHeight);
    gluOrtho2D(0.0, (GLdouble)mWidth, 0.0, (GLdouble)mHeight);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glLineWidth(1.0);
    glColor3f(0.0f, 0.0f, 0.0f);
    glTranslatef(mHeight * 0.05, mHeight * 0.05, 0.0f);
    glBegin(GL_LINE_LOOP);
    for (int i = 0; i < 360; i++)
    {
        double theta = i / 180.0 * M_PI;
        double x = mHeight * 0.04 * cos(theta);
        double y = mHeight * 0.04 * sin(theta);
        glVertex2d(x, y);
    }
    glEnd();

    glColor3f(1, 0, 0);
    glBegin(GL_LINES);
    glVertex2d(0, 0);
    glVertex2d(mHeight * 0.04 * sin(normalized_phase * 2 * M_PI), mHeight * 0.04 * cos(normalized_phase * M_PI * 2));
    glEnd();

    glColor3f(0, 0, 0);
    glLineWidth(2.0);

    glBegin(GL_LINES);
    glVertex2d(0, 0);
    glVertex2d(mHeight * 0.04 * sin(phase * 2 * M_PI), mHeight * 0.04 * cos(phase * M_PI * 2));
    glEnd();

    glPushMatrix();
    glPointSize(2.0);
    glBegin(GL_POINTS);
    glVertex2d(mHeight * 0.04 * sin(phase * 2 * M_PI), mHeight * 0.04 * cos(phase * M_PI * 2));
    glEnd();
    glPopMatrix();

    glPopMatrix();
}

void GLFWApp::drawSimFrame()
{
    initGL();
    setCamera();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    glViewport(0, 0, mWidth, mHeight);
    gluPerspective(mPersp, mWidth / mHeight, 0.1, 100.0);
    gluLookAt(mEye[0], mEye[1], mEye[2], 0.0, 0.0, -1.0, mUp[0], mUp[1], mUp[2]);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    mTrackball.setCenter(Eigen::Vector2d(mWidth * 0.5, mHeight * 0.5));
    mTrackball.setRadius(std::min(mWidth, mHeight) * 0.4);
    mTrackball.applyGLRotation();

    glScalef(mZoom, mZoom, mZoom);
    glTranslatef(mTrans[0] * 0.001, mTrans[1] * 0.001, mTrans[2] * 0.001);
    glEnable(GL_DEPTH_TEST);

    // Simulated Character
    if (mDrawCharacter)
    {
        drawSkeleton(mEnv->getCharacter(0)->getSkeleton()->getPositions(), Eigen::Vector4d(0.65, 0.65, 0.65, 1.0));
        drawShadow();

        if (mSelectedMuscles.size() > 0)
            drawMuscles(mSelectedMuscles, mMuscleRenderType);
        else
            drawMuscles(mEnv->getCharacter(0)->getMuscles(), mMuscleRenderType);
    }

    //  BVH
    if (mDrawReferenceSkeleton)
    {
        Eigen::VectorXd pos = (mDrawPDTarget ? mEnv->getCharacter(0)->getPDTarget() : mEnv->getTargetPositions());
        drawSkeleton(pos, Eigen::Vector4d(1.0, 0.35, 0.35, 1.0));
    }

    if (mMouseDown)
        drawAxis();

    if (mDrawJointSphere)
    {
        for (auto jn : mEnv->getCharacter(0)->getSkeleton()->getJoints())
        {
            Eigen::Vector3d jn_pos = jn->getChildBodyNode()->getTransform() * jn->getTransformFromChildBodyNode() * Eigen::Vector3d::Zero();
            glColor4f(0.0, 0.0, 0.0, 1.0);
            GUI::DrawSphere(jn_pos, 0.01);
            glColor4f(0.5, 0.5, 0.5, 0.2);
            GUI::DrawSphere(jn_pos, 0.1);
        }
    }

    if (mDrawEOE)
    {
        glColor4f(1.0, 0.0, 0.0, 1.0);
        GUI::DrawSphere(mEnv->getCharacter(0)->getSkeleton()->getCOM(), 0.01);
        glColor4f(0.5, 0.5, 0.8, 0.2);
        glBegin(GL_QUADS);
        glVertex3f(-10, mEnv->getLimitY() * mEnv->getCharacter(0)->getGlobalRatio(), -10);
        glVertex3f(10, mEnv->getLimitY() * mEnv->getCharacter(0)->getGlobalRatio(), -10);
        glVertex3f(10, mEnv->getLimitY() * mEnv->getCharacter(0)->getGlobalRatio(), 10);
        glVertex3f(-10, mEnv->getLimitY() * mEnv->getCharacter(0)->getGlobalRatio(), 10);
        glEnd();
    }


    drawGround(1E-3);

    if (!mScreenRecord)
        drawPhase(mEnv->getLocalPhase(true), mEnv->getNormalizedPhase());
}

void GLFWApp::drawGround(double height)
{
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glDisable(GL_LIGHTING);
    double width = 0.005;
    int count = 0;
    glBegin(GL_QUADS);
    for (double x = -100.0; x < 100.01; x += 1.0)
    {
        for (double z = -100.0; z < 100.01; z += 1.0)
        {
            if (count % 2 == 0)
                glColor3f(216.0 / 255.0, 211.0 / 255.0, 204.0 / 255.0);
            else
                glColor3f(216.0 / 255.0 - 0.1, 211.0 / 255.0 - 0.1, 204.0 / 255.0 - 0.1);
            count++;
            glVertex3f(x, height, z);
            glVertex3f(x + 1.0, height, z);
            glVertex3f(x + 1.0, height, z + 1.0);
            glVertex3f(x, height, z + 1.0);
        }
    }
    glEnd();
    glEnable(GL_LIGHTING);
}

void GLFWApp::mouseScroll(double xoffset, double yoffset)
{
    if (yoffset < 0)
        mEye *= 1.05;
    else if ((yoffset > 0) && (mEye.norm() > 0.5))
        mEye *= 0.95;
}

void GLFWApp::mouseMove(double xpos, double ypos)
{
    double deltaX = xpos - mMouseX;
    double deltaY = ypos - mMouseY;
    mMouseX = xpos;
    mMouseY = ypos;
    if (mRotate)
    {
        if (deltaX != 0 || deltaY != 0)
            mTrackball.updateBall(xpos, mHeight - ypos);
    }
    if (mTranslate)
    {
        Eigen::Matrix3d rot;
        rot = mTrackball.getRotationMatrix();
        mTrans += (1 / mZoom) * rot.transpose() * Eigen::Vector3d(deltaX, -deltaY, 0.0);
    }
    if (mZooming)
        mZoom = std::max(0.01, mZoom + deltaY * 0.01);
}

void GLFWApp::mousePress(int button, int action, int mods)
{
    if (action == GLFW_PRESS)
    {
        mMouseDown = true;
        if (button == GLFW_MOUSE_BUTTON_LEFT)
        {
            mRotate = true;
            mTrackball.startBall(mMouseX, mHeight - mMouseY);
        }
        else if (button == GLFW_MOUSE_BUTTON_RIGHT)
            mTranslate = true;
    }
    else if (action == GLFW_RELEASE)
    {
        mMouseDown = false;
        if (button == GLFW_MOUSE_BUTTON_LEFT)
        {
            mRotate = false;
        }
        else if (button == GLFW_MOUSE_BUTTON_RIGHT)
            mTranslate = false;
    }
}

void GLFWApp::reset()
{
    mEnv->reset();
    mRewardBuffer.clear();
}

void GLFWApp::keyboardPress(int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS)
    {
        switch (key)
        {
        case GLFW_KEY_U:
            mEnv->updateParamState();
            mEnv->getCharacter(0)->updateRefSkelParam(mMotionSkeleton);
            reset();
            break;
        case GLFW_KEY_COMMA:
            mEnv->setParamState(mEnv->getParamDefault(), false, true);
            mEnv->getCharacter(0)->updateRefSkelParam(mMotionSkeleton);
            reset();
            break;
        case GLFW_KEY_N:
            mEnv->setParamState(mEnv->getParamMin(), false, true);
            mEnv->getCharacter(0)->updateRefSkelParam(mMotionSkeleton);
            reset();
            break;
        case GLFW_KEY_M:
            mEnv->setParamState(mEnv->getParamMax(), false, true);
            mEnv->getCharacter(0)->updateRefSkelParam(mMotionSkeleton);
            reset();
            break;

        case GLFW_KEY_Z:
        {
            Eigen::VectorXd pos = mEnv->getCharacter(0)->getSkeleton()->getPositions().setZero();
            Eigen::VectorXd vel = mEnv->getCharacter(0)->getSkeleton()->getVelocities().setZero();
            pos[41] = 1.5;
            pos[51] = -1.5;
            mEnv->getCharacter(0)->getSkeleton()->setPositions(pos);
            mEnv->getCharacter(0)->getSkeleton()->setVelocities(vel);
        }
        break;
        // Rendering Key
        case GLFW_KEY_T:
            mDrawReferenceSkeleton = !mDrawReferenceSkeleton;
            break;
        case GLFW_KEY_P:
            mDrawCharacter = !mDrawCharacter;
            break;
        case GLFW_KEY_S:
            update();
            break;
        case GLFW_KEY_R:
            reset();
            break;
        case GLFW_KEY_O:
            mDrawOBJ = !mDrawOBJ;
            break;
        case GLFW_KEY_SPACE:
            mSimulation = !mSimulation;
            break;
        // Camera Setting
        case GLFW_KEY_F:
            mFocus += 1;
            mFocus %= 5;
            break;
        case GLFW_KEY_5:
        case GLFW_KEY_KP_5:
            mCameraMoving = 0;
            mTrackball.setQuaternion(Eigen::Quaterniond::Identity());
            break;

        case GLFW_KEY_7:
        case GLFW_KEY_KP_7:
            mCameraMoving -= 100;
            break;
        case GLFW_KEY_9:
        case GLFW_KEY_KP_9:
            mCameraMoving += 100;
            break;

        case GLFW_KEY_8:
        case GLFW_KEY_KP_8:
            mTrackball.setQuaternion(Eigen::Quaterniond(Eigen::AngleAxisd(0.01 * M_PI, Eigen::Vector3d::UnitX())) * mTrackball.getCurrQuat());
            break;
        case GLFW_KEY_2:
        case GLFW_KEY_KP_2:
            mTrackball.setQuaternion(Eigen::Quaterniond(Eigen::AngleAxisd(-0.01 * M_PI, Eigen::Vector3d::UnitX())) * mTrackball.getCurrQuat());
            break;
        case GLFW_KEY_1:
        case GLFW_KEY_KP_1:
            mEye = Eigen::Vector3d(0, 0, 2.92526);
            break;

        case GLFW_KEY_KP_3: // Muscle Information Rendering
            mFocus = 1;
            mCameraMoving = -400;
            mEye = Eigen::Vector3d(0, 0, 2.92526);

            glfwSetWindowSize(mWindow, 1920, mHeight);
            mCameraMoving = 0;

            {
                Eigen::VectorXd pos = mEnv->getCharacter(0)->getSkeleton()->getPositions().setZero();
                Eigen::VectorXd vel = mEnv->getCharacter(0)->getSkeleton()->getVelocities().setZero();
                pos[41] = 1.5;
                pos[51] = -1.5;
                mEnv->getCharacter(0)->getSkeleton()->setPositions(pos);
                mEnv->getCharacter(0)->getSkeleton()->setVelocities(vel);
            }
            break;

        case GLFW_KEY_0:
        case GLFW_KEY_KP_0:
            mScreenIdx = 0;
            mScreenRecord = !mScreenRecord;
            break;

        case GLFW_KEY_B:
            reset();

            mMotionBuffer.clear();
            while (mEnv->isEOE() == 0)
                update(true);

            exportBVH(mMotionBuffer, mEnv->getCharacter(0)->getSkeleton());
            break;
        default:
            break;
        }
    }
}
void GLFWApp::drawThinSkeleton(const dart::dynamics::SkeletonPtr skelptr)
{
    glColor3f(0.5, 0.5, 0.5);
    // Just Connect the joint position
    for (auto jn : skelptr->getJoints())
    {
        // jn position
        Eigen::Vector3d pos = Eigen::Vector3d::Zero();
        if (jn->getParentBodyNode() == nullptr)
        {
            pos = jn->getTransformFromParentBodyNode().translation();
            pos += jn->getPositions().tail(3);
        }
        else
            pos = jn->getParentBodyNode()->getTransform() * jn->getTransformFromParentBodyNode().translation();

        GUI::DrawSphere(pos, 0.015);

        int j = 0;
        while (true)
        {
            Eigen::Vector3d child_pos;
            if (jn->getChildBodyNode()->getNumChildJoints() > 0)
                child_pos = jn->getChildBodyNode()->getTransform() * jn->getChildBodyNode()->getChildJoint(j)->getTransformFromParentBodyNode().translation();
            else
            {
                child_pos = jn->getChildBodyNode()->getCOM();
                GUI::DrawSphere(child_pos, 0.015);
            }
            double length = (pos - child_pos).norm();

            glPushMatrix();

            // get Angle Axis vector which transform from (0,0,1) to (pos - child_pos) using atan2
            Eigen::Vector3d line = pos - child_pos;
            Eigen::Vector3d axis = Eigen::Vector3d::UnitZ().cross(line.normalized());

            double sin_angle = axis.norm();
            double cos_angle = Eigen::Vector3d::UnitZ().dot(line.normalized());
            double angle = atan2(sin_angle, cos_angle);

            glTranslatef((pos[0] + child_pos[0]) * 0.5, (pos[1] + child_pos[1]) * 0.5, (pos[2] + child_pos[2]) * 0.5);
            glRotatef(angle * 180 / M_PI, axis[0], axis[1], axis[2]);
            GUI::DrawCylinder(0.01, length);
            glPopMatrix();
            j++;

            if (jn->getChildBodyNode()->getNumChildJoints() == j || jn->getChildBodyNode()->getNumChildJoints() == 0)
                break;
        }
    }
}

void GLFWApp::drawSkeleton(const Eigen::VectorXd &pos, const Eigen::Vector4d &color, bool isLineSkeleton)
{
    mMotionSkeleton->setPositions(pos);

    if (!isLineSkeleton)
        for (const auto bn : mMotionSkeleton->getBodyNodes())
            drawSingleBodyNode(bn, color);
}

void GLFWApp::drawShape(const Shape *shape, const Eigen::Vector4d &color)
{
    if (!shape)
        return;

    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_DEPTH_TEST);
    glColor4d(color[0], color[1], color[2], color[3]);
    if (!mDrawOBJ)
    {
        if (shape->is<SphereShape>())
        {
            const auto *sphere = dynamic_cast<const SphereShape *>(shape);
            GUI::DrawSphere(sphere->getRadius());
        }
        else if (shape->is<BoxShape>())
        {
            const auto *box = dynamic_cast<const BoxShape *>(shape);
            GUI::DrawCube(box->getSize());
        }
        else if (shape->is<CapsuleShape>())
        {
            const auto *capsule = dynamic_cast<const CapsuleShape *>(shape);
            GUI::DrawCapsule(capsule->getRadius(), capsule->getHeight());
        }
        else if (shape->is<CylinderShape>())
        {
            const auto *cylinder = dynamic_cast<const CylinderShape *>(shape);
            GUI::DrawCylinder(cylinder->getRadius(), cylinder->getHeight());
        }
    }
    else
    {
        if (shape->is<MeshShape>())
        {
            const auto &mesh = dynamic_cast<const MeshShape *>(shape);
            mShapeRenderer.renderMesh(mesh, false, 0.0, color);
        }
    }
}

void GLFWApp::
    setCamera()
{
    if (mFocus == 1)
    {
        mTrans = -mEnv->getCharacter(0)->getSkeleton()->getCOM();
        mTrans[1] = -1;
        mTrans *= 1000;
    }
    else if (mFocus == 2)
    {
        mTrans = -mEnv->getTargetPositions().segment(3, 3); //-mEnv->getCharacter(0)->getSkeleton()->getCOM();
        mTrans[1] = -1;
        mTrans *= 1000;
    }

    if (mCameraMoving < 0) // Negative
    {
        mCameraMoving++;
        Eigen::Quaterniond r = Eigen::Quaterniond(Eigen::AngleAxisd(0.005 * M_PI, Eigen::Vector3d::UnitY())) * mTrackball.getCurrQuat();
        mTrackball.setQuaternion(r);
    }
    if (mCameraMoving > 0) // Positive
    {
        mCameraMoving--;
        Eigen::Quaterniond r = Eigen::Quaterniond(Eigen::AngleAxisd(-0.005 * M_PI, Eigen::Vector3d::UnitY())) * mTrackball.getCurrQuat();
        mTrackball.setQuaternion(r);
    }
}

void GLFWApp::drawCollision()
{
    const auto result = mEnv->getWorld()->getConstraintSolver()->getLastCollisionResult();
    for (const auto &contact : result.getContacts())
    {
        Eigen::Vector3d v = contact.point;
        Eigen::Vector3d f = contact.force / 1000.0;
        glLineWidth(2.0);
        glColor3f(0.8, 0.8, 0.2);
        glBegin(GL_LINES);
        glVertex3f(v[0], v[1], v[2]);
        glVertex3f(v[0] + f[0], v[1] + f[1], v[2] + f[2]);
        glEnd();
        glColor3f(0.8, 0.8, 0.2);
        glPushMatrix();
        glTranslated(v[0], v[1], v[2]);
        GUI::DrawSphere(0.01);
        glPopMatrix();
    }
}

void GLFWApp::drawMuscles(const std::vector<Muscle *> muscles, MuscleRenderingType renderingType, bool isTransparency)
{
    int count = 0;
    glEnable(GL_LIGHTING);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_DEPTH_TEST);

    for (auto muscle : muscles)
    {
        muscle->Update();
        double a = muscle->activation;
        Eigen::Vector4d color;
        switch (renderingType)
        {
        case activatonLevel:
            color = Eigen::Vector4d(0.2 + 1.6 * a, 0.2, 0.2, 0.1 + 0.9 * a);
            break;
        case passiveForce:
        {
            double f_p = muscle->Getf_p() / mMuscleResolution;
            color = Eigen::Vector4d(0.1, 0.1, 0.1 + 0.9 * f_p, f_p);
            break;
        }
        case contractileForce:
        {
            double f_c = muscle->Getf_A() * a / mMuscleResolution;
            color = Eigen::Vector4d(0.1, 0.1 + 0.9 * f_c, 0.1, f_c);
            break;
        }
        case weakness:
        {
            color = Eigen::Vector4d(0.1, 0.1 + 2.0 * (1.0 - muscle->f0 / muscle->f0_original), 0.1 + 2.0 * (1.0 - muscle->f0 / muscle->f0_original), 0.1 + 2.0 * (1.0 - muscle->f0 / muscle->f0_original));
            break;
        }
        case contracture:
        {
            color = Eigen::Vector4d(0.05 + 10.0 * (1.0 - muscle->l_mt0 / muscle->l_mt0_original), 0.05, 0.05 + 10.0 * (1.0 - muscle->l_mt0 / muscle->l_mt0_original), 0.05 + 5.0 * (1.0 - muscle->l_mt0 / muscle->l_mt0_original));
            break;
        }
        default:
            color.setOnes();
            break;
        }
        if (!isTransparency)
            color[3] = 0.8;

        glColor4dv(color.data());
        if (color[3] > 0.001)
            mShapeRenderer.renderMuscle(muscle, -1.0);
    }
    glEnable(GL_LIGHTING);
}

void GLFWApp::drawShadow()
{
    Eigen::VectorXd pos = mEnv->getCharacter(0)->getSkeleton()->getPositions();

    glDisable(GL_LIGHTING);
    glDisable(GL_COLOR_MATERIAL);
    glPushMatrix();
    glTranslatef(pos[3], 2E-3, pos[5]);
    glScalef(1.0, 1E-4, 1.0);
    glRotatef(30.0, 1.0, 0.0, 1.0);
    glTranslatef(-pos[3], 0.0, -pos[5]);
    drawSkeleton(pos, Eigen::Vector4d(0.1, 0.1, 0.1, 1.0));
    glPopMatrix();
    glEnable(GL_LIGHTING);
}