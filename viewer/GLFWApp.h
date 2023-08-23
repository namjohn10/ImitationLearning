
// #include <pybind11/numpy.h>
#include "dart/gui/Trackball.hpp"
#include "Environment.h"
#include "GLfunctions.h"
#include "ShapeRenderer.h"
#include <glad/glad.h>
#include <GL/glu.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <implot.h>
#include <examples/imgui_impl_glfw.h>
#include <examples/imgui_impl_opengl3.h>
#include <imgui_internal.h>
#include "C3D_Reader.h"

enum MuscleRenderingType
{
    passiveForce = 0,
    contractileForce,
    activatonLevel,
    contracture,
    weakness
};


class GLFWApp
{
public:
    GLFWApp(int argc, char **argv, bool rendermode = true);
    ~GLFWApp();

    void setEnv(Environment *env, std::string metadata = "../data/env.xml");

    void startLoop();
    void initGL();

    void writeBVH(const dart::dynamics::Joint *jn, std::ofstream &_f, const bool isPos = false); // Pose Or Hierarchy
    void exportBVH(const std::vector<Eigen::VectorXd> &motion, const dart::dynamics::SkeletonPtr &skel);

private:
    py::object mns;
    py::object loading_network;

    void update(bool isSave = false);
    void reset();

    // Drawing Component
    void setCamera();

    void drawSimFrame();
    void drawUIFrame();
    void drawUIDisplay();

    void drawGround(double height);
    void drawCollision();

    void drawSkeleton(const Eigen::VectorXd &pos, const Eigen::Vector4d &color, bool isLineSkeleton = false);
    
    void drawThinSkeleton(const dart::dynamics::SkeletonPtr skelptr);

    void drawSingleBodyNode(const BodyNode *bn, const Eigen::Vector4d &color);
    void drawFootStep();
    void drawPhase(double phase, double normalized_phase);

    void drawShape(const dart::dynamics::Shape *shape, const Eigen::Vector4d &color);

    void drawAxis();
    void drawMuscles(const std::vector<Muscle *> muscles, MuscleRenderingType renderingType = activatonLevel, bool isTransparency = true);

    void drawShadow();

    // Mousing Function
    void mouseMove(double xpos, double ypos);
    void mousePress(int button, int action, int mods);
    void mouseScroll(double xoffset, double yoffset);

    // Keyboard Function
    void keyboardPress(int key, int scancode, int action, int mods);

    // Variable
    bool mRenderMode;
    double mWidth, mHeight;
    bool mRotate, mTranslate, mZooming, mMouseDown;

    GLFWwindow *mWindow;
    Environment *mEnv;

    ShapeRenderer mShapeRenderer;
    bool mDrawOBJ;
    bool mSimulation;

    // Trackball/Camera variables
    dart::gui::Trackball mTrackball;
    double mZoom, mPersp, mMouseX, mMouseY;
    Eigen::Vector3d mTrans, mEye, mUp;
    int mCameraMoving, mFocus;

    // Skeleton for kinematic drawing
    dart::dynamics::SkeletonPtr mMotionSkeleton;
    std::vector<std::string> mNetworkPaths;
    std::vector<Network> mNetworks;

    // Reward Map
    std::vector<std::map<std::string, double>> mRewardBuffer;

    // Rendering Option
    bool mDrawReferenceSkeleton;
    bool mDrawCharacter;
    bool mDrawPDTarget;
    bool mDrawJointSphere;
    bool mDrawFootStep;
    bool mStochasticPolicy;
    bool mDrawEOE;

    MuscleRenderingType mMuscleRenderType;
    int mMuscleRenderTypeInt;

    float mMuscleResolution;

    // Muscle Rendering Option
    std::vector<Muscle *> mSelectedMuscles;
    std::vector<bool> mRelatedDofs;
    
    // Screen Record
    bool mScreenRecord;
    int mScreenIdx;

    // For BVH Export
    std::vector<Eigen::VectorXd> mMotionBuffer;
    std::vector<Eigen::Matrix3d> mJointCalibration;

};