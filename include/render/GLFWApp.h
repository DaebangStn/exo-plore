#ifndef MSS_GLFWAPP_H
#define MSS_GLFWAPP_H

#include "core/Record.h"
#include "render/ShapeRenderer.h"
#include "render/GLFunctions.h"
#include "util/struct.h"
#include "dart/gui/Trackball.hpp"
#include "imgui.h"
#include "implot.h"
#include "torch/script.h"
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <ctime>
#include <chrono>

#define SWING_PHASE false
#define STANCE_PHASE true

struct GLFWwindow;
using namespace dart::dynamics;
namespace MASS 
{

class Environment;
class Muscle;

enum class FocusMode { Root, Foot, Unfocus, Count };
enum class DrawMode { Mesh, Primitive, NoSimUpdate, Count };
enum class ControlMode { Inference, FixUpper, FixWhole, Count };
enum class GroundMode { Checker, Chroma, Count };

struct RolloutStatus{
	std::string name, settingPath, fileContents, storeKey;
	char memo[64];
	int cycle; // Left number of gait cycles. if -1, rollout continues, if 0, rollout is finished
	bool pause;
	bool modulate;
	bool isAlarmed;
	bool store;
	bool fake_assist;
	std::deque<YAML::Node> params;

	RolloutStatus() {reset();}
	void reset(){
		cycle = -1; pause = true; modulate = false; isAlarmed = false; store = false; fake_assist = false;
		name = ""; settingPath = ""; fileContents = ""; storeKey = ""; 
		std::memset(memo, 0, sizeof(memo));
		params.clear();
	}

	void step(){
		if (cycle > 0) cycle--;
		if (cycle == 0) pause = true;
	}

	bool paramExists() const { return !params.empty(); }
	int paramSize() const { return params.size(); }
	std::unordered_map<std::string, float> popParams() {
		if (params.empty()) {
			std::cerr << "[GUI] Error: No parameters to pop" << std::endl;
			return {};
		}

		YAML::Node ret = params.front();
		params.pop_front();

		bool parsed = false;
		if (ret["cycle"].IsDefined() && ret["cycle"].IsScalar()) {
			cycle = ret["cycle"].as<int>();
			ret.remove("cycle");
			parsed = true;
		}

		store = false;
		storeKey = "";
		if (ret["store"].IsDefined() && ret["store"].IsScalar()) {
			store = true;
			storeKey = ret["store"].as<std::string>(); 
			ret.remove("store");
			parsed = true;
		}
		if (ret["fake_assist"].IsDefined() && ret["fake_assist"].IsScalar()) {
			fake_assist = true;
			ret.remove("fake_assist");
			parsed = true;
		}
		if (!parsed) std::cerr << "[GUI] Warning: No cycle or store found for the RolloutStatus parameter" << std::endl;

		if (ret["modulate"].IsDefined() && ret["modulate"].IsScalar()) {
			modulate = static_cast<bool>(ret["modulate"].as<float>()); 
			ret.remove("modulate");
		}
		std::unordered_map<std::string, float> env_params = ret.as<std::unordered_map<std::string, float>>();
		pause = false;
		return env_params;
	}
};

class GLFWApp
{
public:
	GLFWApp(const RenderArg &args);
	~GLFWApp();
    
	void startLoop();
	void Reset();
	std::vector<Record*> exportData(const std::unordered_map<std::string, float>& params, int cycle, bool render=false,
                                    bool do_modulate=true);
private:
	const RenderArg mArgs;
    const Eigen::Vector4d color_green = Eigen::Vector4d(0.6, 1.2, 0.6, 0.8);
    const Eigen::Vector4d color_red = Eigen::Vector4d(1.2, 0.6, 0.6, 0.8);
    const Eigen::Vector4d color_blue = Eigen::Vector4d(0.6, 0.6, 1.2, 0.8);
    static constexpr int NUM_PLOT_COLORS = 21;
    const ImVec4 plot_color_list[NUM_PLOT_COLORS] = {
		ImVec4(1.0f, 1.0f, 1.0f, 1.0f),     // White
		ImVec4(1.0f, 0.0f, 0.0f, 1.0f),     // Red
		ImVec4(0.0f, 1.0f, 0.0f, 1.0f),     // Green
		ImVec4(0.0f, 0.5f, 1.0f, 1.0f),     // Sky Blue
		ImVec4(1.0f, 1.0f, 0.0f, 1.0f),     // Yellow
		ImVec4(1.0f, 0.0f, 1.0f, 1.0f),     // Magenta
		ImVec4(0.0f, 1.0f, 1.0f, 1.0f),     // Cyan
		ImVec4(1.0f, 0.5f, 0.0f, 1.0f),     // Orange
		ImVec4(0.6f, 0.2f, 1.0f, 1.0f),     // Purple
		ImVec4(0.5f, 0.5f, 0.5f, 1.0f),     // Gray
		ImVec4(0.0f, 0.6f, 0.3f, 1.0f),     // Dark Green
		ImVec4(0.7f, 0.0f, 0.2f, 1.0f),     // Burgundy
		ImVec4(0.0f, 0.4f, 0.7f, 1.0f),     // Deep Blue
		ImVec4(0.7f, 0.7f, 0.0f, 1.0f),     // Olive
		ImVec4(0.7f, 0.0f, 0.7f, 1.0f),     // Violet
		ImVec4(0.0f, 0.7f, 0.7f, 1.0f),     // Teal
		ImVec4(1.0f, 0.3f, 0.3f, 1.0f),     // Salmon
		ImVec4(0.3f, 1.0f, 0.3f, 1.0f),     // Light Green
		ImVec4(0.3f, 0.3f, 1.0f, 1.0f),     // Light Blue
		ImVec4(1.0f, 0.7f, 0.3f, 1.0f),     // Light Orange
		ImVec4(0.5f, 1.0f, 0.8f, 1.0f),     // Mint
    };
    const int mNumPlotColor = NUM_PLOT_COLORS;
	std::unordered_map<std::string, double> mEnergyRatio = {
		{"Flexions", 0.1482},
		{"Ankle", 0.2723},
		{"Hip", 0.3985},
		{"Knee", 0.1810},
	};

	void _stepPd(Record* pRecord = nullptr);
	void _stepPdOnce();
	void _step(Record* pRecord = nullptr);
	void _stepNetwork();
	void _stepRollout();
	void _testReset(int num_reset=100);
	void _updateGaitCycleData();
	void _summarizeGaitCycleData();

	void TickControlMode();

	void renderWindow();
	void keyboardPress(int key, int scancode, int action, int mods);
    void mouseMove(double xpos, double ypos);
    void mousePress(int button, int action, int mods);
    void mouseScroll(double xoffset, double yoffset) {mZoom += yoffset * 0.01;};

	void _loadWindowSetting();
	void _loadRolloutSetting(const std::string& path);
	void _initGLFW();
	void _initImGui();
	void _initEnv();
	void _resetPlotData();
	void _resetAccumData();
	void _tickDrawMode();

	void SetFocusing();
    void initGL();
	void initFog();
	void initLights();
	void SetMuscleColor();

	// Draw Simulation Frame
	void DrawSimFrame();
	void DrawGround();
	void DrawMuscleTorque();
	void DrawCollision();
	void DrawJoint();
	void DrawNodeCOM();
	void DrawMuscles(const std::vector<Muscle*>& muscles);
	void DrawSkeleton(const dart::dynamics::SkeletonPtr& skel){DrawBodyNode(skel->getRootBodyNode());};
	void DrawTitle();
	void DrawBodyNode(const dart::dynamics::BodyNode* bn);
	void DrawShapeFrame(const dart::dynamics::ShapeFrame* shapeFrame);
	void DrawShape(const dart::dynamics::Shape* shape,const Eigen::Vector4d& color);
	void DrawEntity(const dart::dynamics::Entity* entity);
	void DrawAiMesh(const struct aiScene *sc, const struct aiNode* nd,const Eigen::Affine3d& M,double y);
	void DrawShadow(const Eigen::Vector3d& scale, const aiScene* mesh, double y);
    void DrawForceArrow(float force, const Eigen::Vector3d& dir, const Eigen::Vector3d& pos, const Eigen::Vector4d& color);
    void DrawDevice();
    void DrawDeviceTorque();
    void DrawDeviceTorqueFromThigh();
	void DrawHipTorque();
	void DrawFootStep();
	void DrawPhaseState();
	void _drawDesiredTorque();

	void _drawPhase();
	// Draw UI Frame
	void _drawGui();
    void _drawController();
	void _drawSimData();
	void _drawSimulationInformation();
	void _drawCharacterInformation();
	void _drawMetadata();
	void _drawRolloutParameters();
	void _drawReward();
	void _drawState();
	void _drawAction();
    void _drawPerCycle();
    void _drawDevice();
    void _drawGaitAnalysis();
    void _drawMax();
    void DrawUIDisplay_Metabolic();
    void DrawUIDisplay_MuscleForce();

	std::pair<double, double> _setupGraphAxes(const CBufferData& graph_data, const std::vector<std::string>& keys, 
		int default_x_len, ImAxis y_axis, double y_min, double y_max, bool show_phase);
	void _plotHorizontalBar(const std::map<std::string, float>* horizontal, double x_min, double x_max);
	void _plotPhaseBar(const CBufferData& graph_data, double x_min, double x_max, double y_min, double y_max);
    void _plotHeadlessGraphData(const CBufferData& graph_data, const std::vector<std::string>& keys, ImAxis y_axis, 
		bool show_phase=true, bool plot_avg_copy=false, std::string postfix="");
	void _plotGraphData(const CBufferData& graph_data, const std::vector<std::string>& keys, const std::string& title, 
        double y_min, double y_max, const std::map<std::string, float>* horizontal=nullptr, bool show_phase=true, 
        float height=300, int default_x_len=-1, bool hide_legend=false); // TODO: hide_legend is not working
	void _plotMA2Part();
	void _storeGraphDataKeys(const char* search_key, bool verbose=false);
	void _showGraphDataKeys(const char* search_key, bool verbose=false);
	void _printGraphDataKeys(const char* search_key, bool verbose=false);
	void _printContactPhase(bool verbose=false);
	void _setXminToHeelStrike();
	bool _captureRegionPNG(const char* filename, int x0, int y0, int x1, int y1);
	
	// Video Recording functions
	bool _startVideoRecording(const std::string& filename, int fps = 30);
	void _stopVideoRecording();
	void _recordVideoFrame();
	bool _shouldRecordFrame();

    void calculateAndDisplayFPS(double& lastTime, int& frameCount);
	bool checkFailed(const std::string& msg = "") const;
    std::chrono::steady_clock::time_point lastSaveTime;
    const std::chrono::milliseconds saveInterval = std::chrono::milliseconds(500);

	NN mNN;
	std::map<std::string, bool> mMuscleRenderMap;
	std::map<std::string, bool> mMuscleActivationMap;
	std::map<int, bool> mMuscleDofsSelectMap;
	std::map<int, std::vector<std::string>> mMuscleDofsRenderMap;
	std::map<std::string, int> mMuscleDofsCountMap;
	float mDrawMuscleRange;

    GLFWwindow* mWindow;
	Environment* mEnv;
	CBufferData mGraphData, mGraphCycleData, mStoredGraphData;
	MinMaxBuffer mCycleMinMax;
	AccumData mCycleAccumDistance;
	AccumRMSData mCycleAccumRMS;
	PositiveAccumData mCyclePositiveAccumDistance;
	BufferData mCycleAccumStep;
	RolloutStatus mRolloutStatus;

	Eigen::Affine3d mViewMatrix;
	dart::gui::Trackball mTrackball;
	Eigen::Vector3d mTrans = Eigen::Vector3d(0.0, 0.0, 0.0), mEye = Eigen::Vector3d(0.0, 0.0, 1.0), mUp = Eigen::Vector3d(0.0, 1.0, 0.0);

	float mZoom = 0.3, mPersp = 45.0, mMouseX = 0.0, mMouseY = 0.0;
	double mXmin = 0.0, mXmax = 0.0, mPlotAvgTxtInterval = 20;
	bool mCapture = false, mRotate = false, mTranslate = false, mZooming = false, mAccumData = false, mDisplayMetric = false;
	double mWidth = 1920, mHeight = 1200, mControlWidth = 350, mPlotWidth = 500, mXpos = 100, mYpos = 100;
	float mPlotAvgFontScale = 1.25, mPlotAvgTxtOffset = -20; 
	int xpos = -100000, ypos = -100000, mCameraMoving = 0, mPlotAvgDecimal = 1;
	ShapeRenderer mShapeRenderer;
	
	int mFramerate = 60;
	float mMeasuredFps = 0.0;

	bool mDrawMuscleTotal = true, mDrawMuscleDofsTotal = true, mDrawMuscleMode = true, mDrawMuscleAnchorMode = false, 
		mDrawMuscleRangePart = true, mActivateMuscleTotal = true, mCorrelateParams = false, mDrawGround = true, mDrawTitle = false,
		mDrawCharacter = true, mDrawMuscle = true, mDrawDevice = true, mDrawDeviceTorque = true, mDrawDeviceTorqueFromThigh = true, mDrawFootStep = false, 
		mDrawPhaseState = false, mDrawCollision = false, mDrawReference = false, mDrawMuscleTorque = false, 
		mDrawMusclePassive = false, mDrawHipTorque = false, mDrawNodeCOM = false, mDrawJoint = false, 
		mPlotHideLegend = false, mPlotAverage = false, mPlotJitter = false, mPlotDiff = false, mPlotTitle = true, mPlotShowSmooth = false,
		mIsDeviceDrawMode = false, mExoTune1 = false, mExoTune2 = false;
	bool mOpendGraphTab = false;

	int mRolloutCount = 30;

	float mPlotSmoothAlpha = -1.0f;
	float mGroundHeight = 0.0f;

	int mNumDof;
	int mIncludeAction = 0;

	Eigen::VectorXd mDesiredTorquePrev, mDisplacement, mMuscleTorque;

	bool* mSelectedParameter;
	bool mIsSelectedMode;

	bool phase1_saved = false, phase2_saved = false;
	float mStanceRatioL_no, mSwingRatioL_no, mStanceRatioR_no, mSwingRatioR_no, mStanceRatioL_yes, mSwingRatioL_yes, 
		mStanceRatioR_yes, mSwingRatioR_yes;
	char mSearchKey[64];

	std::string mSelectedMuscleName = "", mSelectedJointName = "", curBody = "";
    YAML::Node mParamCfg = YAML::Node();

    Utils::ModeManager<FocusMode> mFocusMode;
    Utils::ModeManager<DrawMode> mDrawMode = DrawMode::Mesh;
    Utils::ModeManager<ControlMode> mControlMode;
    Utils::ModeManager<GroundMode> mGroundMode = GroundMode::Chroma;

	std::map<std::string, std::vector<double>> mBarXCoords;  // Cache for bar x coordinates
	std::vector<std::string> mShowGraphData;  // Cache for bar x coordinates
	bool mFixWeightEnergy = false;
	int mNumCycRemainFixWeightMA2 = 10, mStoreColorOffset = 3;
	std::unordered_map<std::string, double> mWeightMA2;
	ImPlotFlags mPlotPie = ImPlotFlags_NoMouseText | ImPlotFlags_NoInputs;
    // Capture UI state
    bool mCaptureShowRect = false;
    int mCaptureX0 = 100, mCaptureY0 = 100, mCaptureX1 = 500, mCaptureY1 = 400; // in window coords
    char mCaptureFilename[256] = "capture.png";
    
    // Video Recording state
    bool mVideoRecording = false;
    FILE* mFFmpegPipe = nullptr;
    char mVideoFilename[256] = "simulation_video.mp4";
    int mVideoFPS = 50;
    double mVideoMaxTime; // Maximum recording time in seconds
    double mVideoElapsedTime = 0.0;
    int mVideoFrameSkip = 1; // Skip frames if simulation is too fast
    int mVideoFrameCounter = 0;
    double mLastVideoTime = 0.0;
    double mCachedControlStepTime = 0.0; // Cached control step time
    bool mVideoStopRollout = true; // Stop rollout when recording finishes
};
};

#endif
