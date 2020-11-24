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
#include "indicators/block_progress_bar.hpp"
#include <sstream>

#define SWING_PHASE false
#define STANCE_PHASE true

struct GLFWwindow;
using namespace dart::dynamics;
namespace MASS 
{

enum class DrawMode { Mesh, Primitive, NoSimUpdate, Count };
enum class FocusMode { Root, Foot, Unfocus, Count };
enum class SweepMode { Stop, Pause, Run, Count };
enum class SweepType { Torque, Angle, Count };
class Environment;
class Muscle;

struct SweepConfig{
    int data_idx, num_step, current_step;
    float data_bound1, data_bound2;
	bool draw_on_step = true;
	std::string description;
	char memo[64];
	std::vector<Eigen::VectorXd> mActivationHistory;
    Utils::ModeManager<SweepMode> sweep_mode;
    Utils::ModeManager<SweepType> sweep_type;
    std::unique_ptr<indicators::BlockProgressBar> progress_bar;

    SweepConfig(){reset();}
	void reset(){
		data_idx = 6;
		data_bound1 = 0.0;
		data_bound2 = 0.0;
		current_step = 0;
		num_step = 5000;
		mActivationHistory.clear();
		description = "";
		memset(memo, 0, sizeof(memo));
		sweep_mode.setMode(SweepMode::Stop);
		sweep_type.setMode(SweepType::Torque);
	}
	void run(){
		// sanity check
		if (data_idx < 0) return;
		if (sweep_mode.getMode() == SweepMode::Run) return;
		if (num_step <= 0) return;

		sweep_mode.setMode(SweepMode::Run);
		mActivationHistory.clear();
		mActivationHistory.reserve(num_step + 1);
		current_step = 0;
		std::ostringstream oss;
		oss.precision(2);
		oss << std::fixed;
		oss << "Sweep(idx=" << data_idx << ", [" << data_bound1 << ", " << data_bound2 << "])";
		description = oss.str();
		progress_bar = std::make_unique<indicators::BlockProgressBar>(
			indicators::option::BarWidth{50},
			indicators::option::PrefixText{description.c_str()},
			indicators::option::ShowElapsedTime{true},
			indicators::option::ShowRemainingTime{true},
			indicators::option::MaxProgress{static_cast<uint32_t>(num_step + 1)}
		);
		progress_bar->set_progress(0);
	}
	std::vector<double> getActivationHistory(int muscle_idx) const {
		if (mActivationHistory.empty()) return std::vector<double>();
		
		// Get the number of muscles from the last element in mActivationHistory
		const int total_muscles = mActivationHistory[0].size();
		if (muscle_idx < 6 || muscle_idx >= total_muscles) {
			std::cerr << "Invalid muscle index: " << muscle_idx << " among " << total_muscles << " muscles" << std::endl;
			return std::vector<double>();
		}

		std::vector<double> activation_history;
		activation_history.reserve(num_step + 1);
		for (const auto& activation : mActivationHistory){
			activation_history.emplace_back(activation[muscle_idx]);
		}
		return activation_history;
	}
	void pushActivation(const Eigen::VectorXd& activation){
		if (sweep_mode.getMode() != SweepMode::Run) return;
		mActivationHistory.emplace_back(activation);
	}
	std::vector<double> sweepData() const {
		std::vector<double> data;
		data.reserve(num_step + 1);
		for (int i = 0; i <= num_step; i++){
			data.emplace_back(data_bound1 + (data_bound2 - data_bound1) * i / num_step);
		}
		return data;
	}
	std::pair<int, double> step(){
		int return_idx = -1;
		if (sweep_mode.getMode() == SweepMode::Run){
			if (current_step > num_step) sweep_mode.setMode(SweepMode::Stop);
			else {
				return_idx = data_idx;
				current_step++;
				if (progress_bar) {
					progress_bar->set_progress(current_step);
					if (current_step >= num_step + 1) progress_bar->mark_as_completed();
				}
			}
			
		}
		return std::make_pair(return_idx, data_bound1 + (data_bound2 - data_bound1) * current_step / num_step);
	}

	SweepMode getMode() const {return sweep_mode.getMode();}
	void setMode(SweepMode mode){sweep_mode.setMode(mode);}

};

class Activation
{
public:
	Activation(const RenderArg &args);
	~Activation();
	void startLoop();
	void Reset();

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

	void _step();
	void _resetPlotData();
	void _updatePlotActMuscles();

	void renderWindow();
	void keyboardPress(int key, int scancode, int action, int mods);
    void mouseMove(double xpos, double ypos);
    void mousePress(int button, int action, int mods);
    void mouseScroll(double xoffset, double yoffset) {mZoom += yoffset * 0.01;};

	void _loadWindowSetting();
	void _initGLFW();
	void _initImGui();
	void _initEnv();
	void _tickDrawMode();

	void SetFocusing();
    void initGL();
	void initFog();
	void initLights();
	void SetMuscleColor();

	// Draw Simulation Frame
	void DrawSimFrame();
	void DrawGround();
	void DrawCollision();
	void DrawJoint();
	void DrawNodeCOM();
	void DrawMuscles(const std::vector<Muscle*>& muscles);
	void DrawSkeleton(const dart::dynamics::SkeletonPtr& skel){DrawBodyNode(skel->getRootBodyNode());};
	void DrawBodyNode(const dart::dynamics::BodyNode* bn);
	void DrawShapeFrame(const dart::dynamics::ShapeFrame* shapeFrame);
	void DrawShape(const dart::dynamics::Shape* shape,const Eigen::Vector4d& color);
	void DrawEntity(const dart::dynamics::Entity* entity);
	void DrawAiMesh(const struct aiScene *sc, const struct aiNode* nd,const Eigen::Affine3d& M,double y);
	void DrawShadow(const Eigen::Vector3d& scale, const aiScene* mesh, double y);
    void DrawForceArrow(float force, const Eigen::Vector3d& dir, const Eigen::Vector3d& pos, const Eigen::Vector4d& color);
	void DrawDesiredTorque();

	// Draw UI Frame
	void _drawGui();
    void _drawController();
	void _drawSimData();
	void _drawCharacterInformation();
	void _drawMetadata();
	void _drawActivationHistory();

	std::pair<double, double> _setupGraphAxes(const CBufferData& graph_data, const std::vector<std::string>& keys, 
		int default_x_len, ImAxis y_axis, double y_min, double y_max, bool show_phase);
    void _plotHeadlessGraphData(const CBufferData& graph_data, const std::vector<std::string>& keys, 
        ImAxis y_axis, bool show_phase=true, bool hide_legend=false, bool plot_avg_txt=false, bool plot_avg_copy=false, 
		float y_alpha_smooth=-1.0f, std::string postfix="");
	void _plotGraphData(const CBufferData& graph_data, const std::vector<std::string>& keys, const std::string& title, 
        double y_min, double y_max, const std::map<std::string, float>* horizontal=nullptr, bool show_phase=true, 
        float height=300, int default_x_len=-1, bool hide_legend=false); // TODO: hide_legend is not working

    void calculateAndDisplayFPS(double& lastTime, int& frameCount);

	NN mNN;
	std::map<std::string, bool> mMuscleRenderMap;
	std::map<int, bool> mMuscleDofsSelectMap;
	std::map<int, std::vector<std::string>> mMuscleDofsRenderMap;
	std::map<std::string, int> mMuscleDofsCountMap;
	float mDrawMuscleRange;

    GLFWwindow* mWindow;
	Environment* mEnv;

	Eigen::Affine3d mViewMatrix;
	dart::gui::Trackball mTrackball;
	Eigen::Vector3d mTrans = Eigen::Vector3d(0.0, 0.0, 0.0), mEye = Eigen::Vector3d(0.0, 0.0, 1.0), mUp = Eigen::Vector3d(0.0, 1.0, 0.0);

	float mZoom = 0.25, mPersp = 45.0, mMouseX = 0.0, mMouseY = 0.0;
	double mXmin = 0.0, mXmax = 0.0;
	bool mCapture = false, mRotate = false, mTranslate = false, mZooming = false, mAccumData = false;
	double mWidth = 1920, mHeight = 1200, mControlWidth = 350, mPlotWidth = 500, mXpos = 100, mYpos = 100;
	int xpos = -100000, ypos = -100000, mCameraMoving = 0;
	ShapeRenderer mShapeRenderer;
	
	int mFramerate = 60;
	float mMeasuredFps = 0.0;
	float mForceScale = 0.01;
	float mPlotActMargin = 0.01f;

	bool mDrawMuscleTotal = true, mDrawMuscleDofsTotal = true, mDrawMuscleMode = true, mDrawMuscleAnchorMode = false, 
		mDrawMuscleRangePart = true, mCorrelateParams = false, mDrawGround = true, mDrawCharacter = true, mDrawMuscle = true,  
		mDrawCollision = false, mDrawMusclePassive = false, mDrawNodeCOM = false, mDrawJoint = false, mDrawDesiredTorque = true;

	int mNumDof;
	int mIncludeAction = 0;

	Eigen::VectorXd mActivation;

	bool* mSelectedParameter;
	bool mIsSelectedMode;

	std::string mSelectedMuscleName = "", mSelectedJointName = "", curBody = "";
    YAML::Node mParamCfg = YAML::Node();

    Utils::ModeManager<FocusMode> mFocusMode;
    Utils::ModeManager<DrawMode> mDrawMode = DrawMode::Primitive;
	std::vector<std::string> mPlotActMuscles, mWholeMuscles, mBaseMuscleNames;
	std::map<std::string, std::vector<int>> mPlotActMuscleIndices;
	std::map<std::string, std::vector<std::string>> mBaseToFullMuscles;
	SweepConfig mSweepConfig;

	void _updateBaseMuscleNames();
	double mGroundHeight;

	std::map<std::string, std::vector<double>> mBarXCoords;  // Cache for bar x coordinates
	std::vector<std::string> mShowGraphData;  // Cache for bar x coordinates
	ImPlotFlags mPlotPie = ImPlotFlags_NoMouseText | ImPlotFlags_NoInputs;
};
};

#endif
