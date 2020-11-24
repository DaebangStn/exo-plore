#ifndef STRUCT_H
#define STRUCT_H

#include "torch/script.h"
#include "eigen3/Eigen/Dense"
#include <boost/circular_buffer.hpp>

enum class EnergyMode { 
	A2, MA2, A3, 
    MA3, A, MA,
	M2A, M2A2, M2A3,
	M05A, M05A2, M05A3,
	A15, M05A15, MA15, M2A15,
	BHAR,
	A125, M05A125, MA125, M2A125,
	// MA2COT, // MA2 with COT (filtered)
	// MA2COT2, // MA2 with COT (fix per gait cycle)
	// A2ANK, MA2ANK,
	POWER,
	TAU,
	Count};

enum class EnergyRewardCurve { EXP, LINEAR1, LINEAR2, LINEAR3, LINEAR4, Count };
enum class JntType { Torque, FullMuscle, LowerMuscle, Count };
enum class StateType { 
    base 		= 0,
    sym_gait 	= 1 << 0,
    foot_clear 	= 1 << 1,
    velocity 	= 1 << 2,
    dt_hist 	= 1 << 3,
    mass_ratio 	= 1 << 4,
};
enum class ActionType {
    base 		= 0,
    time_warp 	= 1 << 0,
    upper 		= 1 << 1,
    mirror_act 	= 1 << 2,
    torque_clip = 1 << 3,
};
enum class SwayType {
    none 		= 0,
    arm 		= 1 << 0,
    pelvis 		= 1 << 1,
    spine 		= 1 << 2,
    foot_x 		= 1 << 3,
    foot_angle 	= 1 << 4,
};


struct NN
{
    bool useMuscle = false;
    torch::jit::script::Module sim;
    torch::jit::script::Module muscle;
    Eigen::VectorXd minv;
    Eigen::VectorXd maxv;

    torch::Tensor getTensorData(const std::string& name) const {
        return sim.attr(name).toTensor().to(torch::kCPU, torch::kDouble);
    }
};

struct ExpArg
{
    std::string input_csv;
    std::string torchscript_dir;
    std::string record_config;
    std::string output_dir;
    bool render = false;
};

struct RenderArg
{
    std::string torchscript_dir;
    bool force_device = false;
};

struct MuscleLengthArg
{
    std::string metadata;
    std::string output;
};

struct AccumData{
    std::unordered_map<std::string, double> values;
    AccumData() = default;
    template <typename... Args> 
    AccumData(Args... keys) {(values.emplace(keys, 0.0), ...);}
    AccumData(const std::vector<std::string>& keys) { for (const auto& key : keys) { values[key] = 0.0; } }
    void reset() { for(auto& [key, value] : values) value = 0.0; }
    bool key_exists(const std::string& key) const { return values.find(key) != values.end(); }
    void add_key(const std::string& key) { values[key] = 0.0; }
    std::deque<std::string> keys() const {
        std::deque<std::string> keys;
        for(const auto& [key, value] : values) keys.push_back(key);
        return keys;
    }
    void accumulate(const std::string& key, double value) { 
        if (values.find(key) == values.end()) {
            std::cerr << "[AccumData] Key not found: " << key << std::endl;
            return;
        }
        values[key] += value; 
    }
    void accumulate_abs(const std::string& key, double value) { 
        if (values.find(key) == values.end()) {
            std::cerr << "[AccumData] Key not found: " << key << std::endl;
            return;
        }
        values[key] += abs(value); 
    }
    double operator[](const std::string& key) const {return values.at(key);}
    std::unordered_map<std::string, double> divideBy(double divisor) {
        for(auto& [key, value] : values) values[key] = value / divisor;
        return values;
    }
    std::unordered_map<std::string, double> multipleBy(double multiplier) {
        for(auto& [key, value] : values) values[key] = value * multiplier;
        return values;
    }
};

struct AccumRMSData{
    std::unordered_map<std::string, double> values;
    std::unordered_map<std::string, int> counts;
    AccumRMSData() = default;
    template <typename... Args> AccumRMSData(Args... keys) {(values.emplace(keys, 0.0), ...), (counts.emplace(keys, 0), ...);}
    void reset() { for(auto& [key, value] : values) value = 0.0; for(auto& [key, value] : counts) value = 0; }
    bool key_exists(const std::string& key) const { return values.find(key) != values.end(); }
    void add_key(const std::string& key) { values[key] = 0.0; counts[key] = 0; }
    std::deque<std::string> keys() const {
        std::deque<std::string> keys;
        for(const auto& [key, value] : values) keys.push_back(key);
        return keys;
    }
    double operator[](const std::string& key) const { return sqrt(values.at(key) / counts.at(key)); }
    void accumulate(const std::string& key, double value) {
        if(!key_exists(key)) {
            std::cerr << "[AccumRMSData] Key not found: " << key << std::endl;
            return;
        }
        values[key] += value * value;
        counts[key] += 1;
    }
};

struct PositiveAccumData : public AccumData {
    using AccumData::AccumData; // Inherit constructors
    void accumulate(const std::string& key, double value) {
        if (value > 0) AccumData::accumulate(key, value);
    }
};

struct BufferData{
    std::unordered_map<std::string, std::deque<double>> buffers;
    size_t max_size = 1000;

    BufferData() = default;
    template <typename... Args> 
    BufferData(Args... keys) {(buffers.emplace(keys, std::deque<double>()), ...);}
    void reset() { for(auto& [key, value] : buffers) value.clear(); }
    double get_last(const std::string& key) const { 
        if(buffers.at(key).empty()) {
            std::cerr << "[BufferData] Key not found: " << key << " or empty returning 0.0" << std::endl;
            return 0.0;
        }
        return buffers.at(key).back(); 
    }
    bool key_exists(const std::string& key) const { return buffers.find(key) != buffers.end(); }
    void add_key(const std::string& key) { buffers[key] = std::deque<double>(); }
    void set_max_size(size_t size) { max_size = size; }
    std::deque<std::string> keys() const {
        std::deque<std::string> keys;
        for(const auto& [key, value] : buffers) keys.push_back(key);
        return keys;
    }
    void push(const std::string& key, double value) { 
        if (buffers.find(key) == buffers.end()) {
            std::cerr << "[BufferData] Key not found: " << key << std::endl;
            return;
        }
        if(max_size > 0 && buffers[key].size() >= max_size) {
            // std::cerr << "[BufferData] Buffer size exceeded: " << key << " max size: " << max_size << " popping front" << std::endl;
            buffers[key].pop_front();
        }
        buffers[key].push_back(value); 
    }
    std::unordered_map<std::string, double> average() const {
        std::unordered_map<std::string, double> avg;
        for(const auto& [key, value] : buffers) avg[key] = std::accumulate(value.begin(), value.end(), 0.0) / value.size();
        return avg;
    }
    const std::deque<double>& operator[](const std::string& key) const { return buffers.at(key); }
};


struct MinMaxBuffer{
    std::unordered_map<std::string, double> max_buffer;
    std::unordered_map<std::string, double> min_buffer;
    const double _inf = std::numeric_limits<double>::infinity();

    MinMaxBuffer() = default;
    template <typename... Args> 
    MinMaxBuffer(Args... keys) {(max_buffer.emplace(keys, -_inf), ...), (min_buffer.emplace(keys, _inf), ...);}
    void reset() { for(auto& [key, value] : max_buffer) value = -_inf; for(auto& [key, value] : min_buffer) value = _inf; }
    double get_max(const std::string& key) const { 
        if(!key_exists(key)) {
            std::cerr << "[MinMaxBuffer] Key not found: " << key << std::endl;
            return -_inf;
        }
        return max_buffer.at(key); 
    }
    double get_min(const std::string& key) const { 
        if(!key_exists(key)) {
            std::cerr << "[MinMaxBuffer] Key not found: " << key << std::endl;
            return _inf;
        }
        return min_buffer.at(key); 
    }
    bool key_exists(const std::string& key) const { return max_buffer.find(key) != max_buffer.end(); }
    void add_key(const std::string& key) { max_buffer[key] = -_inf; min_buffer[key] = _inf; }
    std::deque<std::string> keys() const {
        std::deque<std::string> keys;
        for(const auto& [key, value] : max_buffer) keys.push_back(key);
        return keys;
    }
    void push(const std::string& key, double value) { 
        if (!key_exists(key)) {
            std::cerr << "[MinMaxBuffer] Key not found: " << key << std::endl;
            return;
        }
        max_buffer[key] = std::max(max_buffer[key], value); 
        min_buffer[key] = std::min(min_buffer[key], value); 
    }
    std::pair<double, double> operator[](const std::string& key) const { return {min_buffer.at(key), max_buffer.at(key)}; }
};

struct CBufferData{
    std::unordered_map<std::string, boost::circular_buffer<double>> buffers;
    const size_t graph_data_size = 1000;
    CBufferData() = default;
    template <typename... Args> 
    CBufferData(Args... keys) {(buffers.emplace(keys, boost::circular_buffer<double>(graph_data_size)), ...);}
    void reset() { for(auto& [key, value] : buffers) value.clear(); }
    void clear() { buffers.clear(); }
    void reset_key(const std::string& key) { buffers[key].clear(); }
    void deepcopy(const std::string& key, const CBufferData& target_data) { 
        if (!key_exists(key)) add_key(key);
        reset_key(key);
        const auto& target_buffer = target_data[key];
        buffers[key].assign(target_buffer.begin(), target_buffer.end());
    }
    double get_last(const std::string& key) const { return buffers.at(key).back(); }
    bool key_exists(const std::string& key) const { return buffers.find(key) != buffers.end(); }
    bool key_exists(const std::vector<std::string>& keys) const {
        for(const auto& key : keys) if(!key_exists(key)) return false;
        return true;
    }
    void add_key(const std::string& key) { buffers[key] = boost::circular_buffer<double>(graph_data_size); }
    std::deque<std::string> keys() const {
        std::deque<std::string> keys;
        for(const auto& [key, value] : buffers) keys.push_back(key);
        return keys;
    }
    void push(const std::string& key, double value) { 
        if (buffers.find(key) == buffers.end()) {
            std::cerr << "[CBufferData] Key not found: " << key << std::endl;
            return;
        }
        buffers[key].push_back(value); 
    }
    void push_ma(const std::string& key, double value, double alpha = 0.005) {
        if (buffers.find(key) == buffers.end()) {
            std::cerr << "[CBufferData] Key not found: " << key << std::endl;
            return;
        }
        if(buffers[key].empty()) {
            buffers[key].push_back(value);
            return;
        }
        buffers[key].push_back(value * alpha + buffers[key].back() * (1 - alpha));
    }
    const boost::circular_buffer<double>& operator[](const std::string& key) const { return buffers.at(key); }
};

ExpArg export_arguments(int argc, char* argv[]);
RenderArg render_arguments(int argc, char* argv[]);
MuscleLengthArg muscle_length_arguments(int argc, char* argv[]);

// Conversions
torch::Tensor eigen_vec_to_torch(const Eigen::VectorXd& e);
torch::Tensor eigen_mat_to_torch(const Eigen::MatrixXd& e);
torch::Tensor eigen_mat_to_torch_f(Eigen::MatrixXf e);
Eigen::VectorXd torch_to_eigen_vec(const torch::Tensor& t);

#endif //STRUCT_H
