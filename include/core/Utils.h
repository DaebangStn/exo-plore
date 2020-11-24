#ifndef __MASS_UTILS_H__
#define __MASS_UTILS_H__

#include "dart/dart.hpp"
#include <type_traits>
#include <boost/circular_buffer.hpp>

namespace MASS
{

using namespace dart::dynamics;
namespace Utils
{
    template <typename ModeType>
    class ModeManager {
        static_assert(std::is_enum<ModeType>::value, "ModeManager requires an enum type!");
    public:
        ModeManager() : currentMode(static_cast<ModeType>(0)) {}
        ModeManager(ModeType initialMode) : currentMode(initialMode) {}

        ModeType nextMode() {
            currentMode = static_cast<ModeType>((static_cast<int>(currentMode) + 1) % static_cast<int>(ModeType::Count));
            return currentMode;
        }
        void setMode(ModeType newMode) {currentMode = newMode;}
        ModeType getMode() const {return currentMode;}
        [[nodiscard]] std::string modeToString() const {return modeStrings[static_cast<int>(currentMode)];}
        static void setModeStrings(const std::vector<std::string>& strings) {modeStrings = strings;}

    private:
        ModeType currentMode;
        static std::vector<std::string> modeStrings;
    };

    bool IsYaml(const std::filesystem::path &path);
    bool IsYaml(const std::string &path);
    bool IsSameValue(const std::unordered_map<std::string, float>& a, const std::unordered_map<std::string, float>& b);
    inline bool case_insensitive_compare(const std::string &str1, const std::string &str2){
        return ((str1.size() == str2.size()) && std::equal(str1.begin(), str1.end(), str2.begin(), [](char a, char b) {
            return tolower(a) == tolower(b);
        }));
    }
    inline bool case_insensitive_starts_with(const std::string &str1, const std::string &str2){
        return str1.size() >= str2.size() && std::equal(str2.begin(), str2.end(), str1.begin(), [](char a, char b) {
            return tolower(a) == tolower(b);
        });
    }
    inline bool startsWith(const std::string &str, const std::string &prefix){ return str.rfind(prefix, 0) == 0;}
    double clamp(double v, double min_v, double max_v);
    float clamp(float v, float min_v, float max_v);
    Eigen::Vector3d clamp(Eigen::Vector3d v, double min_v, double max_v);
    template <typename Derived>
    void clip_vector(Eigen::MatrixBase<Derived>& v, typename Derived::Scalar min, typename Derived::Scalar max) {
        v = v.array().min(max).max(min);
    }
    template <typename Derived>
    void clip_vector(Eigen::MatrixBase<Derived>& v,
                     const Eigen::MatrixBase<Derived>& min,  // Pass by value
                     const Eigen::MatrixBase<Derived>& max) {
        v = v.array().min(max.array()).max(min.array());
    }
    bool compare(std::pair<double,double> a, std::pair<double,double> b);
    std::vector<int> sort_indices(const std::vector<double>& val);
    void printFirstN(const Eigen::VectorXd& vector, int n);
    Inertia addInertia(const Inertia& inertia1, const Inertia& inertia2);
    bool close(double a, double b, double epsilon = 1e-4f);
    double exp_of_squared(const Eigen::VectorXd &vec, double w);
    double exp_of_squared(const Eigen::Vector3d &vec, double w);
    double exp_of_squared(double val, double w);
    double exp_of_root(double val, double w);
    double exp_of_L1_norm(const Eigen::VectorXd &vec, double w);
    double exp_of_L1_norm(const Eigen::Vector3d &vec, double w);
    double exp_of_L1_norm(double val, double w);

    void setSkelPos(SkeletonPtr skel, Eigen::VectorXd pos);
    void setSkelVel(SkeletonPtr skel, Eigen::VectorXd vel);
    void setSkelPosAndVel(SkeletonPtr skel, Eigen::VectorXd pos, Eigen::VectorXd vel);

    void DataStandardize(std::vector<double>& toData, std::vector<double>& fromData, int div, bool filter);    
    std::string VectorXd2String(const Eigen::VectorXd& input);
    Eigen::Vector3d String2Vector3d(const std::string& input);
    std::vector<double> Split2Double(const std::string& input, int num);
    std::vector<std::unordered_map<std::string, float>> ParseParamCsv(const std::filesystem::path &csv_path);

    struct MAFilter {
        double alpha;
        double state;
        bool firstUpdate;
        MAFilter(double alpha) : alpha(alpha), state(0.0), firstUpdate(true) {}
        double get() const { return state; }
        double update(double value) {
            if (firstUpdate) {
                state = value;
                firstUpdate = false;
            }else {
                state = alpha * value + (1 - alpha) * state;
            }
            return state;
        }
        void reset(){
            firstUpdate = true;
            state = 0.0;
        }
    };

    template <typename Enum>
    class FlagSet {
        static_assert(std::is_enum_v<Enum>, "FlagSet requires an enum type");
        using Underlying = std::underlying_type_t<Enum>;

    public:
        // Default constructor: no flags set
        FlagSet() : value_(0) {}

        // Construct from a single enum value
        FlagSet(Enum flag) : value_(static_cast<Underlying>(flag)) {}

        // Combine flags with |
        FlagSet operator|(const FlagSet& other) const {
            return FlagSet(value_ | other.value_);
        }

        // Combine flags with &
        FlagSet operator&(const FlagSet& other) const {
            return FlagSet(value_ & other.value_);
        }

        // Check if a flag is set
        bool has(Enum flag) const {
            return (value_ & static_cast<Underlying>(flag)) != 0;
        }

        // Add a flag
        FlagSet& add(Enum flag) {
            value_ |= static_cast<Underlying>(flag);
            return *this;
        }

        // Remove a flag
        FlagSet& remove(Enum flag) {
            value_ &= ~static_cast<Underlying>(flag);
            return *this;
        }

        // Implicit conversion to underlying value for compatibility
        explicit operator Underlying() const { return value_; }

        // Equality operator
        bool operator==(const FlagSet& other) const {
            return value_ == other.value_;
        }

        // Inequality operator
        bool operator!=(const FlagSet& other) const {
            return value_ != other.value_;
        }

    private:
        FlagSet(Underlying value) : value_(value) {}
        Underlying value_;
    };

    // Allow combining enum values directly with |
    template <typename Enum>
    FlagSet<Enum> operator|(Enum lhs, Enum rhs) {
        return FlagSet<Enum>(lhs) | FlagSet<Enum>(rhs);
    }

    template <typename Enum>
    FlagSet<Enum> fromInt(int value) {
        return FlagSet<Enum>(static_cast<Enum>(value));
    }
}

}

#endif