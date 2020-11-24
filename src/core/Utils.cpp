#include <iostream>
#include "core/Utils.h"
#include <boost/tokenizer.hpp>
#include <type_traits>

using namespace std;


namespace MASS::Utils
{

    bool IsYaml(const filesystem::path &path)
    {
        return path.extension().string() == ".yaml";
    }

    bool IsYaml(const string &path)
    {
        return filesystem::path(path).extension().string() == ".yaml";
    }

    bool IsSameValue(const std::unordered_map<std::string, float>& a, const std::unordered_map<std::string, float>& b){
        if(a.size() != b.size()) return false;
        for (const auto& [key, value] : a) {
            auto it = b.find(key);
            if (it == b.end() || !close(value, it->second, 1e-5)) return false;
        }
        return true;
    }

vector<unordered_map<string, float>> ParseParamCsv(const filesystem::path &csv_path)
{
    ifstream file(csv_path);
    if (!file.is_open())
    {
        cerr << "Error: Could not open parameter csv " << csv_path << endl;
        exit(1);
    }
    typedef boost::tokenizer<boost::escaped_list_separator<char>> Tokenizer;
    std::string line;

    vector<unordered_map<string, float>> param_list;
    getline(file, line);
    Tokenizer header_tok(line);
    vector<string> header(header_tok.begin(), header_tok.end());
    for (auto &col : header)
    {
        std::transform(col.begin(), col.end(), col.begin(), [](unsigned char c) { return std::tolower(c); });
    }

    while (getline(file, line))
    {
        bool is_empty_or_whitespace = true;
        for (char c : line)
        {
            if (!isspace(c))
            {
                is_empty_or_whitespace = false;
                break;
            }
        }
        if (is_empty_or_whitespace)
        {
            continue;
        }

        Tokenizer tok(line);
        vector<string> data(tok.begin(), tok.end());
        unordered_map<string, float> param;
        if(header.size() != data.size())
        {
            cerr << "Error: CSV header and data size mismatch" << endl;
            cerr << "Header size: " << header.size() << " Data size: " << data.size() << " At line " << line << endl;
            exit(1);
        }
        for (int i = 0; i < header.size(); i++)
        {
            try{
                param[header[i]] = stof(data[i]);
            } catch (const std::invalid_argument& e) {
                cerr << "Error: Invalid argument in CSV at line " << line << endl;
                exit(1);
            }
        }
        param_list.push_back(param);
    }
    file.close();
    return param_list;
}

void printFirstN(const Eigen::VectorXd& vector, int n) {
    n = std::min(n, static_cast<int>(vector.size()));
    for (int i = 0; i < n; ++i) {
        std::cout << vector[i] << " ";
    }
    std::cout << std::endl;
}

std::string VectorXd2String(const Eigen::VectorXd& input){
    std::string res;
    for(int i=0; i<input.rows(); i++)
        res+=std::to_string(input[i])+" ";
    return res;
}

Eigen::Vector3d String2Vector3d(const std::string& input){
    std::vector<double> v = Split2Double(input, 3);
    Eigen::Vector3d res;
    res << v[0], v[1], v[2];

    return res;
}

Inertia addInertia(const Inertia& inertia1, const Inertia& inertia2) {
    // Extract properties
    double mass1 = inertia1.getMass();
    Eigen::Vector3d com1 = inertia1.getLocalCOM();
    Eigen::Matrix3d moment1 = inertia1.getMoment();

    double mass2 = inertia2.getMass();
    Eigen::Vector3d com2 = inertia2.getLocalCOM();
    Eigen::Matrix3d moment2 = inertia2.getMoment();

    // Combined mass
    double totalMass = mass1 + mass2;

    // Combined COM (weighted average)
    Eigen::Vector3d totalCOM = (mass1 * com1 + mass2 * com2) / totalMass;

    // Combined moment of inertia using Parallel Axis Theorem
    Eigen::Vector3d d1 = com1 - totalCOM;
    Eigen::Vector3d d2 = com2 - totalCOM;

    Eigen::Matrix3d totalMoment = moment1 + mass1 * (d1 * d1.transpose() - d1.squaredNorm() * Eigen::Matrix3d::Identity()) +
                                  moment2 + mass2 * (d2 * d2.transpose() - d2.squaredNorm() * Eigen::Matrix3d::Identity());

    // Construct the resulting inertia
    Inertia combinedInertia;
    combinedInertia.setMass(totalMass);
    combinedInertia.setLocalCOM(totalCOM);
    combinedInertia.setMoment(totalMoment);

    return combinedInertia;
}


std::vector<double> Split2Double(const std::string& input, int num)
{
    std::vector<double> result;
    std::string::size_type sz = 0, nsz = 0;
    for(int i = 0; i < num; i++){
        result.push_back(std::stold(input.substr(sz), &nsz));
        sz += nsz;
    }
    return result;
}

double clamp(double v, double min_v, double max_v)
{
	if(max_v < v)
		return max_v;
	else if(min_v > v)
		return min_v;
	else
		return v;
}

float clamp(float v, float min_v, float max_v)
{
	if(max_v < v)
		return max_v;
	else if(min_v > v)
		return min_v;
	else
		return v;
}

Eigen::Vector3d clamp(Eigen::Vector3d v, double min_v, double max_v)
{
	for(int i=0; i<3; i++)
	{
		if(max_v < v[i])
			v[i] = max_v;
		else if(min_v > v[i])
			v[i] = min_v;
	}	
	
	return v;
}

bool close(double a, double b, double epsilon) {
    return std::abs(a - b) < epsilon;
}

bool compare(std::pair<double, double> a, std::pair<double, double> b)
{
	if(a.first == b.first)
		return true;
	else
		return a.first < b.first;	
}

std::vector<int> sort_indices(const std::vector<double> &val)
{
	std::vector<int> idx(val.size());
	std::iota(idx.begin(), idx.end(), 0);

	std::sort(idx.begin(), idx.end(), [&val](int i1, int i2) { return val[i1] < val[i2]; });

	return idx;
}

double exp_of_squared(const Eigen::VectorXd &vec, double w)
{
	return exp(-w * vec.squaredNorm() / vec.rows());
}

double exp_of_squared(const Eigen::Vector3d &vec, double w)
{
	return exp(-w * vec.squaredNorm() / vec.rows());
}

double exp_of_squared(double val, double w)
{
	return exp(-w * val * val);
}

double exp_of_root(double val, double w)
{
	return exp(-w * sqrt(abs(val)));
}

double exp_of_L1_norm(const Eigen::VectorXd &vec, double w)
{
	return exp(-w * vec.lpNorm<1>());
}

double exp_of_L1_norm(const Eigen::Vector3d &vec, double w)
{
	return exp(-w * vec.lpNorm<1>());
}

double exp_of_L1_norm(double val, double w)
{
	return exp(-w * std::abs(val));
}

void setSkelPos(SkeletonPtr skel, Eigen::VectorXd pos)
{
	if(skel == nullptr){
		std::cout << "Skeleton is nullptr" << std::endl;
		return;
	}
	
	skel->setPositions(pos);
	skel->computeForwardKinematics(true, false, false);
}

void setSkelVel(SkeletonPtr skel, Eigen::VectorXd vel)
{
	if(skel == nullptr){
		std::cout << "Skeleton is nullptr" << std::endl;
		return;
	}
	
	skel->setVelocities(vel);
	skel->computeForwardKinematics(false, true, false);
}

void setSkelPosAndVel(SkeletonPtr skel, Eigen::VectorXd pos, Eigen::VectorXd vel)
{
	if(skel == nullptr){
		std::cout << "Skeleton is nullptr" << std::endl;
		return;
	}

	skel->setPositions(pos);	
	skel->setVelocities(vel);	
	skel->computeForwardKinematics(true, true, false);
}

void DataStandardize(std::vector<double>& toData, std::vector<double>& fromData, int div, bool filter)
{
	toData.clear();
	
	if(fromData.size() == 0)
		return;

	if(div > fromData.size())
		div = fromData.size()-1;

	if(div < 1)
		div = 1;

	double formIntval = 1.0/(double)(fromData.size()-1.0);
	double toInterval = 1.0/(double)div;
	for(int j=0; j<=div; j++){
		double upper = j*toInterval;
		double lower = formIntval;

		unsigned int a = upper/lower;
		double b1 = fmod(upper, lower);
		double b2 = lower - b1;

		a = a < fromData.size() ? a : fromData.size()-1; // boundary check for fromdata
		double x1 = fromData[a];
		if(b1 == 0)
			toData.push_back(fromData[a]);
		else{
			a = a+1 < fromData.size() ? a : fromData.size()-2;
			double x2 = fromData[a+1];
			double y = (b1*x2 + b2*x1)/(b1 + b2);
			toData.push_back(y);
		}				
	}
	
	if(filter)
	{
		for(int j=0; j<toData.size(); j++)
		{
			if(j>0)
				toData[j] = 0.1*toData[j] + 0.9*toData[j-1];
		}
	}	
	
	fromData.clear();	
}
}

