#ifndef RECORD_H

#define RECORD_H

#include <filesystem>
#include <map>
#include <queue>
#include <Eigen/Dense>

class Record
{
public:
    explicit Record(const std::vector<std::string>& field_names);
    ~Record();

    void add(unsigned int sim_step, std::queue<std::pair<std::string, double>> data); // Queue[(Field, Value)]
    void reset();
    void save(const std::string& path, const std::string& extension = "csv") const;
    unsigned int get_nrow() const { return nrow; }
    unsigned int get_ncol() const { return ncol; }
    static void save(const std::vector<Record>& records, const std::vector<std::unordered_map<std::string, float>>& params,
                     const std::string& path, const std::string& extension = "csv");
    static void save(const std::vector<Record*>& pRecords, const std::vector<std::unordered_map<std::string, float>>& params,
                     const std::string& path, const std::string& extension = "csv");
    static std::vector<std::string> apply_prefix(const std::string& prefix, const std::vector<std::string>& suffixes);
    static std::vector<std::string>& cat_vec(std::vector<std::string>& v1, const std::vector<std::string>& v2);
protected:
    void resize_if_needed(unsigned int requested_size);
    void print_csv_head(std::ofstream& ofs) const;
    void print_csv_body(std::ofstream& ofs, int param_idx) const;

    Eigen::MatrixXd _data;  // row index means the simulation step
    const std::unordered_map<std::string, int> field_to_colidx;
    const std::vector<std::string> field_names;
    const unsigned int ncol;
    unsigned int nrow; // maximun value of added rows (simulation steps)
    const unsigned int data_chuck;
};

#endif //RECORD_H
