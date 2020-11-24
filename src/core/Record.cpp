#include "core/Record.h"
#include "util/path.h"
#include <iostream>
#include <map>
#include <fstream>


using namespace std;
using namespace Eigen;


Record::Record(const vector<string>& field_names):
field_to_colidx([&field_names]()
    {
        unordered_map<string, int> _map;
        for (size_t i = 0; i < field_names.size(); i++)
        {
            _map.emplace(field_names[i], i);
        }
        return _map;
    }()),
field_names(field_names),
ncol(field_names.size()), nrow(0),
data_chuck(100)
{
    assert(field_names[0] == "step");
    _data = MatrixXd::Zero(data_chuck, ncol);
    _data.col(0) = VectorXd::LinSpaced(data_chuck, 0, data_chuck - 1); // filling the step column
}

Record::~Record() = default;

void Record::resize_if_needed(const unsigned int requested_size)
{
    if(const auto prev_row = _data.rows(); requested_size > prev_row)
    {
        const auto nchuck = (static_cast<long>(requested_size) - prev_row) / data_chuck + 1;
        try
        {
            _data.conservativeResize(prev_row + nchuck * data_chuck, NoChange);
            _data.block(prev_row, 0, nchuck * data_chuck, 1) = // filling the step column
                VectorXd::LinSpaced(nchuck * data_chuck, prev_row, prev_row + nchuck * data_chuck - 1);
        }
        catch (const exception& e)
        {
            cerr << "[Error::Record] while resizing, " << e.what() << endl;
        }
    }
}

void Record::add(const unsigned int sim_step, queue<pair<string, double>> data)
{
    resize_if_needed(sim_step + 1);
    nrow = max(nrow, sim_step + 1);
    while (!data.empty())
    {
        const auto& [field, value] = data.front();
        if (const auto it = field_to_colidx.find(field); it != field_to_colidx.end())
        {
            _data(sim_step, it->second) = value;
        }
        else
        {
#ifdef LOG_VERBOSE
            cerr << "[Warning::Record] adding non-existant field: " << field << " to record" << endl;
#endif
        }
        data.pop();
    }
}

void Record::reset()
{
    _data = MatrixXd::Zero(data_chuck, ncol);
    nrow = 0;
}

void Record::save(const string& path, const string& extension) const
{
    assert(extension == "csv");
    const filesystem::path filepath = path + "." + extension;

    ofstream ofs(filepath);
    if (!ofs)
    {
        cerr << "Error: Cannot open file: " << filepath << endl;
        return;
    }

    // print header for csv
    if (!field_names.empty())
    {
        auto it = field_names.begin();
        ofs << *it; ++it; // Print the first element without a comma
        for (; it != field_names.end(); ++it)
        {
            ofs << "," << *it;
        }
        ofs << endl;
    }
    // print data for csv
    for (int row=0; row<nrow; row++)
    {
        if (ncol > 0)
        {
            int col = 0;
            ofs << _data(row, col); ++col;
            for (; col<ncol; col++)
            {
                ofs << "," << _data(row, col);
            }
            ofs << endl;
        }
    }
    ofs.close();
}

void Record::print_csv_head(std::ofstream& ofs) const{
    if(!ofs){
        cerr << "Error: Cannot open file in print_csv_head" << endl;
        return;
    }
    ofs << "param_idx";
    for (const auto& field : field_names)
    {
        ofs << "," << field;
    }
    ofs << endl;
}

void Record::print_csv_body(std::ofstream& ofs, const int param_idx) const{
    if(!ofs){
        cerr << "Error: Cannot open file in print_csv_body idx: " << param_idx << endl;
        return;
    }
for (int row = 0; row < nrow; row++)
{
    ofs << param_idx;
    for (int col = 0; col < ncol; col++)
    {
        ofs << "," << std::fixed << std::setprecision(4) << _data(row, col);
    }
    ofs << endl;
}}

void Record::save(const std::vector<Record*>& pRecords, const std::vector<std::unordered_map<std::string, float>>& params,
    const std::string& path, const std::string& extension)
{
    vector<Record> records;
    for(const auto& pRecord: pRecords){
        records.emplace_back(*pRecord);
    }
    Record::save(records, params, path, extension);
}


void Record::save(const vector<Record>& records, const vector<unordered_map<std::string, float>>& params, const string& path,
    const string& extension)
{
    assert(extension == "csv");
    const filesystem::path record_path = path + "." + extension;

    ofstream record_f(record_path);
    if (!record_f)
    {
        cerr << "Error: Cannot open file: " << record_path << endl;
        return;
    }
    // print out the records
    records[0].print_csv_head(record_f);
    for(int i=0; i < records.size(); i++){
        records[i].print_csv_body(record_f, i);
    }
    record_f.close();
}

vector<string> Record::apply_prefix(const string& prefix, const vector<string>& suffixes) {
    vector<string> result;
    for (const auto& suffix : suffixes) {
        result.emplace_back(prefix + string("_").append(suffix));
    }
    return result;
}

vector<string>& Record::cat_vec(vector<string>& v1, const vector<string>& v2) {
    v1.insert(v1.end(), v2.begin(), v2.end());
    return v1;
}

