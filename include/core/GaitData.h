#pragma once

#include <experimental/filesystem>
#include <map>
#include <string>
#include <iostream>
#include <utility>
namespace fs = std::experimental::filesystem;

enum GaitDataType {
    PerSimFrame, PerSimFrameVec, PerConFrame, PerCycle, ParamList
};

struct GaitDataField {
    std::string name;
    std::vector<double> data;
    std::vector<std::vector<double>> data_vec;

    GaitDataField() = default;
    explicit GaitDataField(std::string name) : name(std::move(name)) {}

    const double& operator[](int i) const {
        return data[i];
    }

    double& operator[](int i) {
        return data[i];
    }

    void add(double v) { data.push_back(v); }
    void add(const std::vector<double>& v) { data_vec.push_back(v); }
    void merge(const GaitDataField& other) {
        data.insert(data.end(), other.data.begin(), other.data.end());
        data_vec.insert(data_vec.end(), other.data_vec.begin(), other.data_vec.end());
    }
};

struct GaitDataCategory {
    std::string name;
    GaitDataType type=PerCycle;
    std::vector<GaitDataField> fields;
    std::map<std::string, int> field_indices;

    GaitDataCategory() = default;
    GaitDataCategory(std::string name, GaitDataType type) : name(std::move(name)), type(type) {}

    int num_columns() const { return fields.size(); }
    int num_rows() const { return fields.empty() ? 0 : fields[0].data.size(); }

    const GaitDataField& operator[](const std::string& name) const {
        return fields[field_indices.at(name)];
    }

    GaitDataField& operator[](const std::string& name) {
        return fields[field_indices.at(name)];
    }

    void print_field_indices(){
        for(const auto entry : field_indices){
            std::cout << "key : " << entry.first << " idx : " << entry.second << std::endl;
        }
    }

    GaitDataCategory& add_field(std::string name) {
        int field_idx = fields.size();
        field_indices[name] = field_idx;
        fields.push_back(GaitDataField(name));
        return *this;
    }

    GaitDataCategory& add_data(const std::string& field_name, double value) {
        fields[field_indices[field_name]].add(value);
        return *this;
    }
    GaitDataCategory& add_data(const std::string& field_name, std::vector<double> value) {
        fields[field_indices[field_name]].add(value);
        return *this;
    }
    GaitDataCategory& add_data(int i, double value) {
        fields[i].add(value);
        return *this;
    }    

    void merge(const GaitDataCategory& other) {
        if (type != other.type) {
            throw std::invalid_argument("Cannot merge GaitDataCategory of different types.");
        }

        for (const auto& field : other.fields) {
            if (field_indices.find(field.name) != field_indices.end()) {
                fields[field_indices[field.name]].merge(field);
            } else {
                add_field(field.name).fields.back().merge(field);
            }
        }
    }

    int entry_count() const 
    {
        int n_entries = fields[0].data.size();
        for (auto& field : fields) {
            if (n_entries != field.data.size()) {
                std::cout << "Invalid entry count in category " << name << "!" << std::endl;
                exit(EXIT_FAILURE);
            }
        }
        return n_entries;
    }

    int entry_vec_count() const 
    {
        int n_entries = fields[0].data_vec.size();
        for(auto& field : fields) {
            if (n_entries != field.data_vec.size()) {
                std::cout << "Invalid entry vec count in category " << name << "!" << std::endl;
                exit(EXIT_FAILURE);
            }
        }
        return n_entries;
    }

    GaitDataCategory& convert_to_con_frame(int sims_per_con = 16) 
    {
        if (type != PerSimFrame) {
            std::cout << "Can only convert PerSimFrame to PerConFrame!" << std::endl;
            exit(EXIT_FAILURE);
        }
        for (int fidx = 0; fidx < fields.size(); fidx++) {
            auto& field = fields[fidx];
            int new_size = field.data.size() / sims_per_con;
            GaitDataField new_field;
            new_field.name = field.name;
            new_field.data.resize(new_size, 0);
            for (int i = 0; i < new_size; i++) {
                for (int k = 0; k < sims_per_con; k++) {
                    new_field.data[i] += field.data[16*i + k];
                }
                new_field.data[i] /= sims_per_con;
            }
            field = new_field;
        }
        return *this;
    }

    void import_csv(const std::string& filename) 
    {        
        std::ifstream ifs(filename);

        std::string token;
        std::string line;

        std::getline(ifs, line);
        std::istringstream ss(line);

        std::getline(ss, token, ',');
        if (token == "cycle") type = PerCycle;
        else if (token == "con_frame") type = PerConFrame;
        else if (token == "sim_frame") type = PerSimFrame;
        else if (token == "work_item") type = ParamList;

        std::string field_name;
        while (std::getline(ss, field_name, ',')) {
            field_indices[field_name] = fields.size();
            fields.emplace_back(field_name);
        }

        int rows = 0;
        while (std::getline(ifs, line)) {
            std::istringstream ss(line);
            std::getline(ss, token, ',');
            int i = 0;
            while (std::getline(ss, token, ',')) {
                fields[i].data.push_back(std::stod(token));
                i++;
            }
            rows++;
        }

        for (auto& field : fields) {
            assert(rows == field.data.size());
        }
    }

    void export_csv(const std::string& filename) const 
    {
        // There are many ways to sample in the commit 4c79c898, only PerSimFrameVec and PerCycle are supported
        std::ofstream ofs(filename);
        switch (type) {
            case PerSimFrameVec: ofs << "sim_frame"; break;
            case PerCycle: ofs << "cycle"; break;
        }

        std::map<std::string, std::vector<std::vector<double>>> field_map;        
        for(auto& field : fields) 
        {
            ofs << ',' << field.name;
            field_map[field.name] = std::vector<std::vector<double>>();
            if(type != PerSimFrameVec && field_map[field.name].empty())
                field_map[field.name].emplace_back();
        }
        
        ofs << '\n';
        if(type == PerSimFrameVec)
        {
            for (auto& field : fields) 
            {
                int n_entries = entry_vec_count();                                       
                for (int i=0; i<n_entries; i++) {
                    std::vector<double> cur = field.data_vec[i];
                    field_map[field.name].push_back(cur);
                }
            }            
        }
        else{
            int n_entries = entry_count();
            for (int i = 0; i < n_entries; i++) {
                for (auto& field : fields) {
                    ofs << ',' << field.data[i];
                    field_map[field.name][0].push_back((double)field.data[i]);
                }
                ofs << '\n';
            }
        }
        if(type == PerSimFrameVec){
            int cycleNum = entry_vec_count();
            for(int i=0; i<cycleNum; i++)
            {
                int num_rows = 201;
                for(auto& field: fields)
                {
                    int n = field_map[field.name][i].size();
                    num_rows = std::min(num_rows, n);
                }
                std::cout << "[Export]" << " size: " << num_rows << std::endl;
                // if (num_rows != 201)
                // {
                //     continue;
                // }
                for(int j=0; j<num_rows; j++)
                {
                    ofs << i+1;
                    ofs << ',';
                    ofs << j+1;
                    for (auto& field : fields){
                        ofs << ',' << field_map[field.name][i][j];
                    }
                    ofs << '\n';
                }

                // Original code
                // for(int j=0; j<201; j++)
                // {
                //     ofs << i+1;
                //     ofs << ',';
                //     ofs << j+1;
                //     for (auto& field : fields){
                //         ofs << ',' << field_map[field.name][i][j];
                //     }
                //     ofs << '\n';
                // }
            }
        }
    }
};

struct GaitData {
    std::map<std::string, GaitDataCategory> categories;

    GaitDataCategory& add_category(std::string name, GaitDataType type) {
        categories.insert({name, GaitDataCategory(name, type)});
        return categories[name];
    }

    const GaitDataCategory& operator[](const std::string& category_name) const {
        return categories.at(category_name);
    }

    GaitDataCategory& operator[](const std::string& category_name) {
        return categories.at(category_name);
    }

    // + operator to merge two GaitData structs
    GaitData operator+(const GaitData& other) const {
        GaitData result = *this; // Start with a copy of this GaitData

        for (const auto& [name, category] : other.categories) {
            if (result.categories.find(name) != result.categories.end()) {
                result.categories[name].merge(category); // Merge categories with the same name
            } else {
                result.categories[name] = category; // Add new categories
            }
        }

        return result;
    }

    // Optional: += operator for in-place merging
    GaitData& operator+=(const GaitData& other) {
        for (const auto& [name, category] : other.categories) {
            if (categories.find(name) != categories.end()) {
                categories[name].merge(category); // Merge categories with the same name
            } else {
                categories[name] = category; // Add new categories
            }
        }
        return *this;
    }

    void export_csv(const std::string& save_pathname) 
    {
        fs::path save_path(save_pathname);
        fs::create_directory(save_path);
        for (auto& [_, cat] : categories) {
            cat.export_csv(save_path / (cat.name + ".csv"));            
        }
    }
};

