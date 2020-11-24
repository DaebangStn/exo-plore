#ifndef PATH_H
#define PATH_H

#include <filesystem>

const std::string PROJECT_ROOT_DIR = MASS_ROOT_DIR;
const std::filesystem::path PROJECT_ROOT_PATH = std::filesystem::path(PROJECT_ROOT_DIR);
const std::filesystem::path OBJ_ASSET_DIR = PROJECT_ROOT_PATH / "data/OBJ";

std::filesystem::path path_rel_to_abs(const char* rel_path);
std::filesystem::path path_rel_to_abs(const std::string &rel_path);
std::filesystem::path path_rel_to_abs(const std::filesystem::path &rel_path);
std::string path_obj(const std::string &obj_name);
std::string default_csv_path();
std::string timestamp(const std::string& fmt);

#endif //PATH_H
