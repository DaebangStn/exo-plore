#include "util/path.h"
#include <chrono>
#include <iomanip>


using namespace std;


filesystem::path path_rel_to_abs(const char* rel_path)
{
    return PROJECT_ROOT_PATH / rel_path;
}

filesystem::path path_rel_to_abs(const string &rel_path)
{
    return PROJECT_ROOT_PATH / rel_path;
}

std::filesystem::path path_rel_to_abs(const std::filesystem::path &rel_path)
{
    return PROJECT_ROOT_PATH / rel_path;
}

string path_obj(const string &obj_name)
{
    return (OBJ_ASSET_DIR / obj_name).string();
}

string default_csv_path()
{
    return timestamp("%m%d_%H%M%S") + ".csv";
}

string timestamp(const string& fmt)
{
    const auto now = chrono::system_clock::now();
    const time_t now_c = chrono::system_clock::to_time_t(now);
    ostringstream oss;
    oss << put_time(localtime(&now_c), fmt.c_str());
    return oss.str();
}