// This is the sampling code which uses the class render/GLFWApp.h
// You can find the version which uses class Rollout of this code in src/render/export2.cpp

#include "core/Environment.h"
#include "core/Record.h"
#include "render/GLFWApp.h"
#include "util/struct.h"
#include "util/path.h"
#include "indicators/block_progress_bar.hpp"
#include "indicators/cursor_control.hpp"
#include <yaml-cpp/yaml.h>
#include <chrono>


using namespace std;
using namespace indicators;


void worker(const ExpArg& args, BlockProgressBar& bar, atomic<int> &counter, vector<unordered_map<string, float>> &work_items,
            mutex& input_mtx)
{
    const auto config_path = path_rel_to_abs(args.record_config);
    bool force_exo = false;
    bool do_modulate = true;
    int cycle = 1;
    if (!filesystem::exists(config_path)) {
        std::cerr << "Record config file not found: " << config_path << std::endl;
        return;
    }
    try {
        YAML::Node config = YAML::LoadFile(config_path);

        // Ensure the 'record' node exists and is a map
        if (!config["sample"] || !config["sample"].IsMap()) {
            std::cerr << "Invalid or missing 'sample' section in config file: " << config_path << std::endl;
            return;
        }

        YAML::Node sample = config["sample"];
        if (sample["force_exo"]) {
            force_exo = sample["force_exo"].as<bool>();
        }
        if (sample["cycle"]) {
            cycle = sample["cycle"].as<int>();
        }
        if (sample["modulate"]) {
            do_modulate = sample["modulate"].as<bool>();
        }
    } catch (const YAML::BadFile& e) {
        std::cerr << "Failed to open record config file: " << config_path << " - " << e.what() << std::endl;
    }
    catch (const YAML::ParserException& e) {
        std::cerr << "Failed to parse record config file: " << e.what() << std::endl;
    }
    catch (const YAML::Exception& e) {
        std::cerr << "Error processing record config file: " << e.what() << std::endl;
    }


    const auto env = new MASS::Environment(args.render, force_exo);
    vector<Record*> records;
    RenderArg render_args;
    render_args.torchscript_dir = args.torchscript_dir;
    render_args.force_device = args.render;
    MASS::GLFWApp app(render_args);
    // app.setEnv(env); // Removed as it does not exist
    env->LoadRecordConfig(args.record_config);
    bar.set_progress(0.0);
    while(true)
    {
        input_mtx.lock();
        if (counter == work_items.size())
        {
            input_mtx.unlock();
            break;
        }
        const auto& sample = work_items[counter++];
        input_mtx.unlock();
        const auto out_ = app.exportData(sample, cycle, args.render, do_modulate);
        records.insert(records.end(), out_.begin(), out_.end());
        bar.tick();
    }
    delete env;
    Record::save(records, work_items, path_rel_to_abs(args.output_dir) / "data");
}


int main(int argc, char** argv)
{
    show_console_cursor(false);
    const ExpArg args = export_arguments(argc, argv);
    atomic<int> counter(0);
    mutex input_mtx;
    const auto in_csv_abs = path_rel_to_abs(args.input_csv);
    auto work_items = MASS::Utils::ParseParamCsv(in_csv_abs);

    const auto ts_abs = path_rel_to_abs(args.torchscript_dir);
    const auto output_dir_abs = path_rel_to_abs(args.output_dir) /
        (ts_abs.filename().string() + "_" + timestamp("%m%d_%H%M%S"));
    create_directory(output_dir_abs);
    
    try {
        filesystem::copy(in_csv_abs, output_dir_abs / "param.csv", filesystem::copy_options::overwrite_existing);
    } catch (const filesystem::filesystem_error& e) {
        cerr << "Error copying parameter csv file: " << e.what() << endl;
    }
    
    BlockProgressBar bar{
        option::BarWidth{50},
        option::PrefixText{"Exporting data"},
        option::ShowElapsedTime{true},
        option::ShowRemainingTime{true},
        option::MaxProgress{work_items.size()},
    };

    const auto start = chrono::high_resolution_clock::now();
    worker(args, bar, counter, work_items, input_mtx);
    const auto end = chrono::high_resolution_clock::now();
    const auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    const auto seconds = chrono::duration_cast<chrono::seconds>(duration);
    const auto milliseconds = duration - seconds;
    cout << endl << "Elapsed time: " << seconds.count() << "s " << milliseconds.count() << "ms" << endl;
    return 0;
}