#include "util/struct.h"
#include <boost/program_options.hpp>


ExpArg export_arguments(int argc, char* argv[]) {
    namespace po = boost::program_options;
    ExpArg args;

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message")
        ("input_csv,i", po::value<std::string>(&args.input_csv)->required(), "input CSV file")
        ("torchscript_dir,s", po::value<std::string>(&args.torchscript_dir)->required(), "TorchScript directory")
        ("record_config,c", po::value<std::string>(&args.record_config)->required(), "record config file")
        ("output_dir,o", po::value<std::string>(&args.output_dir)->required(), "output CSV file")
        ("render,r", po::bool_switch(&args.render), "render the sampling process (#worker=1)");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        exit(0); // Exit after printing help
    }

    try {
        po::notify(vm);
    }
    catch (const po::error &e) {
        std::cerr << "Error: " << e.what() << "\n";
        std::cerr << desc << "\n";
        exit(1);
    }
    std::cout << "Input CSV file: " << args.input_csv << "\n";
    std::cout << "Model from: " << args.torchscript_dir << "\n";
    std::cout << "Record config file: " << args.record_config << "\n";
    std::cout << "Output directory: " << args.output_dir << "\n";
    return args;
}


RenderArg render_arguments(int argc, char* argv[]) {
    namespace po = boost::program_options;
    RenderArg args;

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message")
        ("torchscript_dir,s", po::value<std::string>(&args.torchscript_dir)->required(), "TorchScript directory")
        ("force_device,d", po::bool_switch(&args.force_device), "render the sampling process (#worker=1)");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        exit(0); // Exit after printing help
    }

    try {
        po::notify(vm);
    }
    catch (const po::error &e) {
        std::cerr << "Error: " << e.what() << "\n";
        std::cerr << desc << "\n";
        exit(1);
    }
    std::cout << "Model from: " << args.torchscript_dir << "\n";
    return args;
}

MuscleLengthArg muscle_length_arguments(int argc, char* argv[]) {
    namespace po = boost::program_options;
    MuscleLengthArg args;

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message")
        ("metadata_path,m", po::value<std::string>(&args.metadata)->required(), "metadata path")
        ("output_path,o", po::value<std::string>(&args.output)->required(), "output path");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        exit(0); // Exit after printing help
    }

    try {
        po::notify(vm);
    }
    catch (const po::error &e) {
        std::cerr << "Error: " << e.what() << "\n";
        std::cerr << desc << "\n";
        exit(1);
    }
    std::cout << "Metadata path: " << args.metadata << "\n";
    std::cout << "Output path: " << args.output << "\n";
    return args;
}


torch::Tensor eigen_vec_to_torch(const Eigen::VectorXd& e)
{
    Eigen::VectorXf e_f = e.cast<float>();
    torch::Tensor t = torch::from_blob(e_f.data(), {e_f.size()}, torch::kFloat).clone(); // Clone to own the data
    t.requires_grad_(false);
    return t;
}

torch::Tensor eigen_vec_to_torch(Eigen::VectorXf e)
{
    torch::Tensor t = torch::from_blob(e.data(), {e.size()}, torch::kFloat).clone(); // Clone to own the data
    t.requires_grad_(false);
    return t;
}

torch::Tensor eigen_mat_to_torch(const Eigen::MatrixXd& e)
{
    Eigen::MatrixXf ef = e.cast<float>();
    auto t = torch::from_blob(ef.data(), {e.rows(), e.cols()}, torch::TensorOptions().dtype(torch::kFloat));
    return t.clone().to(torch::kCUDA, /*non_blocking=*/true).transpose(0, 1).contiguous();
}

torch::Tensor eigen_mat_to_torch_f(Eigen::MatrixXf e)
{
    auto t = torch::from_blob(e.data(), {e.rows(), e.cols()}, torch::TensorOptions().dtype(torch::kFloat));
    return t.clone().to(torch::kCUDA, /*non_blocking=*/true).transpose(0, 1).contiguous();
}

Eigen::VectorXd torch_to_eigen_vec(const torch::Tensor& t)
{
    torch::Tensor t_cpu = t.to(torch::kCPU).contiguous();
    if (t_cpu.dtype() != torch::kDouble) t_cpu = t_cpu.to(torch::kDouble);
    Eigen::Map<Eigen::VectorXd> ef(t_cpu.data_ptr<double>(), t_cpu.size(0));
    return ef;
}