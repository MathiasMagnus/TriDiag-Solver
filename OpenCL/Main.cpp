#include <TriDiagSolver.hpp>

// TCLAP includes
#include <tclap/CmdLine.h>

// OpenCL includes
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 100
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/cl2.hpp>

// STL includes
#include <string>       // std::string
#include <cstddef>      // std::size_t
#include <iostream>     // std::cout, std::cerr
#include <stdexcept>    // std::runtime_error
#include <random>       // std::default_random_engine
#include <utility>      // std::pair


using real = float;
using solver_internal = float;
enum array
{
    d = 0,
    du = 1,
    dl = 2,
    b = 3
};

int main(int argc, char** argv)
{
    try
    {
        std::string banner = "OpenCL-TriDiagSolver sample";

        TCLAP::CmdLine cli(banner);

        TCLAP::ValueArg<std::string> input_arg("i", "input", "Path to input file", false, "./", "path");
        TCLAP::ValueArg<std::string> output_arg("o", "output", "Path to output file", false, "", "path");
        TCLAP::ValueArg<std::string> validate_arg("v", "validate", "Path to validation file", false, "", "path");
        TCLAP::ValueArg<std::size_t> length_arg("l", "length", "Length of input", false, 262144, "positive integral");
        TCLAP::ValueArg<std::size_t> platform_arg("p", "platform", "Index of platform to use", false, 0, "positive integral");
        TCLAP::ValueArg<std::size_t> device_arg("d", "device", "Number of input points", false, 0, "positive integral");
        TCLAP::ValueArg<std::string> type_arg("t", "type", "Type of device to use", false, "default", "[cpu|gpu|acc]");
        TCLAP::SwitchArg quiet_arg("q", "quiet", "Suppress standard output", false);

        cli.add(input_arg);
        cli.add(output_arg);
        cli.add(validate_arg);
        cli.add(length_arg);
        cli.add(platform_arg);
        cli.add(device_arg);
        cli.add(type_arg);
        cli.add(quiet_arg);

        cli.parse(argc, argv);

        if (!quiet_arg.getValue()) std::cout << banner << std::endl << std::endl;

        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        if (platforms.empty()) throw std::runtime_error{ "No OpenCL platform detected." };

        cl::Platform platform = platforms.at(platform_arg.getValue());

        if (!quiet_arg.getValue()) std::cout << "Selected platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;

        cl_device_type device_type = CL_DEVICE_TYPE_DEFAULT;
        if (type_arg.getValue() == "cpu") device_type = CL_DEVICE_TYPE_CPU;
        if (type_arg.getValue() == "gpu") device_type = CL_DEVICE_TYPE_GPU;
        if (type_arg.getValue() == "acc") device_type = CL_DEVICE_TYPE_ACCELERATOR;

        std::vector<cl::Device> devices;
        platform.getDevices(device_type, &devices);

        cl::Device device = devices.at(device_arg.getValue());

        if (!quiet_arg.getValue()) std::cout << "Selected device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

        if (device.getInfo<CL_DEVICE_EXTENSIONS>().find("cl_khr_global_int32_base_atomics") == std::string::npos) throw std::runtime_error{ "Selected device does not support cl_khr_global_int32_base_atomics" };

        std::vector<cl_context_properties> props{ CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platform()), 0 };
        cl::Context context{ device, props.data() };

        cl::CommandQueue queue{ context, device, CL_QUEUE_PROFILING_ENABLE };

        tridiag_solver<real, solver_internal> solver{ queue };

        auto prng = [engine = std::default_random_engine{},
                     dist = std::uniform_real_distribution<real>{ -1, 1 }]() mutable { return dist(engine); };
        std::array<std::vector<real>, 4> arrays;
        for (auto& arr : arrays) std::generate_n(std::back_inserter(arr), length_arg.getValue(), prng);

        arrays[dl].at(0) = 0;
        arrays[du].at(length_arg.getValue() - 1) = 0;

        std::array<cl::Buffer, 4> buffers;
        std::transform(arrays.begin(), arrays.end(),
                       buffers.begin(),
                       [&](std::vector<real> & arr)
        {
                return cl::Buffer{ context, arr.begin(), arr.end(), false }; // false = read_only
        });

        solver.gtsv_spike_partial_diag_pivot(buffers[dl], buffers[d], buffers[du], buffers[b]).wait();

        cl::copy(queue, buffers[b], arrays[b].begin(), arrays[b].end());
    }
    catch (TCLAP::ArgException& e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
        std::exit(EXIT_FAILURE);
    }
    catch (cl::BuildError e)
    {
        std::cerr << e.what() << "(" << e.err() << ")" << std::endl;
        for (auto& log : e.getBuildLog())
        {
            std::cerr << "Log for " << log.first.getInfo<CL_DEVICE_NAME>() << std::endl;
            std::cerr << log.second << std::endl;
        }
        std::exit(e.err());
    }
    catch (cl::Error e)
    {
        std::cerr << e.what() << "(" << e.err() << ")" << std::endl;
        std::exit(e.err());
    }
    catch (std::exception e)
    {
        std::cerr << e.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }

    return 0;
}