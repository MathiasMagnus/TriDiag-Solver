#include <TriDiagSolver.hpp>

#include <Options.hpp>

// TCLAP includes
#include <tclap/CmdLine.h>  // TCLAP::ArgException

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
        const std::string banner = "TriDiagSolver-OpenCL sample";

        const cli::options opts = cli::parse(argc, argv, banner);

        if (!opts.quiet) std::cout << banner << std::endl << std::endl;

        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        if (platforms.empty()) throw std::runtime_error{ "No OpenCL platform detected." };

        cl::Platform platform = platforms.at(opts.plat_id);

        if (!opts.quiet) std::cout << "Selected platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;

        std::vector<cl::Device> devices;
        platform.getDevices(opts.dev_type, &devices);

        cl::Device device = devices.at(opts.dev_id);

        if (!opts.quiet) std::cout << "Selected device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

        if (device.getInfo<CL_DEVICE_EXTENSIONS>().find("cl_khr_global_int32_base_atomics") == std::string::npos) throw std::runtime_error{ "Selected device does not support cl_khr_global_int32_base_atomics" };

        std::vector<cl_context_properties> props{ CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platform()), 0 };
        cl::Context context{ device, props.data() };

        cl::CommandQueue queue{ context, device, CL_QUEUE_PROFILING_ENABLE };

        tridiag_solver<real, solver_internal> solver{ queue };

        auto prng = [engine = std::default_random_engine{},
                     dist = std::uniform_real_distribution<real>{ -1, 1 }]() mutable { return dist(engine); };
        std::array<std::vector<real>, 4> arrays;
        for (auto& arr : arrays) std::generate_n(std::back_inserter(arr), opts.length, prng);

        arrays[dl].at(0) = 0;
        arrays[du].at(opts.length - 1) = 0;

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