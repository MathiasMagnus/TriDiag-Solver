#include <Validate.hpp>

// TriDiagSolver includes
//#include <TriDiagSolver.hpp>

// STL includes
#include <cstddef>      // std::size_t
#include <iostream>     // std::cout, std::cerr
#include <stdexcept>    // std::runtime_error
#include <random>       // std::default_random_engine

int main(int argc, char** argv)
{
    try
    {
        const cli::options opts = cli::parse(argc, argv);

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

        using real = float;
        using solver_internal = float;

        tridiag_solver<real, solver_internal> solver{ queue };

        auto prng = [engine = std::default_random_engine{},
                     dist = std::uniform_real_distribution<real>{ -1, 1 }]() mutable { return dist(engine); };
        std::vector<real> d, du, dl, b(opts.length);
        std::generate_n(std::back_inserter(d),  opts.length, prng);
        std::generate_n(std::back_inserter(du), opts.length/* - 1*/, prng);
        std::generate_n(std::back_inserter(dl), opts.length/* - 1*/, prng);

        cl::Buffer d_buf(context, d.begin(), d.end(), false),    // false = read_only
                   du_buf(context, du.begin(), du.end(), false), // false = read_only
                   dl_buf(context, dl.begin(), dl.end(), false), // false = read_only
                   b_buf(context, CL_MEM_READ_WRITE, opts.length * sizeof(real));

        solver.gtsv_spike_partial_diag_pivot(dl_buf, d_buf, du_buf, b_buf).wait();

        cl::copy(queue, b_buf, b.begin(), b.end());
    }
    catch (TCLAP::ArgException & e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
        std::exit(EXIT_FAILURE);
    }

    return 0;
}