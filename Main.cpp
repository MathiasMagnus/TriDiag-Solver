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

        if (device.getInfo<CL_DEVICE_EXTENSIONS>().find("cl_khr_global_int32_base_atomics") == std::string::npos) throw std::runtime_error{ "Selected device does not support double precision" };

        std::vector<cl_context_properties> props{ CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platform()), 0 };
        cl::Context context{ device, props.data() };

        cl::CommandQueue command_queue{ context, device, CL_QUEUE_PROFILING_ENABLE };

        std::ifstream source_file("./Peak.cl");
        if (!source_file.is_open()) throw std::runtime_error("Cannot open ./Peak.cl");

        std::string source_string{ std::istreambuf_iterator<char>{source_file}, std::istreambuf_iterator<char>{} };

        cl::Program program{ context, source_string };

        std::stringstream build_opts;
        build_opts <<
            "-cl-mad-enable " <<
            "-cl-no-signed-zeros " <<
            "-cl-finite-math-only " <<
            "-cl-single-precision-constant ";

        if (!quiet_arg.getValue()) { std::cout << "Building program..."; std::cout.flush(); }
        program.build({ device }, build_opts.str().c_str());
        if (!quiet_arg.getValue()) { std::cout << " done." << std::endl; }

        interaction = cl::Kernel(program, "interaction");
        forward_euler = cl::Kernel(program, "forward_euler");

        if (!quiet_arg.getValue())
            std::cout << "Interaction kernel preferred WGS: " << interaction.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device) << std::endl;
        if (!quiet_arg.getValue())
            std::cout << "Forward Euler kernel preferred WGS: " << forward_euler.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device) << std::endl;

        if (!quiet_arg.getValue()) { std::cout << "Reading input file..."; std::cout.flush(); }
        particles = read_particle_file(input_arg.getValue());
        if (!quiet_arg.getValue()) { std::cout << " done." << std::endl; }

        buffer = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, particles.size() * sizeof(particle), particles.data());

        interaction.setArg(0, buffer);
        forward_euler.setArg(0, buffer);
        forward_euler.setArg(1, 0.001f);

        interaction_gws = cl::NDRange(particles.size());
        interaction_lws = cl::NullRange;
        euler_gws = cl::NDRange(particles.size());
        euler_lws = cl::NullRange;

        // Run warm-up kernels
        command_queue.enqueueNDRangeKernel(interaction, cl::NullRange, interaction_gws, interaction_lws, nullptr, &interaction_event);
        command_queue.enqueueNDRangeKernel(forward_euler, cl::NullRange, euler_gws, euler_lws);

        command_queue.finish();

        // Reset data
        command_queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, particles.size() * sizeof(particle), particles.data());
    }
    catch (TCLAP::ArgException& e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
        std::exit(EXIT_FAILURE);
    }
    catch (cl::Error error)
    {
        std::cerr << error.what() << "(" << error.err() << ")" << std::endl;

        if (std::string(error.what()) == "clBuildProgram")
        {
            if (program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device) == CL_BUILD_ERROR)
                std::cerr << std::endl << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        }

        exit(error.err());
    }

    return 0;
}