#include <Options.hpp>

// TCLAP includes
#include <tclap/CmdLine.h>


cli::options cli::parse(int argc, char** argv, const std::string banner)
{
    TCLAP::CmdLine cli(banner);

    TCLAP::ValueArg<std::size_t> length_arg("l", "length", "Length of input", false, 262144, "positive integral");
    TCLAP::ValueArg<std::size_t> platform_arg("p", "platform", "Index of platform to use", false, 0, "positive integral");
    TCLAP::ValueArg<std::size_t> device_arg("d", "device", "Number of input points", false, 0, "positive integral");
    TCLAP::ValueArg<std::string> type_arg("t", "type", "Type of device to use", false, "default", "[cpu|gpu|acc]");
    TCLAP::ValueArg<std::string> input_arg("i", "input", "Path to input file", false, "./", "path");
    TCLAP::ValueArg<std::string> output_arg("o", "output", "Path to output file", false, "", "path");
    TCLAP::ValueArg<std::string> validate_arg("v", "validate", "Path to validation file", false, "", "path");
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

    auto device_type = [](TCLAP::ValueArg<std::string> & x) -> cl_device_type
    {
        if (x.getValue() == "cpu") return CL_DEVICE_TYPE_CPU;
        if (x.getValue() == "gpu") return CL_DEVICE_TYPE_GPU;
        if (x.getValue() == "acc") return CL_DEVICE_TYPE_ACCELERATOR;
        return CL_DEVICE_TYPE_DEFAULT;
    };

    return { length_arg.getValue(), platform_arg.getValue(), device_arg.getValue(),
            device_type(type_arg), input_arg.getValue(), output_arg.getValue(), validate_arg.getValue(),
            quiet_arg.getValue() };
}