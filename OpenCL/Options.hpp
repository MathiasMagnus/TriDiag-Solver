// OpenCL includes
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 100
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/cl2.hpp>   // cl_device_type

// STL includes
#include <cstddef>      // std::size_t
#include <string>       // std::string


namespace cli
{
    struct options
    {
        std::size_t length, plat_id, dev_id;
        cl_device_type dev_type;
        std::string input, output, validate;
        bool quiet;
    };

    options parse(int argc, char** argv, const std::string banner);
}
