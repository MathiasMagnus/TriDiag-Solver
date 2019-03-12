// TCLAP includes
#include <tclap/CmdLine.h>

// OpenCL includes
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 100
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/cl2.hpp>

// Standard C++ includes
#include <algorithm>
#include <complex>
#include <numeric>
#include <valarray>


namespace cli
{
    struct options
    {
        std::size_t length, plat_id, dev_id;
        cl_device_type dev_type;
        std::string input, output, validate;
        bool quiet;
    };

    options parse(int argc, char** argv)
    {
        std::string banner = "OpenCL-TriDiagSolver validator";

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

        auto device_type = [](TCLAP::ValueArg<std::string>& x) -> cl_device_type
        {
            if (x.getValue() == "cpu") return CL_DEVICE_TYPE_CPU;
            if (x.getValue() == "gpu") return CL_DEVICE_TYPE_GPU;
            if (x.getValue() == "acc") return CL_DEVICE_TYPE_ACCELERATOR;
            return CL_DEVICE_TYPE_DEFAULT;
        };

        return {length_arg.getValue(), platform_arg.getValue(), device_arg.getValue(),
                device_type(type_arg), input_arg.getValue(), output_arg.getValue(), validate_arg.getValue(),
                quiet_arg.getValue()};
    }
}


template <typename T>
bool validate(std::valarray<T> x,
              std::valarray<T> y,
              double threshold)
{
    auto ref_integral = std::norm((x).sum());
    auto diff_integral = std::norm((x - y).sum());

    return diff_integral / ref_integral < threshold;
}

template <typename T>
bool validate(cl::CommandQueue q,
              cl::Buffer x,
              T* y,
              double threshold)
{
    auto count = x.getInfo<CL_MEM_SIZE>() / sizeof(T);

    std::valarray<T> x_val(count),
                     y_val(count);

    cudaThreadSynchronize();
    
    cl::copy(q, x, std::begin(x_val), std::end(x_val));
    cudaError_t err = cudaMemcpy(&(*std::begin(y_val)), y, count, cudaMemcpyDeviceToHost);

    return validate(x_val, y_val, threshold);
}

// STL includes
#include <string>       // std::string
#include <fstream>      // std::ifstream
#include <stdexcept>    // std::runtime_error
#include <iterator>     // std::istreambuf_iterator
#include <sstream>      // std::stringstream
#include <vector>       // std::vector
#include <algorithm>    // std::max
#include <complex>

#include <cuda_runtime.h>

#include "spike_kernel.hxx"

template <typename T, typename TT>
class tridiag_solver
{
public:

    tridiag_solver(cl::CommandQueue queue);
    
    cl::Event gtsv_spike_partial_diag_pivot(std::vector<T>& dl, std::vector<T>& d, std::vector<T>& du, std::vector<T>& b);
    cl::Event gtsv_spike_partial_diag_pivot(cl::Buffer dl, cl::Buffer d, cl::Buffer du, cl::Buffer b, std::vector<cl::Event> wait = {});

private:

    cl::CommandQueue queue_;

    cl::Kernel foward_marshaling_bxb_kernel,
               tiled_diag_pivot_x1_kernel,
               spike_local_reduction_x1_kernel,
               spike_global_solving_x1_kernel,
               spike_local_solving_x1_kernel,
               spike_back_sub_x1_kernel,
               back_marshaling_bxb_kernel;

    cl::size_type m_pad, b_dim, s, stride,
                  last_m;

    cl::Buffer flag, dl_buffer, d_buffer, du_buffer, b_buffer,
               w_buffer, v_buffer, c2_buffer,
               x_level_2, w_level_2, v_level_2;

    std::vector<cl::Event> forward_events,
                           pivot_event,
                           local_reduction_event,
                           global_solving_event,
                           local_solving_event,
                           back_sub_event,
                           back_marshal_event;

    bool* c_flag;
    T *c_dl_buffer, *c_d_buffer, *c_du_buffer, *c_b_buffer,
      *c_w_buffer, *c_v_buffer, *c_c2_buffer,
      *c_x_level_2, *c_w_level_2, *c_v_level_2;

    void findBestGrid(cl::size_type m, cl::size_type tile_marshal);

    std::string host_type_to_cl_name(float) { return "float"; }
    std::string host_type_to_cl_name(double) { return "double"; }
    std::string host_type_to_cl_name(std::complex<float>) { return "float2"; }
    std::string host_type_to_cl_name(std::complex<double>) { return "double2"; }
};


template <typename T, typename TT>
tridiag_solver<T, TT>::tridiag_solver(cl::CommandQueue queue)
    : queue_(queue)
    , last_m(0)
    , forward_events(4)
    , pivot_event(1)
    , local_reduction_event(1)
    , global_solving_event(1)
    , local_solving_event(1)
    , back_sub_event(1)
    , back_marshal_event(1)
{
    std::ifstream source_file("./TriDiagSolver.cl");
    if (!source_file.is_open()) throw std::runtime_error("Cannot open ./TriDiagSolver.cl");

    std::string source_string{ std::istreambuf_iterator<char>{source_file}, std::istreambuf_iterator<char>{} };

    cl::Program program{ queue.getInfo<CL_QUEUE_CONTEXT>(), source_string };

    std::stringstream build_opts;
    build_opts <<
        "-cl-mad-enable " <<
        "-cl-no-signed-zeros " <<
        "-cl-finite-math-only " <<
        "-cl-single-precision-constant " <<
        "-Delem=" << host_type_to_cl_name(T{}) << " " <<
        "-Dreal=" << host_type_to_cl_name(TT{}) << " " <<
        "-I./";

    program.build({ queue.getInfo<CL_QUEUE_DEVICE>() }, build_opts.str().c_str());

    foward_marshaling_bxb_kernel = cl::Kernel{ program, "foward_marshaling_bxb" };
    tiled_diag_pivot_x1_kernel = cl::Kernel{ program, "tiled_diag_pivot_x1" };
    spike_local_reduction_x1_kernel = cl::Kernel{ program, "spike_local_reduction_x1" };
    spike_global_solving_x1_kernel = cl::Kernel{ program, "spike_global_solving_x1" };
    spike_local_solving_x1_kernel = cl::Kernel{ program, "spike_local_solving_x1" };
    spike_back_sub_x1_kernel = cl::Kernel{ program, "spike_back_sub_x1" };
    back_marshaling_bxb_kernel = cl::Kernel{ program, "back_marshaling_bxb" };

    //cudaFuncSetCacheConfig(tiled_diag_pivot_x1<T, TT>, cudaFuncCachePreferL1);
    //cudaFuncSetCacheConfig(spike_GPU_back_sub_x1<T>, cudaFuncCachePreferL1);
}

template <typename T, typename TT>
cl::Event tridiag_solver<T, TT>::gtsv_spike_partial_diag_pivot(std::vector<T>& dl, std::vector<T>& d, std::vector<T>& du, std::vector<T>& b)
{
    cl::Buffer d_buf(queue_.getInfo<CL_QUEUE_CONTEXT>(), d.begin(), d.end(), false), // false = read_only
               du_buf(queue_.getInfo<CL_QUEUE_CONTEXT>(), du.begin(), du.end(), false), // false = read_only
               dl_buf(queue_.getInfo<CL_QUEUE_CONTEXT>(), dl.begin(), dl.end(), false), // false = read_only
               b_buf(queue_.getInfo<CL_QUEUE_CONTEXT>(), CL_MEM_READ_WRITE, d.size() * sizeof(T));

    cl::Event result = gtsv_spike_partial_diag_pivot(dl_buf, d_buf, du_buf, b_buf).wait();

    cl::copy(queue_, b_buf, b.begin(), b.end());

    return result;
}

template <typename T, typename TT>
cl::Event tridiag_solver<T, TT>::gtsv_spike_partial_diag_pivot(cl::Buffer dl, cl::Buffer d, cl::Buffer du, cl::Buffer b, std::vector<cl::Event> wait)
{
    cl::size_type tile = 128;
    cl::size_type tile_marshal = 16;

    cl::size_type m = d.getInfo<CL_MEM_SIZE>() / sizeof(T);

    if (last_m != m) // cache grid for subsequent calls
    {
        findBestGrid(m, tile_marshal);
    }

    cl::size_type local_reduction_share_size = 2 * b_dim * 3 * sizeof(T);
    cl::size_type global_share_size = 2 * s * 3 * sizeof(T);
    cl::size_type local_solving_share_size = (2 * b_dim * 2 + 2 * b_dim + 2) * sizeof(T);
    cl::size_type marshaling_share_size = tile_marshal * (tile_marshal + 1) * sizeof(T);

    int c_s = s; //griddim.x
    int c_stride = stride;
    int c_b_dim = b_dim, c_m_pad = m_pad;
    int c_tile = 128;
    int c_tile_marshal = 16;
    int c_T_size = sizeof(T);

    int c_local_reduction_share_size = 2 * c_b_dim * 3 * c_T_size;
    int c_global_share_size = 2 * c_s * 3 * c_T_size;
    int c_local_solving_share_size = (2 * c_b_dim * 2 + 2 * c_b_dim + 2) * c_T_size;
    int c_marshaling_share_size = c_tile_marshal * (c_tile_marshal + 1) * c_T_size;

    dim3 c_g_data(c_b_dim / c_tile_marshal, c_s);
    dim3 c_b_data(c_tile_marshal, c_tile_marshal);

    if (last_m != m) // resize temp buffers only if needed
    {
        cl::Context ctx = queue_.getInfo<CL_QUEUE_CONTEXT>();

        flag = cl::Buffer(ctx, CL_MEM_READ_WRITE, sizeof(cl_bool));
        dl_buffer = cl::Buffer(ctx, CL_MEM_READ_WRITE, sizeof(T) * m_pad);
        d_buffer = cl::Buffer(ctx, CL_MEM_READ_WRITE, sizeof(T) * m_pad);
        du_buffer = cl::Buffer(ctx, CL_MEM_READ_WRITE, sizeof(T) * m_pad);
        b_buffer = cl::Buffer(ctx, CL_MEM_READ_WRITE, sizeof(T) * m_pad);
        w_buffer = cl::Buffer(ctx, CL_MEM_READ_WRITE, sizeof(T) * m_pad);
        v_buffer = cl::Buffer(ctx, CL_MEM_READ_WRITE, sizeof(T) * m_pad);
        c2_buffer = cl::Buffer(ctx, CL_MEM_READ_WRITE, sizeof(T) * m_pad);

        x_level_2 = cl::Buffer(ctx, CL_MEM_READ_WRITE, sizeof(T) * s * 2);
        w_level_2 = cl::Buffer(ctx, CL_MEM_READ_WRITE, sizeof(T) * s * 2);
        v_level_2 = cl::Buffer(ctx, CL_MEM_READ_WRITE, sizeof(T) * s * 2);

        cudaMalloc((void**)& c_flag, sizeof(bool) * c_m_pad);
        cudaMalloc((void**)& c_dl_buffer, c_T_size * c_m_pad);
        cudaMalloc((void**)& c_d_buffer, c_T_size * c_m_pad);
        cudaMalloc((void**)& c_du_buffer, c_T_size * c_m_pad);
        cudaMalloc((void**)& c_b_buffer, c_T_size * c_m_pad);
        cudaMalloc((void**)& c_w_buffer, c_T_size * c_m_pad);
        cudaMalloc((void**)& c_v_buffer, c_T_size * c_m_pad);
        cudaMalloc((void**)& c_c2_buffer, c_T_size * c_m_pad);

        cudaMalloc((void**)& c_x_level_2, c_T_size * c_s * 2);
        cudaMalloc((void**)& c_w_level_2, c_T_size * c_s * 2);
        cudaMalloc((void**)& c_v_level_2, c_T_size * c_s * 2);
    }

    cl::KernelFunctor<cl::Buffer,
                      cl::Buffer,
                      cl::LocalSpaceArg,
                      cl_int,
                      cl_int,
                      cl_int,
                      T> cl_foward_marshaling_bxb{ foward_marshaling_bxb_kernel };

    cl::KernelFunctor<cl::Buffer,
                      cl::Buffer,
                      cl::Buffer,
                      cl::Buffer,
                      cl::Buffer,
                      cl::Buffer,
                      cl::Buffer,
                      cl::Buffer,
                      cl_int,
                      cl_int> cl_tiled_diag_pivot_x1{ tiled_diag_pivot_x1_kernel };

    cl::KernelFunctor<cl::Buffer,
                      cl::Buffer,
                      cl::Buffer,
                      cl::Buffer,
                      cl::Buffer,
                      cl::Buffer,
                      cl::LocalSpaceArg,
                      cl_int> cl_spike_local_reduction_x1{ spike_local_reduction_x1_kernel };

    cl::KernelFunctor<cl::Buffer,
                      cl::Buffer,
                      cl::Buffer,
                      cl::LocalSpaceArg,
                      cl_int> cl_spike_global_solving_x1{ spike_global_solving_x1_kernel };

    cl::KernelFunctor<cl::Buffer,
                      cl::Buffer,
                      cl::Buffer,
                      cl::Buffer,
                      cl::LocalSpaceArg,
                      cl_int> cl_spike_local_solving_x1{ spike_local_solving_x1_kernel };

    cl::KernelFunctor<cl::Buffer,
                      cl::Buffer,
                      cl::Buffer,
                      cl::Buffer,
                      cl_int> cl_spike_back_sub_x1{ spike_back_sub_x1_kernel };

    cl::KernelFunctor<cl::Buffer,
                      cl::Buffer,
                      cl::LocalSpaceArg,
                      cl_int,
                      cl_int,
                      cl_int> cl_back_marshaling_bxb{ back_marshaling_bxb_kernel };

    cl::EnqueueArgs fwd_enq_args{ queue_,
                                  wait,
                                  cl::NDRange{ b_dim, s * tile_marshal },      // g_data
                                  cl::NDRange{ tile_marshal, tile_marshal } }; // b_data
    // data layout transformation
    forward_events = { cl_foward_marshaling_bxb(fwd_enq_args, dl_buffer, dl, cl::Local(marshaling_share_size), (cl_int)stride, (cl_int)b_dim, (cl_int)m, 0),
                       cl_foward_marshaling_bxb(fwd_enq_args, d_buffer,  d,  cl::Local(marshaling_share_size), (cl_int)stride, (cl_int)b_dim, (cl_int)m, 1),
                       cl_foward_marshaling_bxb(fwd_enq_args, du_buffer, du, cl::Local(marshaling_share_size), (cl_int)stride, (cl_int)b_dim, (cl_int)m, 0),
                       cl_foward_marshaling_bxb(fwd_enq_args, b_buffer,  b,  cl::Local(marshaling_share_size), (cl_int)stride, (cl_int)b_dim, (cl_int)m, 0) };

    T *c_dl = (T*)queue_.enqueueMapBuffer(dl, true, CL_MAP_READ, 0, sizeof(T) * m_pad),
      *c_d = (T*)queue_.enqueueMapBuffer(d, true, CL_MAP_READ, 0, sizeof(T) * m_pad),
      *c_du = (T*)queue_.enqueueMapBuffer(du, true, CL_MAP_READ, 0, sizeof(T) * m_pad),
      *c_b = (T*)queue_.enqueueMapBuffer(b, true, CL_MAP_READ, 0, sizeof(T) * m_pad);

    foward_marshaling_bxb<T><<<c_g_data, c_b_data, c_marshaling_share_size>>>(c_dl_buffer, c_dl, c_stride, c_b_dim, m, cuGet<T>(0));
    foward_marshaling_bxb<T><<<c_g_data, c_b_data, c_marshaling_share_size>>>(c_d_buffer, c_d, c_stride, c_b_dim, m, cuGet<T>(1));
    foward_marshaling_bxb<T><<<c_g_data, c_b_data, c_marshaling_share_size>>>(c_du_buffer, c_du, c_stride, c_b_dim, m, cuGet<T>(0));
    foward_marshaling_bxb<T><<<c_g_data, c_b_data, c_marshaling_share_size>>>(c_b_buffer, c_b, c_stride, c_b_dim, m, cuGet<T>(0));

    if (!validate(queue_, dl_buffer, c_dl_buffer, 1e-6)) { std::cerr << "dl_buffer" << std::endl; throw std::runtime_error{"dl_buffer"}; }
    if (!validate(queue_, d_buffer, c_d_buffer, 1e-6)) { std::cerr << "d_buffer" << std::endl; throw std::runtime_error{"d_buffer"}; }
    if (!validate(queue_, du_buffer, c_du_buffer, 1e-6)) { std::cerr << "du_buffer" << std::endl; throw std::runtime_error{"du_buffer"}; }
    if (!validate(queue_, b_buffer, c_b_buffer, 1e-6)) { std::cerr << "b_buffer" << std::endl; throw std::runtime_error{"b_buffer"}; }

    // partitioned solver
    pivot_event = { cl_tiled_diag_pivot_x1(cl::EnqueueArgs{ queue_,
                                                         forward_events,
                                                         cl::NDRange{ s * b_dim },
                                                         cl::NDRange{ b_dim } },
                                        b_buffer,
                                        w_buffer,
                                        v_buffer,
                                        c2_buffer,
                                        flag,
                                        dl_buffer,
                                        d_buffer,
                                        du_buffer,
                                        (cl_int)stride,
                                        (cl_int)tile) };

    // SPIKE solver
    local_reduction_event = { cl_spike_local_reduction_x1(cl::EnqueueArgs{ queue_,
                                                                        pivot_event,
                                                                        cl::NDRange{ s * b_dim },
                                                                        cl::NDRange{ b_dim } },
                                                       b_buffer,
                                                       w_buffer,
                                                       v_buffer,
                                                       x_level_2,
                                                       w_level_2,
                                                       v_level_2,
                                                       cl::Local(local_reduction_share_size),
                                                       (cl_int)stride) };

    global_solving_event = { cl_spike_global_solving_x1(cl::EnqueueArgs{ queue_,
                                                                      local_reduction_event,
                                                                      cl::NDRange{ 1 * 32 },
                                                                      cl::NDRange{ 32 } },
                                                     x_level_2,
                                                     w_level_2,
                                                     v_level_2,
                                                     cl::Local(global_share_size),
                                                     (cl_int)s) };

    local_solving_event = { cl_spike_local_solving_x1(cl::EnqueueArgs{ queue_,
                                                                    global_solving_event,
                                                                    cl::NDRange{ s * b_dim },
                                                                    cl::NDRange{ b_dim } },
                                                   b_buffer,
                                                   w_buffer,
                                                   v_buffer,
                                                   x_level_2,
                                                   cl::Local(local_solving_share_size),
                                                   (cl_int)stride) };

    back_sub_event = { cl_spike_back_sub_x1(cl::EnqueueArgs{ queue_,
                                                          local_solving_event,
                                                          cl::NDRange{ s * b_dim },
                                                          cl::NDRange{ b_dim } },
                                         b_buffer,
                                         w_buffer,
                                         v_buffer,
                                         x_level_2,
                                         (cl_int)stride) };

    back_marshal_event = { cl_back_marshaling_bxb(cl::EnqueueArgs{ queue_,
                                                                back_sub_event,
                                                                cl::NDRange{ b_dim, s * tile_marshal },
                                                                cl::NDRange{ tile_marshal, tile_marshal } },
                                               b,
                                               b_buffer,
                                               cl::Local(marshaling_share_size),
                                               (cl_int)stride,
                                               (cl_int)b_dim,
                                               (cl_int)m) };

    if (last_m != m) // cache grid for subsequent calls
    {
        last_m = m;
    }

    return back_marshal_event.at(0);
}

template <typename T, typename TT>
void tridiag_solver<T, TT>::findBestGrid(cl::size_type m, cl::size_type tile_marshal)
{
    cl::size_type B_DIM_MAX, S_MAX;

    if (sizeof(T) == 4) { // float
        B_DIM_MAX = 256;
        S_MAX = 512;
    }
    else if (sizeof(T) == 8) { // double and complex
        B_DIM_MAX = 128;
        S_MAX = 256;
    }
    else { // doubleComplex
        B_DIM_MAX = 64;
        S_MAX = 128;
    }

    // b_dim must be multiple of 32
    if (m < B_DIM_MAX * tile_marshal) {
        b_dim = std::max<cl::size_type>(32, (m / (32 * tile_marshal)) * 32);
        s = 1;
        m_pad = ((m + b_dim * tile_marshal - 1) / (b_dim * tile_marshal)) * (b_dim * tile_marshal);
        stride = m_pad / (s * b_dim);
    }
    else {
        b_dim = B_DIM_MAX;

        s = 1;
        do {
            cl::size_type s_tmp = s * 2;
            cl::size_type m_pad_tmp = ((m + s_tmp * b_dim * tile_marshal - 1) / (s_tmp * b_dim * tile_marshal)) * (s_tmp * b_dim * tile_marshal);
            double diff = (double)(m_pad_tmp - m) / double(m);
            // We do not want to have more than 20% oversize
            if (diff < .2) {
                s = s_tmp;
            }
            else {
                break;
            }
        } while (s < S_MAX);

        m_pad = ((m + s * b_dim * tile_marshal - 1) / (s * b_dim * tile_marshal)) * (s * b_dim * tile_marshal);
        stride = m_pad / (s * b_dim);
    }
}
