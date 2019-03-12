// OpenCL includes
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 100
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/cl2.hpp>

// STL includes
#include <string>       // std::string
#include <fstream>      // std::ifstream
#include <stdexcept>    // std::runtime_error
#include <iterator>     // std::istreambuf_iterator
#include <sstream>      // std::stringstream
#include <vector>       // std::vector
#include <algorithm>    // std::max
#include <complex>


template <typename T, typename TT>
class tridiag_solver
{
public:

    tridiag_solver(cl::CommandQueue queue);
        
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

    cl::Buffer flag_buffer, dl_buffer, d_buffer, du_buffer, b_buffer,
               w_buffer, v_buffer, c2_buffer,
               x_level_2, w_level_2, v_level_2;

    std::vector<cl::Event> forward_events,
                           pivot_event,
                           local_reduction_event,
                           global_solving_event,
                           local_solving_event,
                           back_sub_event,
                           back_marshal_event;

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

    if (last_m != m) // resize temp buffers only if needed
    {
        cl::Context ctx = queue_.getInfo<CL_QUEUE_CONTEXT>();

        flag_buffer = cl::Buffer(ctx, CL_MEM_READ_WRITE, sizeof(cl_uchar) * m_pad);
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
    }
    std::vector<cl_uchar> flag_result(flag_buffer.getInfo<CL_MEM_SIZE>() / sizeof(cl_uchar));
    std::vector<T> dl_result(dl_buffer.getInfo<CL_MEM_SIZE>() / sizeof(T)),
                   d_result(d_buffer.getInfo<CL_MEM_SIZE>() / sizeof(T)),
                   du_result(du_buffer.getInfo<CL_MEM_SIZE>() / sizeof(T)),
                   b_result(b_buffer.getInfo<CL_MEM_SIZE>() / sizeof(T)),
                   w_result(w_buffer.getInfo<CL_MEM_SIZE>() / sizeof(T)),
                   v_result(v_buffer.getInfo<CL_MEM_SIZE>() / sizeof(T)),
                   c2_result(c2_buffer.getInfo<CL_MEM_SIZE>() / sizeof(T)),
                   x2_result(x_level_2.getInfo<CL_MEM_SIZE>() / sizeof(T)),
                   w2_result(w_level_2.getInfo<CL_MEM_SIZE>() / sizeof(T)),
                   v2_result(v_level_2.getInfo<CL_MEM_SIZE>() / sizeof(T));
    

    cl::KernelFunctor<cl::Buffer,
                      cl::Buffer,
                      cl::LocalSpaceArg,
                      cl_int,
                      cl_int,
                      cl_int,
                      T> foward_marshaling_bxb{ foward_marshaling_bxb_kernel };

    cl::KernelFunctor<cl::Buffer,
                      cl::Buffer,
                      cl::Buffer,
                      cl::Buffer,
                      cl::Buffer,
                      cl::Buffer,
                      cl::Buffer,
                      cl::Buffer,
                      cl_int,
                      cl_int> tiled_diag_pivot_x1{ tiled_diag_pivot_x1_kernel };

    cl::KernelFunctor<cl::Buffer,
                      cl::Buffer,
                      cl::Buffer,
                      cl::Buffer,
                      cl::Buffer,
                      cl::Buffer,
                      cl::LocalSpaceArg,
                      cl_int> spike_local_reduction_x1{ spike_local_reduction_x1_kernel };

    cl::KernelFunctor<cl::Buffer,
                      cl::Buffer,
                      cl::Buffer,
                      cl::LocalSpaceArg,
                      cl_int> spike_global_solving_x1{ spike_global_solving_x1_kernel };

    cl::KernelFunctor<cl::Buffer,
                      cl::Buffer,
                      cl::Buffer,
                      cl::Buffer,
                      cl::LocalSpaceArg,
                      cl_int> spike_local_solving_x1{ spike_local_solving_x1_kernel };

    cl::KernelFunctor<cl::Buffer,
                      cl::Buffer,
                      cl::Buffer,
                      cl::Buffer,
                      cl_int> spike_back_sub_x1{ spike_back_sub_x1_kernel };

    cl::KernelFunctor<cl::Buffer,
                      cl::Buffer,
                      cl::LocalSpaceArg,
                      cl_int,
                      cl_int,
                      cl_int> back_marshaling_bxb{ back_marshaling_bxb_kernel };

    cl::EnqueueArgs fwd_enq_args{ queue_,
                                  wait,
                                  cl::NDRange{ b_dim, s * tile_marshal },      // g_data
                                  cl::NDRange{ tile_marshal, tile_marshal } }; // b_data
    // data layout transformation
    forward_events = { foward_marshaling_bxb(fwd_enq_args, dl_buffer, dl, cl::Local(marshaling_share_size), (cl_int)stride, (cl_int)b_dim, (cl_int)m, 0),
                       foward_marshaling_bxb(fwd_enq_args, d_buffer,  d,  cl::Local(marshaling_share_size), (cl_int)stride, (cl_int)b_dim, (cl_int)m, 1),
                       foward_marshaling_bxb(fwd_enq_args, du_buffer, du, cl::Local(marshaling_share_size), (cl_int)stride, (cl_int)b_dim, (cl_int)m, 0),
                       foward_marshaling_bxb(fwd_enq_args, b_buffer,  b,  cl::Local(marshaling_share_size), (cl_int)stride, (cl_int)b_dim, (cl_int)m, 0) };

    cl::WaitForEvents(forward_events);
   
    cl::copy(queue_, dl_buffer, dl_result.begin(), dl_result.end()); if (std::any_of(dl_result.cbegin(), dl_result.cend(), std::isnan<T>)) throw std::runtime_error{ "NaN detected" };
    cl::copy(queue_, d_buffer, d_result.begin(), d_result.end()); if (std::any_of(d_result.cbegin(), d_result.cend(), std::isnan<T>)) throw std::runtime_error{ "NaN detected" };
    cl::copy(queue_, du_buffer, du_result.begin(), du_result.end()); if (std::any_of(du_result.cbegin(), du_result.cend(), std::isnan<T>)) throw std::runtime_error{ "NaN detected" };

    // partitioned solver
    pivot_event = { tiled_diag_pivot_x1(cl::EnqueueArgs{ queue_,
                                                         forward_events,
                                                         cl::NDRange{ s * b_dim },
                                                         cl::NDRange{ b_dim } },
                                        b_buffer,
                                        w_buffer,
                                        v_buffer,
                                        c2_buffer,
                                        flag_buffer,
                                        dl_buffer,
                                        d_buffer,
                                        du_buffer,
                                        (cl_int)stride,
                                        (cl_int)tile) };

    cl::WaitForEvents(pivot_event);

    cl::copy(queue_, b_buffer, b_result.begin(), b_result.end()); if (std::any_of(b_result.cbegin(), b_result.cend(), std::isnan<T>)) throw std::runtime_error{ "NaN detected" };
    cl::copy(queue_, w_buffer, w_result.begin(), w_result.end()); if (std::any_of(w_result.cbegin(), w_result.cend(), std::isnan<T>)) throw std::runtime_error{ "NaN detected" };
    cl::copy(queue_, v_buffer, v_result.begin(), v_result.end()); if (std::any_of(v_result.cbegin(), v_result.cend(), std::isnan<T>)) throw std::runtime_error{ "NaN detected" };
    cl::copy(queue_, c2_buffer, c2_result.begin(), c2_result.end()); if (std::any_of(c2_result.cbegin(), c2_result.cend(), std::isnan<T>)) throw std::runtime_error{ "NaN detected" };
    cl::copy(queue_, flag_buffer, flag_result.begin(), flag_result.end());

    // SPIKE solver
    local_reduction_event = { spike_local_reduction_x1(cl::EnqueueArgs{ queue_,
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

    cl::WaitForEvents(local_reduction_event);

    cl::copy(queue_, b_buffer, b_result.begin(), b_result.end());
    cl::copy(queue_, w_buffer, w_result.begin(), w_result.end());
    cl::copy(queue_, v_buffer, v_result.begin(), v_result.end());
    cl::copy(queue_, x_level_2, x2_result.begin(), x2_result.end());
    cl::copy(queue_, w_level_2, w2_result.begin(), w2_result.end());
    cl::copy(queue_, v_level_2, v2_result.begin(), v2_result.end());

    global_solving_event = { spike_global_solving_x1(cl::EnqueueArgs{ queue_,
                                                                      local_reduction_event,
                                                                      cl::NDRange{ 1 * 32 },
                                                                      cl::NDRange{ 32 } },
                                                     x_level_2,
                                                     w_level_2,
                                                     v_level_2,
                                                     cl::Local(global_share_size),
                                                     (cl_int)s) };

    local_solving_event = { spike_local_solving_x1(cl::EnqueueArgs{ queue_,
                                                                    global_solving_event,
                                                                    cl::NDRange{ s * b_dim },
                                                                    cl::NDRange{ b_dim } },
                                                   b_buffer,
                                                   w_buffer,
                                                   v_buffer,
                                                   x_level_2,
                                                   cl::Local(local_solving_share_size),
                                                   (cl_int)stride) };

    back_sub_event = { spike_back_sub_x1(cl::EnqueueArgs{ queue_,
                                                          local_solving_event,
                                                          cl::NDRange{ s * b_dim },
                                                          cl::NDRange{ b_dim } },
                                         b_buffer,
                                         w_buffer,
                                         v_buffer,
                                         x_level_2,
                                         (cl_int)stride) };

    back_marshal_event = { back_marshaling_bxb(cl::EnqueueArgs{ queue_,
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
