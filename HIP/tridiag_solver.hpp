#pragma once

/*******************************************************************************************************
                              University of Illinois/NCSA Open Source License
                                 Copyright (c) 2012 University of Illinois
                                          All rights reserved.

                                        Developed by: IMPACT Group
                                          University of Illinois
                                      http://impact.crhc.illinois.edu

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
to deal with the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
 and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

  Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimers.
  Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimers in the documentation and/or other materials provided with the distribution.
  Neither the names of IMPACT Group, University of Illinois, nor the names of its contributors may be used to endorse or promote products derived from this Software without specific prior written permission.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS WITH THE SOFTWARE.

*******************************************************************************************************/

#include "hip_definitions.hpp"
#include <hip/hip_runtime.h>

#include "spike_kernel.hpp"

template <typename T, typename TT>
class tridiag_solver
{
public:

	tridiag_solver(int m, int k = 1);
	~tridiag_solver();

	void solve(const T* dl, const T* d, const T* du, T* b);

private:

	int m, k, m_pad, b_dim, s, stride, last_m;
	static constexpr int tile = 128;
	static constexpr int tile_marshal = 16;
	static constexpr int T_size = sizeof(T);

	int local_reduction_share_size, global_share_size, local_solving_share_size, marshaling_share_size;

	dim3 g_data, b_data, g_dp, g_spike;

	bool* flag;   // tag for pivoting
	T* dl_buffer;
	T* d_buffer;
	T* du_buffer;
	T* b_buffer;
	T* w_buffer;
	T* v_buffer;
	T* c2_buffer;

	T* w_mirror;
	T* v_mirror;
	T* x_mirror2;
	T* w_mirror2;
	T* v_mirror2;

	T* x_level_2;
	T* w_level_2;
	T* v_level_2;

	void find_best_grid();
	void gtsv_spike_partial_diag_pivot_v1(const T* dl, const T* d, const T* du, T* b);
	void gtsv_spike_partial_diag_pivot_few(const T* dl, const T* d, const T* du, T* b);

	template <typename U, typename UU>
	auto max(U&& lhs, UU&& rhs){ return lhs > rhs ? lhs : rhs; };
};

template <typename T, typename TT>
tridiag_solver<T, TT>::tridiag_solver(int m_in, int k_in)
	: m{m_in}, k{k_in}
{
	find_best_grid();

	local_reduction_share_size = 2*b_dim*3*T_size;
	global_share_size = 2*s*3*T_size;
	local_solving_share_size = (2*b_dim*2+2*b_dim+2)*T_size;
	marshaling_share_size = tile_marshal*(tile_marshal+1)*T_size;

	g_data = dim3(b_dim/tile_marshal,s);
	b_data = dim3(tile_marshal,tile_marshal);

	if(k != 1)
	{
		g_dp = dim3(s,k-1);
		g_spike = dim3(s,k);
	}

	cudaMalloc((void **)&flag, sizeof(bool)*m_pad);
	cudaMalloc((void **)&dl_buffer, T_size*m_pad);
	cudaMalloc((void **)&d_buffer, T_size*m_pad);
	cudaMalloc((void **)&du_buffer, T_size*m_pad);
	cudaMalloc((void **)&b_buffer, T_size*m_pad*k); //same as x
	cudaMalloc((void **)&w_buffer, T_size*m_pad);
	cudaMalloc((void **)&v_buffer, T_size*m_pad);
	cudaMalloc((void **)&c2_buffer, T_size*m_pad);

	if(k != 1)
	{
		cudaMalloc((void **)&w_mirror, T_size*b_dim*s*2);
		cudaMalloc((void **)&v_mirror, T_size*b_dim*s*2);

		cudaMalloc((void **)&x_mirror2, T_size*s*2*k);
		cudaMalloc((void **)&w_mirror2, T_size*s*2);
		cudaMalloc((void **)&v_mirror2, T_size*s*2);
	}
	else
	{
		cudaMalloc((void **)&x_level_2, T_size*s*2);
		cudaMalloc((void **)&w_level_2, T_size*s*2);
		cudaMalloc((void **)&v_level_2, T_size*s*2);
	}

	cudaFuncSetCacheConfig((tiled_diag_pivot_x1<T,TT>),cudaFuncCachePreferL1);
	if(k != 1)
	{
		cudaFuncSetCacheConfig((tiled_diag_pivot_x_few<T,TT>),cudaFuncCachePreferL1);
		cudaFuncSetCacheConfig(spike_GPU_back_sub_x_few<T>,cudaFuncCachePreferL1);
	}
	else
	{
		cudaFuncSetCacheConfig(spike_GPU_back_sub_x1<T>,cudaFuncCachePreferL1);
	}
}

template <typename T, typename TT>
tridiag_solver<T, TT>::~tridiag_solver()
{
	cudaFree(flag);
	cudaFree(dl_buffer);
	cudaFree(d_buffer);
	cudaFree(du_buffer);
	cudaFree(b_buffer);
	cudaFree(w_buffer);
	cudaFree(v_buffer);
	cudaFree(c2_buffer);
	if(k != 1)
	{
		cudaFree(w_mirror);
		cudaFree(v_mirror);
		cudaFree(x_mirror2);
		cudaFree(w_mirror2);
		cudaFree(v_mirror2);
	}
	else
	{
		cudaFree(x_level_2);
		cudaFree(w_level_2);
		cudaFree(v_level_2);
	}
}

template <typename T, typename TT> 
void tridiag_solver<T, TT>::solve(const T* dl, const T* d, const T* du, T* b)
{
	if(k<=0) return;
	if(k==1) gtsv_spike_partial_diag_pivot_v1(dl, d, du, b);
	else     gtsv_spike_partial_diag_pivot_few(dl, d, du, b);
}

template <typename T, typename TT>
void tridiag_solver<T, TT>::find_best_grid()
{
    int B_DIM_MAX, S_MAX;

    if ( sizeof(T) == 4) {
        B_DIM_MAX = 256;
        S_MAX     = 512;    
    }
    else if (sizeof(T) == 8){ /* double and complex */
        B_DIM_MAX = 128;
        S_MAX     = 256;     
    }
    else { /* doubleComplex */
        B_DIM_MAX = 64;
        S_MAX      = 128;    
    }

    /* b_dim must be multiple of 32 */
    if ( m < B_DIM_MAX * tile_marshal ) {
        b_dim = max( 32, (m/(32*tile_marshal))*32);
        s = 1;
        m_pad = ((m + b_dim * tile_marshal -1)/(b_dim * tile_marshal)) * (b_dim * tile_marshal);
        stride = m_pad/(s*b_dim);    
    }
    else {
        b_dim = B_DIM_MAX;
        
        s = 1;
        do {
            int s_tmp = s * 2;
            int m_pad_tmp = ((m + s_tmp*b_dim*tile_marshal -1)/(s_tmp*b_dim*tile_marshal)) * (s_tmp*b_dim*tile_marshal);
            float diff = (float)(m_pad_tmp - m)/float(m);
            /* We do not want to have more than 20% oversize */
            if ( diff < .2 ) {
                s = s_tmp;
            }
            else {
                break;
            }
        } while (s < S_MAX);
                       
        m_pad = ((m + s*b_dim*tile_marshal -1)/(s*b_dim*tile_marshal)) * (s*b_dim*tile_marshal);
        stride = m_pad/(s*b_dim);
    }
}

template <typename T, typename TT>
void tridiag_solver<T, TT>::gtsv_spike_partial_diag_pivot_v1(const T* dl, const T* d, const T* du, T* b)
{
	// data layout transformation
	hipLaunchKernelGGL(HIP_KERNEL_NAME(foward_marshaling_bxb<T>), g_data ,b_data, marshaling_share_size, 0, dl_buffer, dl, stride, b_dim,m, cuGet<T>(0));
	hipLaunchKernelGGL(HIP_KERNEL_NAME(foward_marshaling_bxb<T>), g_data ,b_data, marshaling_share_size, 0, d_buffer,  d,  stride, b_dim,m, cuGet<T>(1));
	hipLaunchKernelGGL(HIP_KERNEL_NAME(foward_marshaling_bxb<T>), g_data ,b_data, marshaling_share_size, 0, du_buffer, du, stride, b_dim,m, cuGet<T>(0));
	hipLaunchKernelGGL(HIP_KERNEL_NAME(foward_marshaling_bxb<T>), g_data ,b_data, marshaling_share_size, 0, b_buffer,  b,  stride, b_dim,m, cuGet<T>(0));

	// partitioned solver
	hipLaunchKernelGGL(HIP_KERNEL_NAME(tiled_diag_pivot_x1<T,TT>), s, b_dim, 0, 0, b_buffer, w_buffer, v_buffer, c2_buffer, flag, dl_buffer, d_buffer, du_buffer, stride, tile);

	// SPIKE solver
	hipLaunchKernelGGL(HIP_KERNEL_NAME(spike_local_reduction_x1<T>), s, b_dim,local_reduction_share_size, 0, b_buffer,w_buffer,v_buffer,x_level_2, w_level_2, v_level_2,stride);
	hipLaunchKernelGGL(HIP_KERNEL_NAME(spike_GPU_global_solving_x1),1,32,global_share_size,0, x_level_2,w_level_2,v_level_2,s);
	hipLaunchKernelGGL(HIP_KERNEL_NAME(spike_GPU_local_solving_x1<T>),s,b_dim,local_solving_share_size,0,b_buffer,w_buffer,v_buffer,x_level_2,stride);
	hipLaunchKernelGGL(HIP_KERNEL_NAME(spike_GPU_back_sub_x1<T>),s,b_dim,0,0,b_buffer,w_buffer,v_buffer, x_level_2,stride);

	hipLaunchKernelGGL(HIP_KERNEL_NAME(back_marshaling_bxb<T>),g_data ,b_data, marshaling_share_size,0,b,b_buffer,stride,b_dim,m);
}

template <typename T, typename TT>
void tridiag_solver<T, TT>::gtsv_spike_partial_diag_pivot_few(const T* dl, const T* d, const T* du, T* b)
{
	// data layout transformation
	hipLaunchKernelGGL(HIP_KERNEL_NAME(foward_marshaling_bxb<T>),g_data ,b_data, marshaling_share_size,0,dl_buffer, dl, stride, b_dim,m, cuGet<T>(0));
	hipLaunchKernelGGL(HIP_KERNEL_NAME(foward_marshaling_bxb<T>),g_data ,b_data, marshaling_share_size,0,d_buffer,  d,  stride, b_dim,m, cuGet<T>(1));
	hipLaunchKernelGGL(HIP_KERNEL_NAME(foward_marshaling_bxb<T>),g_data ,b_data, marshaling_share_size,0,du_buffer, du, stride, b_dim,m, cuGet<T>(0));

	// TODO: it will be replaced by a kernel
	for(int i=0;i<k;i++)
	{
		hipLaunchKernelGGL(HIP_KERNEL_NAME(foward_marshaling_bxb<T>),g_data ,b_data, marshaling_share_size,0,b_buffer+i*m_pad,  b+i*m,  stride, b_dim,m, cuGet<T>(0));
	}

	// partitioned solver

	// solve w, v and the fist x
	hipLaunchKernelGGL(HIP_KERNEL_NAME(tiled_diag_pivot_x1<T,TT>),s,b_dim,0,0,b_buffer, w_buffer, v_buffer, c2_buffer, flag, dl_buffer, d_buffer, du_buffer, stride, tile);
	// solve the rest x
	hipLaunchKernelGGL(HIP_KERNEL_NAME(tiled_diag_pivot_x_few<T,TT>),g_dp,b_dim,0,0,b_buffer+m_pad,flag,dl_buffer,c2_buffer,du_buffer,stride,tile,m_pad);

	// SPIKE solver
	hipLaunchKernelGGL(HIP_KERNEL_NAME(spike_local_reduction_x_few<T>),g_spike,b_dim,local_reduction_share_size,0,b_buffer,w_buffer,v_buffer,w_mirror,v_mirror,x_mirror2,w_mirror2,v_mirror2,stride,m_pad);
	hipLaunchKernelGGL(HIP_KERNEL_NAME(spike_GPU_global_solving_x_few<T>),k,32,global_share_size,0,x_mirror2,w_mirror2,v_mirror2,s);
	hipLaunchKernelGGL(HIP_KERNEL_NAME(spike_GPU_local_solving_x_few<T>),g_spike,b_dim,local_solving_share_size,0,b_buffer,w_mirror,v_mirror,x_mirror2,stride,m_pad);
	hipLaunchKernelGGL(HIP_KERNEL_NAME(spike_GPU_back_sub_x_few<T>),g_spike,b_dim,0,0,b_buffer,w_buffer,v_buffer,x_mirror2,stride,m_pad);

	// TODO: it will be replaced by a kernel
	for(int i=0;i<k;i++)
	{
		hipLaunchKernelGGL(HIP_KERNEL_NAME(back_marshaling_bxb<T>),g_data ,b_data, marshaling_share_size,0,b+i*m,b_buffer+i*m_pad,stride,b_dim,m);
	}
}
