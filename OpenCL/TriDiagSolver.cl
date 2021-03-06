#include <TriDiagSolverOps.cl>

//#pragma OPENCL EXTENSION cl_amd_printf : enable

// Data layout transformation for inputs
kernel void foward_marshaling_bxb(global elem* x,
                                  global const elem* y,
								  local elem* share,
                                  const int h_stride,
                                  const int l_stride,
                                  int m,
                                  elem pad)
{	
	int b_dim = get_local_size(0); 	//16

	int global_in = get_group_id(1)*l_stride*h_stride + (get_group_id(0)*b_dim+get_local_id(1))*h_stride+get_local_id(0);
	int global_out = get_group_id(1)*l_stride*h_stride + get_local_id(1)*l_stride + get_group_id(0)*b_dim+get_local_id(0);
	int shared_in = get_local_id(1)*(b_dim+1)+get_local_id(0);
	int shared_out = get_local_id(0)*(b_dim+1)+get_local_id(1);

	for (int k = 0 ; k < h_stride ; k += b_dim)
	{	
		share[shared_in] = global_in >= m ? pad : y[global_in];
		global_in += b_dim;
		
		barrier(CLK_LOCAL_MEM_FENCE);
		
		x[global_out] = share[shared_out];
		global_out += b_dim * l_stride;

		barrier(CLK_LOCAL_MEM_FENCE);
	}		
}

// Data layout transformation for results
kernel void back_marshaling_bxb(global elem* x,
                                global const elem* y,
								local elem* share,
                                const int h_stride,
                                const int l_stride,
                                int m)
{	
	int b_dim;
	
	int global_in;
	int global_out;
	int shared_in;
	int shared_out;
	
	b_dim = get_local_size(0); 	//16

	global_out = get_group_id(1)*l_stride*h_stride +  (get_group_id(0)*b_dim+get_local_id(1))*h_stride+get_local_id(0);
	global_in = get_group_id(1)*l_stride*h_stride + get_local_id(1)*l_stride + get_group_id(0)*b_dim+get_local_id(0);
	shared_in = get_local_id(1)*(b_dim+1)+get_local_id(0);
	shared_out = get_local_id(0)*(b_dim+1)+get_local_id(1);
	
	for (int k = 0 ; k < h_stride ; k += b_dim)
	{
		share[shared_in] = y[global_in];
		global_in += b_dim*l_stride;
		
		barrier(CLK_LOCAL_MEM_FENCE);
		
        if (global_out < m) {
		    x[global_out] = share[shared_out];
        }
		global_out += b_dim;

		barrier(CLK_LOCAL_MEM_FENCE);
	}		
}


// Partitioned solver with tiled diagonal pivoting
// real_elem to hold the type of sgema (it is necesary for complex variants
kernel void tiled_diag_pivot_x1(global elem* x,
                                global elem* w,          // left halo
                                global elem* v,          // right halo
                                global elem* b_buffer,   // modified main diag
                                global uchar* flag,      // buffer to tag pivot
                                global const elem* a,    // lower diag
                                global const elem* b,    // main diag
                                global const elem* c,    // upper diag
                                const int stride,
                                const int tile)                                    
{
	int b_dim;
	int ix;
	int bx;
	
	bx = get_group_id(0);
	b_dim = get_local_size(0);
	ix = bx*stride*b_dim+get_local_id(0);

	
	int k = 0;
	elem b_k,b_k_1,a_k_1,c_k,c_k_1,a_k_2;
	elem x_k,x_k_1;
	elem w_k,w_k_1;
	elem v_k_1;
	
	real kia = (sqrt(5.0)-1.0)/2.0;
	b_k = b[ix];
	c_k = c[ix];
    x_k = x[ix];
	w_k = a[ix];
	
	a_k_1 = a[ix+b_dim];
	b_k_1 = b[ix+b_dim];
	c_k_1 = c[ix+b_dim];
    x_k_1 = x[ix+b_dim];
	
	a_k_2 = a[ix+2*b_dim];
		
	// forward
	for (int i = 1 ; i <= tile ; i++)
	{
		while (k < (stride * i) / tile)
		{        
			real sgema;
			
			// math.h has an intrinsics for float, double 
            sgema = max( clAbs(c_k), clAbs(a_k_1));
			sgema = max( sgema, clAbs(b_k_1));
			sgema = max( sgema, clAbs(c_k_1));
			sgema = max( sgema, clAbs(a_k_2));			
			
			if(clAbs(b_k) * sgema >= kia * clAbs(c_k) * clAbs(a_k_1))
			{    
                elem b_inv = clInv(b_k);
				// write back
				flag[ix] = true;
				
				x_k = clMul(x_k, b_inv);
				w_k = clMul(w_k, b_inv);
				
				x[ix] = x_k;		//k
				w[ix] = w_k;		//k
				b_buffer[ix] = b_k;

				barrier(CLK_GLOBAL_MEM_FENCE);

				if( k < stride-1)
				{
					ix += b_dim;
					// update					        
                    x_k = clFma(-a_k_1, x_k, x_k_1);                // k+1
					w_k = clMul(-a_k_1, w_k);                       // k+1                    					
                    b_k = clFma(-a_k_1, clMul(c_k , b_inv), b_k_1); // k+1
					
					if( k < stride-2)				
					{
						// load new data
						b_k_1 = b[ix + b_dim];  // k+2
						a_k_1 = a_k_2;		    // k+2
                        x_k_1 = x[ix + b_dim];
						c_k   = c_k_1;		    // k+1
						c_k_1 = c[ix + b_dim];  // k+2
						
						a_k_2 = k < (stride - 3) ? a[ix + 2 * b_dim] : clReal(0); // k+3
					}
					else // k = stride -2
					{
						b_k_1 = clReal(0);
						a_k_1 = clReal(0);
						x_k_1 = clReal(0);
						c_k   = clReal(0);
						c_k_1 = clReal(0);
						a_k_2 = clReal(0);
					}
				}
				else // k = stride -1
				{
					v[ix] = clMul(c[ix], b_inv);
					barrier(CLK_GLOBAL_MEM_FENCE);
					ix   += b_dim;
				}
				
				k += 1;
			}
			else
			{		
				elem delta = clInv(clFma(b_k, b_k_1, -clMul(c_k,a_k_1)));

                x[ix] = clMul(clFma(x_k, b_k_1, -clMul(c_k, x_k_1)), delta);
				w[ix] = clMul(w_k, clMul(b_k_1, delta));
				b_buffer[ix] = b_k;
				flag[ix] = false;

				barrier(CLK_GLOBAL_MEM_FENCE);
				
                x_k_1 = clFma(b_k, x_k_1, -clMul(a_k_1, x_k)); //k+1
                x_k_1 = clMul(x_k_1, delta);
				w_k_1 = clMul(clMul(-a_k_1, w_k), delta);	  //k+1
				
				x[ix + b_dim]        = x_k_1;	  //k+1
				w[ix + b_dim]        = w_k_1;	  //k+1
				b_buffer[ix + b_dim] = b_k_1;
				flag[ix + b_dim] = false;

				barrier(CLK_GLOBAL_MEM_FENCE);
				
				if (k < stride - 2)
				{
					ix += 2 * b_dim;		
					// update
                    x_k = clFma(-a_k_2, x_k_1, x[ix]); // k+2
					w_k = clMul(-a_k_2, w_k_1);        // k+2
                    b_k = clMul(clMul(a_k_2, b_k), clMul(c_k_1, delta)); // k+2
                    //b_k -= b[ix]; // Wrong?
					b_k = b[ix] - b_k;
					
					if(k < stride - 3)
					{
						// load new data
						c_k = c[ix];     //k+2
						b_k_1 = b[ix + b_dim];  //k+3
						a_k_1 = a[ix + b_dim];  //k+3
						c_k_1 = c[ix + b_dim];  //k_3
                        x_k_1 = x[ix + b_dim];  //k+3
						a_k_2 = k < stride - 4 ? a[ix + 2 * b_dim] : clReal(0);
					}
					else // k = stride - 3
					{
						b_k_1 = clReal(0);
						a_k_1 = clReal(0);
						c_k   = clReal(0);
						c_k_1 = clReal(0);
						x_k_1 = clReal(0);
						a_k_2 = clReal(0);
					}
				}
				else // k = stride - 2
				{
					
					elem v_temp;
					v_temp = clMul(c[ix + b_dim], delta);					
                    v[ix]  = clMul(v_temp, -c_k);
					v[ix + b_dim] = clMul(v_temp, b_k);
					barrier(CLK_GLOBAL_MEM_FENCE);
					ix += 2 * b_dim;
				}				
				k += 2;
			}
			
		}
	}
	
	// backward
	// last one
	k -= 1;
	ix -= b_dim;
	if (flag[ix])
	{
		x_k_1 = x[ix];
		w_k_1 = w[ix];
		v_k_1 = v[ix];
		k -= 1;
	}
	else // 2-by-2
	{
		ix -= b_dim;
		x_k_1 = x[ix];
		w_k_1 = w[ix];
		v_k_1 = v[ix];
		k -= 2;
	}
	ix -= b_dim;
	
	for (int i = tile - 1 ; i >= 0 ; i--)
	{
		while( k >= (i * stride) / tile)
		{
			if (flag[ix]) // 1-by-1
			{
				c_k = c[ix];
				b_k = b_buffer[ix];				
                
                elem tempDiv = clDiv(-c_k, b_k);
                x_k_1 = clFma(x_k_1, tempDiv, x[ix]);                				
                w_k_1 = clFma(w_k_1, tempDiv, w[ix]);                
				v_k_1 = clMul(v_k_1, tempDiv);
                
				x[ix] = x_k_1;
				w[ix] = w_k_1;
				v[ix] = v_k_1;

				barrier(CLK_GLOBAL_MEM_FENCE);

				k -= 1;
			}
			else {
			
				elem delta;
				b_k   = b_buffer[ix - b_dim];
				c_k   = c[ix - b_dim];
				a_k_1 = a[ix];
				b_k_1 = b_buffer[ix];
				c_k_1 = c[ix];
                delta = clFma(b_k, b_k_1, -clMul(c_k,a_k_1));
				delta = clDiv(clReal(1), delta);
                
                elem prod = clMul(c_k_1, clMul(b_k, delta));
                
				x[ix] = clFma(-x_k_1, prod, x[ix]);
				w[ix] = clFma(-w_k_1, prod, w[ix]);
				v[ix] = clMul(-v_k_1, prod);

				barrier(CLK_GLOBAL_MEM_FENCE);
				
                ix  -= b_dim;
                prod = clMul(c_k_1, clMul(c_k, delta));
                
				x_k_1 = clFma(x_k_1, prod, x[ix]);
				w_k_1 = clFma(w_k_1, prod, w[ix]);
				v_k_1 = clMul(v_k_1, prod);
				x[ix] = x_k_1;
				w[ix] = w_k_1;
				v[ix] = v_k_1;

				barrier(CLK_GLOBAL_MEM_FENCE);
				
				k -= 2;
			}
			ix -= b_dim;
		}		
	}	
}

// SPIKE solver within a thread block for 1x rhs
kernel void spike_local_reduction_x1(global elem* x,
                                     global elem* w,
                                     global elem* v,
                                     global elem* x_mirror,
                                     global elem* w_mirror,
                                     global elem* v_mirror,
									 local elem* shared,
                                     const int stride) // stride per thread
{
	int tx = get_local_id(0);
	int b_dim = get_local_size(0);
	int bx = get_group_id(0);
        
	local elem* sh_w = shared;				
	local elem* sh_v = sh_w + 2 * b_dim;				
	local elem* sh_x = sh_v + 2 * b_dim;			
	
	//a ~~ w
	//b ~~ I
	//c ~~ v
	//d ~~ x
	
	int base = bx * stride * b_dim;
	
	// load halo to scratchpad
	sh_w[tx] = w[base + tx];
	sh_w[tx + b_dim] = w[base + tx + (stride - 1) * b_dim];
	sh_v[tx] = v[base + tx];
	sh_v[tx + b_dim] = v[base + tx + (stride - 1) * b_dim];
	sh_x[tx] = x[base + tx];
	sh_x[tx + b_dim] = x[base + tx + (stride - 1) * b_dim];
	
	barrier(CLK_LOCAL_MEM_FENCE);

	if (tx == 0 && bx == 0) printf("sh_x[%d] = %f", tx, sh_x[tx]);
	
	int scaler = 2;
	{
	    int index;
        int up_index;
        int down_index;
		index = scaler * tx + scaler / 2 - 1;
		up_index= scaler * tx;
		down_index = scaler * tx + scaler - 1;
		if (scaler == 2 && bx == 0 && (tx < b_dim / scaler) && tx == 49) printf("tx = %d\tsh_x[index + b_dim] = %f\tsh_x[index + 1] = %f\tsh_w[index + 1] = %f\tsh_w[index + b_dim] = %f", tx, sh_x[index + b_dim], sh_x[index + 1], sh_w[index + 1], sh_w[index + b_dim]);
		if (scaler == 2 && bx == 0 && (tx < b_dim / scaler) && tx == 49) printf("tx = %d\tsh_v[index + b_dim] = %f\tsh_v[index + 1] = %f\tsh_x[up_index] = %f\tsh_x[down_index + b_dim] = %f", tx, sh_v[index + b_dim], sh_v[index + 1], sh_x[up_index], sh_x[down_index + b_dim]);
		if (scaler == 2 && bx == 0 && (tx < b_dim / scaler) && tx == 49) printf("tx = %d\tsh_w[up_index] = %f\tsh_v[up_index] = %f\tsh_v[down_index + b_dim] = %f\tsh_w[down_index + b_dim] = %f", tx, sh_w[up_index], sh_v[up_index], sh_v[down_index + b_dim], sh_w[down_index + b_dim]);

		if (tx < b_dim / scaler && ((index + b_dim >= 2 * b_dim) || (index + 1 >= 2 * b_dim) || (up_index >= 2 * b_dim) || (down_index + b_dim >= 2 * b_dim))) printf("%d will overindex in group %d", tx, bx);
	}

	if (tx == 0 && bx == 0) printf("0 * 0 = %f", clMul(0.0f, 0.0f));
	
	while (scaler <= b_dim)
	{
        int index;
        int up_index;
        int down_index;
		elem det;

		if (tx < b_dim / scaler)
		{	
			index = scaler * tx + scaler / 2 - 1;
			up_index= scaler * tx;
			down_index = scaler * tx + scaler - 1;
			det = clReal(1);
			det = clFma(-sh_v[index + b_dim], sh_w[index + 1], det);
			if (tx == 49) det = 0;
			else det = clDiv(clReal(1), det);
			if (scaler == 2 && bx == 0 && (tx < b_dim / scaler) && tx == 49) printf("det = %d", det);
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		elem temp1, temp2, temp3, temp4;

		if (tx < b_dim / scaler)
		{
			elem d1,d2;
			d1 = sh_x[index + b_dim];
			d2 = sh_x[index + 1];

			if (scaler == 2 && bx == 0 && (tx < b_dim / scaler) && tx == 49) printf("d1 = %f\td2 = %f\tsh_v[index + b_dim] = %f", d1, d2, sh_v[index + b_dim]);
			if (scaler == 2 && bx == 0 && (tx < b_dim / scaler) && tx == 49) printf("index + b_dim = %d\tindex + 1 = %d\tup_index = %d\tdown_index + b_dim = %d", index + b_dim, index + 1, up_index, down_index + b_dim);

            temp1 = clMul(clFma(sh_v[index + b_dim], -d2, d1), det);
            temp2 = clMul(clFma(sh_w[index + 1], -d1, d2), det);			
            temp3 = clMul(sh_w[index + b_dim], clMul(sh_w[index + 1], -det));	            
			temp4 = clMul(sh_w[index + b_dim], det);

			if (scaler == 2 && bx == 0 && (tx < b_dim / scaler) && tx == 49) printf("tx = %d\ttemp1 = %f\ttemp2 = %f\ttemp3 = %f\ttemp4 = %f\tclFma(sh_v[index + b_dim], -d2, d1) = %f\tclMul(..., det) = %f", tx, temp1, temp2, temp3, temp4, clFma(sh_v[index + b_dim], -d2, d1), clMul(clFma(sh_v[index + b_dim], -d2, d1), det));
		}
		barrier(CLK_LOCAL_MEM_FENCE); // remove with printf
		if (tx < b_dim / scaler)
		{
			sh_x[index + b_dim] = temp1;
			sh_x[index + 1]     = temp2;
			sh_w[index + 1]     = temp3;
			sh_w[index + b_dim] = temp4;
		}
		barrier(CLK_LOCAL_MEM_FENCE); // remove with printf
		if (scaler == 2 && bx == 0 && (tx < b_dim / scaler) && tx == 49) printf("tx = %d\tsh_x[index + b_dim] = %f\tsh_x[index + 1] = %f\tsh_w[index + 1] = %f\tsh_w[index + b_dim] = %f", tx, sh_x[index + b_dim], sh_x[index + 1], sh_w[index + 1], sh_w[index + b_dim]);
		if (scaler == 2 && bx == 0 && (tx < b_dim / scaler) && tx == 49) printf("tx = %d\tsh_v[index + b_dim] = %f\tsh_v[index + 1] = %f\tsh_x[up_index] = %f\tsh_x[down_index + b_dim] = %f", tx, sh_v[index + b_dim], sh_v[index + 1], sh_x[up_index], sh_x[down_index + b_dim]);
		if (scaler == 2 && bx == 0 && (tx < b_dim / scaler) && tx == 49) printf("tx = %d\tsh_w[up_index] = %f\tsh_v[up_index] = %f\tsh_v[down_index + b_dim] = %f\tsh_w[down_index + b_dim] = %f", tx, sh_w[up_index], sh_v[up_index], sh_v[down_index + b_dim], sh_w[down_index + b_dim]);
		if (tx < b_dim / scaler)
		{
			sh_v[index + b_dim] = clMul(sh_v[index + b_dim], clMul(sh_v[index + 1], -det));            
			sh_v[index + 1]     = clMul(sh_v[index + 1], det);
			
			//boundary
            sh_x[up_index] 		     = clFma(sh_x[index + 1], -sh_v[up_index], sh_x[up_index]);
			sh_x[down_index + b_dim] = clFma(sh_x[index + b_dim], -sh_w[down_index + b_dim], sh_x[down_index + b_dim]);
            
            sh_w[up_index] = clFma(sh_w[index + 1], -sh_v[up_index], sh_w[up_index]);
			sh_v[up_index] = clMul(-sh_v[index + 1], sh_v[up_index]);

            sh_v[down_index + b_dim] = clFma(sh_v[index + b_dim], sh_w[down_index + b_dim], sh_v[down_index + b_dim]);
            sh_w[down_index + b_dim] = clMul(sh_w[index + b_dim], sh_w[down_index + b_dim]);
			
		}

		barrier(CLK_LOCAL_MEM_FENCE); // remove with printf
		//if (scaler * tx == 0 && (isnan(sh_x[scaler * tx + scaler / 2 - 1 + 1]) || isnan(sh_v[scaler * tx]))) printf("tx = %d\tindex = %d\tscaler = %d\tsh_x[index + 1] = %f\t-sh_v[up_index] = %f\tsh_x[up_index] = %f", tx, scaler * tx + scaler / 2 - 1, scaler, sh_x[scaler * tx + scaler / 2 - 1 + 1], -sh_v[scaler * tx], sh_x[scaler * tx]);

		if (scaler == 2 && bx == 0 && (tx < b_dim / scaler) && tx == 49) printf("tx = %d\tsh_x[index + b_dim] = %f\tsh_x[index + 1] = %f\tsh_w[index + 1] = %f\tsh_w[index + b_dim] = %f", tx, sh_x[index + b_dim], sh_x[index + 1], sh_w[index + 1], sh_w[index + b_dim]);
		if (scaler == 2 && bx == 0 && (tx < b_dim / scaler) && tx == 49) printf("tx = %d\tsh_v[index + b_dim] = %f\tsh_v[index + 1] = %f\tsh_x[up_index] = %f\tsh_x[down_index + b_dim] = %f", tx, sh_v[index + b_dim], sh_v[index + 1], sh_x[up_index], sh_x[down_index + b_dim]);
		if (scaler == 2 && bx == 0 && (tx < b_dim / scaler) && tx == 49) printf("tx = %d\tsh_w[up_index] = %f\tsh_v[up_index] = %f\tsh_v[down_index + b_dim] = %f\tsh_w[down_index + b_dim] = %f", tx, sh_w[up_index], sh_v[up_index], sh_v[down_index + b_dim], sh_w[down_index + b_dim]);

		scaler *= 2;

		barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	//write out
	w[base + tx] = sh_w[tx];
	w[base + tx + (stride - 1) * b_dim] = sh_w[tx + b_dim];
	
	v[base + tx] = sh_v[tx];
	v[base + tx + (stride - 1) * b_dim] = sh_v[tx + b_dim];

	if (tx == 0 && bx == 0) printf("base = %d\ttx = %d\tstride = %d\tb_dim = %d\tsh_x[tx] = %f", base, tx, stride, b_dim, sh_x[tx]);
	
	x[base + tx] = sh_x[tx];
	x[base + tx + (stride - 1) * b_dim] = sh_x[tx + b_dim];
	
	//write mirror
	if(tx < 1)
	{
		int g_dim = get_global_size(0);
		w_mirror[bx] = sh_w[0];
		w_mirror[g_dim + bx] = sh_w[2 * b_dim - 1];
		
		v_mirror[bx] = sh_v[0];
		v_mirror[g_dim + bx] = sh_v[2 * b_dim - 1];
		
		x_mirror[bx] = sh_x[0];
		x_mirror[g_dim + bx] = sh_x[2 * b_dim - 1];
	}

	barrier(CLK_LOCAL_MEM_FENCE); // remove with printf
	if (tx == 0 && bx == 0) printf("sh_x[%d] = %f", tx, sh_x[tx]);
}

///////////////////////////
/// a global level SPIKE solver for oneGPU
/// One block version
///
////////////////////
kernel void spike_global_solving_x1(global elem* x,
                                        global elem* w,
                                        global elem* v,
                                        local elem* shared,
                                        const int len)
{
	int ix = get_local_id(0);
	int b_dim = get_local_size(0);
	
	local elem* sh_w = shared;
	local elem* sh_v = sh_w + 2 * len;
	local elem* sh_x = sh_v + 2 * len;

	
	//a ~~ w
	//b ~~ I
	//c ~~ v
	//d ~~ x
	
	// read data
	while (ix < len)
	{
		sh_w[ix] = w[ix];
		sh_w[ix + len] = w[ix + len];
		
		sh_v[ix] = v[ix];
		sh_v[ix + len] = v[ix + len];
		
		sh_x[ix] = x[ix];
		sh_x[ix + len] = x[ix + len];
		
		ix += b_dim;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
	int scaler = 2;
	while (scaler <= len)
	{
		ix = get_local_id(0);
		while (ix < len / scaler)
		{
			int index = scaler * ix + scaler / 2 - 1;;
			int up_index = scaler * ix;
			int down_index = scaler * ix + scaler - 1;

			elem det = clInv(clFma(-sh_v[index + len], sh_w[index + 1], clReal(1)));
            
			elem d1,d2;
			d1 = sh_x[index + len];
			d2 = sh_x[index + 1];
			
            sh_x[index + len] = clMul(clFma(sh_v[index + len], -d2, d1), det);
            sh_x[index + 1]   = clMul(clFma(sh_w[index + 1], -d1, d2), det);
						
            sh_w[index + 1]   = clMul(sh_w[index + len], clMul(sh_w[index + 1], -det));
			sh_w[index + len] = clMul(sh_w[index + len], det);
									
            sh_v[index + len] = clMul(sh_v[index + len], clMul(sh_v[index + 1], -det));    
			sh_v[index + 1]   = clMul(sh_v[index + 1], det);
			
			//boundary
            sh_x[up_index] 		   = clFma(sh_x[index + 1], -sh_v[up_index], sh_x[up_index]); 
            sh_x[down_index + len] = clFma(sh_x[index + len], -sh_w[down_index+len], sh_x[down_index + len]);
						
            sh_w[up_index] = clFma(sh_w[index + 1], -sh_v[up_index], sh_w[up_index]);

            sh_v[up_index]         = clMul(-sh_v[index + 1], sh_v[up_index]);	
            sh_v[down_index + len] = clFma(sh_v[index + len], -sh_w[down_index + len], sh_v[down_index + len]);
            
            sh_w[down_index + len] = clMul( -sh_w[index + len], sh_w[down_index + len]);
			
			ix += b_dim;
			
		}

		scaler *= 2;
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	// backward reduction
	scaler = len / 2;
	while (scaler >= 2)
	{
		ix = get_local_id(0);
		while(ix < len/scaler)
		{
			int index = scaler * ix + scaler / 2 - 1;
			int up_index = scaler * ix - 1;
			int down_index = scaler * ix + scaler;

			up_index = up_index < 0 ? 0 : up_index;
			down_index = down_index < len ? down_index : len - 1;
			
            sh_x[index + len] = clFma(-sh_w[index + len], sh_x[up_index + len], sh_x[index + len]);
            sh_x[index + len] = clFma(-sh_v[index + len], sh_x[down_index], sh_x[index + len]);
            
			sh_x[index + 1]   = clFma(-sh_w[index + 1], sh_x[up_index + len], sh_x[index + 1]);
            sh_x[index + 1]   = clFma(-sh_v[index + 1], sh_x[down_index], sh_x[index + 1]);
            
			ix += b_dim;
		}

		scaler /= 2;
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	// write out
	ix = get_local_id(0);
	while (ix < len)
	{
		x[ix] = sh_x[ix];
		x[ix + len] = sh_x[ix + len];
		ix += b_dim;
	}
}

/// a thread-block level SPIKE solver 
kernel void spike_local_solving_x1(global elem* x,
                                       global const elem* w,
                                       global const elem* v,
                                       global const elem* x_mirror,
                                       local elem* shared,
                                       const int stride)
{
	int tx = get_local_id(0);
	int b_dim = get_local_size(0);
	int bx = get_group_id(0);
	
	local elem* sh_w = shared;				
	local elem* sh_v = sh_w + 2 * b_dim;				
	local elem* sh_x = sh_v + 2 * b_dim; //sh_x is 2*b_dim + 2
	
	//a ~~ w
	//b ~~ I
	//c ~~ v
	//d ~~ x
	
	int base = bx * stride * b_dim;
	
	// load halo to scratchpad
	sh_w[tx] = w[base + tx];
	sh_w[tx + b_dim] = w[base + tx + (stride - 1) * b_dim];
	sh_v[tx] = v[base + tx];
	sh_v[tx + b_dim] = v[base + tx + (stride - 1) * b_dim];
	
	// swap the order of x
	// why
	sh_x[tx + 1] = x[base + tx + (stride - 1) * b_dim];
	sh_x[tx + b_dim + 1] = x[base + tx];
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if (tx < 1)
	{
		int g_dim = get_global_size(0);
		sh_x[0] = bx > 0? x_mirror[bx - 1 + g_dim] : clReal(0);
		sh_x[2 * b_dim + 1] = bx < g_dim - 1 ? x_mirror[bx + 1] : clReal(0);
		
		sh_x[b_dim + 1] = x_mirror[bx];
		sh_x[b_dim] = x_mirror[bx + g_dim];		
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
	int scaler = b_dim;
	while (scaler >= 2)
	{
		if (tx < b_dim / scaler)
		{
			int index = scaler*tx+scaler/2-1;
			int up_index = scaler*tx;
			int down_index = scaler*tx + scaler+1;
	
			sh_x[index + 1] = clFma(sh_w[index + b_dim], -sh_x[up_index], sh_x[index + 1]);
            sh_x[index + 1] = clFma(sh_v[index + b_dim], -sh_x[down_index + b_dim], sh_x[index + 1]);
            
            sh_x[index + b_dim + 2] = clFma(sh_w[index + 1], -sh_x[up_index], sh_x[index + b_dim + 2]);
            sh_x[index + b_dim + 2] = clFma(sh_v[index + 1], -sh_x[down_index + b_dim], sh_x[index + b_dim + 2]);
		}

		scaler /= 2;
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	// write out
	x[base + tx] = sh_x[tx + b_dim + 1];
	x[base + tx + (stride - 1) * b_dim] = sh_x[tx + 1];
}

// backward substitution for SPIKE solver
kernel void spike_back_sub_x1(global elem* x,
                                  global const elem* w,
                                  global const elem* v,
                                  global const elem* x_mirror,
                                  const int stride)
{
	int tx = get_local_id(0);
	int b_dim = get_local_size(0);
	int bx = get_group_id(0);

	int base = bx * stride * b_dim;
	elem x_up, x_down;
	
	if (tx > 0 && tx < b_dim - 1)
	{
		x_up   = x[base + tx - 1 + (stride - 1) * b_dim];
		x_down = x[base + tx + 1];
	}
	else
	{
		int g_dim = get_global_size(0);
		if (tx == 0)
		{
			x_up   = bx > 0 ? x_mirror[bx - 1 + g_dim] : clReal(0);
			x_down = x[base + tx + 1];
		}
		else
		{
			x_up   = x[base + tx - 1 + (stride - 1) * b_dim];
			x_down = bx < g_dim - 1 ? x_mirror[bx + 1] : clReal(0);
		}
	}
	
	for(int k = 1 ; k < stride - 1 ; k++)
	{        
        x[base + tx + k * b_dim] = clFma( w[base + tx + k * b_dim], -x_up,   x[base + tx + k * b_dim]);
        x[base + tx + k * b_dim] = clFma( v[base + tx + k * b_dim], -x_down, x[base + tx + k * b_dim]);
	}
}
