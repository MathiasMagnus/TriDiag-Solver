// Data layout transformation for inputs
kernel void foward_marshaling_bxb(global elem* x,
                                  global const elem* y,
								  local elem* share,
                                  const int h_stride,
                                  const int l_stride,
                                  int m,
                                  elem pad)
{	
	int b_dim;

	int global_in;
	int global_out;
	int shared_in;
	int shared_out;
		
	b_dim = blockDim.x; 	//16

	global_in = blockIdx.y*l_stride*h_stride + (blockIdx.x*b_dim+threadIdx.y)*h_stride+threadIdx.x;
	global_out = blockIdx.y*l_stride*h_stride + threadIdx.y*l_stride + blockIdx.x*b_dim+threadIdx.x;
	shared_in = threadIdx.y*(b_dim+1)+threadIdx.x;
	shared_out = threadIdx.x*(b_dim+1)+threadIdx.y;

	for (int k=0 ; k < h_stride ; k += b_dim)
	{	
		share[shared_in] = global_in >= m ? pad : y[global_in];
		global_in += b_dim;
		
		barrer(CLK_LOCAL_MEM_FENCE);
		
		x[global_out] = share[shared_out];
		global_out += b_dim * l_stride;

		barrer(CLK_LOCAL_MEM_FENCE);
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
	
	b_dim = blockDim.x; 	//16

	global_out = blockIdx.y*l_stride*h_stride +  (blockIdx.x*b_dim+threadIdx.y)*h_stride+threadIdx.x;
	global_in = blockIdx.y*l_stride*h_stride + threadIdx.y*l_stride + blockIdx.x*b_dim+threadIdx.x;
	shared_in = threadIdx.y*(b_dim+1)+threadIdx.x;
	shared_out = threadIdx.x*(b_dim+1)+threadIdx.y;
	
	for (int k = 0 ; k < h_stride ; k += b_dim)
	{
		share[shared_in] = y[global_in];
		global_in += b_dim*l_stride;
		
		barrer(CLK_LOCAL_MEM_FENCE);
		
        if (global_out < m) {
		    x[global_out] = share[shared_out];
        }
		global_out += b_dim;

		barrer(CLK_LOCAL_MEM_FENCE);
	}		
}


// Partitioned solver with tiled diagonal pivoting
// real_elem to hold the type of sgema (it is necesary for complex variants
kernel void tiled_diag_pivot_x1(global elem* x,
                                global elem* w,          // left halo
                                global elem* v,          // right halo
                                global elem* b_buffer,   // modified main diag
                                global bool* flag,       // buffer to tag pivot
                                global const elem* a,    // lower diag
                                global const elem* b,    // main diag
                                global const elem* c,    // upper diag
                                const int stride,
                                const int tile)                                    
{
	int b_dim;
	int ix;
	int bx;
	
	bx = blockIdx.x;
	b_dim = blockDim.x;
	ix = bx*stride*b_dim+threadIdx.x;

	
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
				
                x_k_1 = clFma(b_k, x_k_1, -cuMul(a_k_1, x_k)); //k+1
                x_k_1 = clMul(x_k_1, delta);
				w_k_1 = clMul(cuMul(-a_k_1, w_k), delta);	  //k+1
				
				x[ix + b_dim]        = x_k_1;	  //k+1
				w[ix + b_dim]        = w_k_1;	  //k+1
				b_buffer[ix + b_dim] = b_k_1;				
				flag[ix + b_dim] = false;	
				
				if (k < stride - 2)
				{
					ix += 2 * b_dim;		
					// update
                    x_k = clFma(-a_k_2, x_k_1, x[ix]); // k+2
					w_k = clMul(-a_k_2, w_k_1);        // k+2
                    b_k = clMul(clMul(a_k_2, b_k), clMul(c_k_1, delta)); // k+2
                    b_k -= b[ix];
					
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
				k -= 1;
			}
			else {
			
				elem delta;
				b_k   = b_buffer[ix - b_dim];
				c_k   = c[ix - b_dim];
				a_k_1 = a[ix];
				b_k_1 = b_buffer[ix];
				c_k_1 = c[ix];
                delta = clFma(b_k, b_k_1, -cuMul(c_k,a_k_1));
				delta = clDiv(clReal(1), delta);
                
                elem prod = clMul(c_k_1, clMul(b_k, delta));
                
				x[ix] = clFma(-x_k_1, prod, x[ix]);
				w[ix] = clFma(-w_k_1, prod, w[ix]);
				v[ix] = clMul(-v_k_1, prod);
				
                ix  -= b_dim;
                prod = clMul(c_k_1, clMul(c_k, delta));
                
				x_k_1 = clFma(x_k_1, prod, x[ix]);
				w_k_1 = clFma(w_k_1, prod, w[ix]);
				v_k_1 = clMul(v_k_1, prod);
				x[ix] = x_k_1;
				w[ix] = w_k_1;
				v[ix] = v_k_1;
				k -= 2;
			}
			ix -= b_dim;
		}		
	}	
}