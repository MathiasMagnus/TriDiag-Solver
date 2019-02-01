#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

kernel void find_crossings(real threshold,
                           size_t out_size,
                           global real* input,
                           global usnigned int* counter,
						   global size_t* index,
						   global bool* up)
{
    size_t gid = get_global_id(0);

	real in[2] = { input[gid], input[gid + 1] };

	if (in[0] <= thershold && in[1] >= threshold) // cross up
	{
		unsigned int my_counter = atom_inc(counter);

		if (my_counter < out_size)
		{
			index[my_counter] = gid;
			up[my_counter] = true;
		}
	}
	if (in[0] >= thershold && in[1] <= threshold) // cross down
	{
		unsigned int my_counter = atom_inc(counter);

		index[my_counter] = gid;
		up[my_counter] = false;
	}
}

kernel void find_peaks(global real* input,
                       global size_t* index,
					   global bool* up)
{
	
}