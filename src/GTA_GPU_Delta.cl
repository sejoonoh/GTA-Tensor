/*
* @file         GTA_GPU_Delta.cl
* @main author  Sejoon Oh (ohhenrie@snu.ac.kr), Seoul National University
* @author       Namyong Park (namyongp@cs.cmu.edu), Carnegie Mellon University
* @author		Jun-gi Jang (elnino4@snu.ac.kr), Seoul National University
* @author       Lee Sael (saellee@gmail.com), Seoul National University
* @author       U Kang (ukang@snu.ac.kr), Seoul National University
* @version      1.0
* @date         2018-07-14
*
* A General Framework for Tucker Factorization on Heterogeneous Platforms
*
* This software is free of charge under research purposes.
* For commercial purposes, please contact the main author.
*
*/
__kernel void compute_delta(__global int *index, __global  float *val, __global  int *core_index,
	__global float *core_val, __global  float *FactorM, __global float *Delta, int order, int nnz, int rank, int current, int Core_N, int mult) {
	int pos = get_global_id(0);
	if (pos < nnz) {
		int i, j, k, l, pre_val = pos*rank, pre2=pos*order, pre3, pre4;
		float tmp[105]; //The size (target rank) must be properly adjusted to datasets
		int tmp2[25]; 
		for (i = 0; i < rank; i++) tmp[i] = 0;
		for (i = 0; i < order; i++) tmp2[i] = index[pre2+i];
		for (i = 0; i < Core_N; i++) {
			pre3 = 0; pre4 = i*order;
			j = core_index[pre4 + current];
			float temp = core_val[i];
			for (k = 0; k < order; k++) {
				if (k != current) {
					int row = tmp2[k], col = core_index[pre4 + k];
					temp *= FactorM[pre3 + row*rank + col];
				}
				pre3 += mult;
			}
			tmp[j] += temp;
		}
		for (i = 0; i < rank; i++) Delta[pre_val + i] = tmp[i];
	}
}
