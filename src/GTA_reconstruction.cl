/*
* @file         GTA_reconstruction.cl
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
__kernel void recon(__global int *index, __global  float *val, __global  int *core_index,
	__global float *core_val, __global  float *FactorM, __global  float *Error_T, int order, int nnz, int rank, int Core_N, int mult) {
	int pos = get_global_id(0);
	if (pos < nnz) {
		int i, j, k, l, pre2=pos*order, pre3, pre4;
		float res=0;
		int tmp2[25];
		for (i = 0; i < order; i++) tmp2[i] = index[pre2 + i];
		for (i = 0; i < Core_N; i++) {
			pre3 = 0; 
			pre4 = i*order;
			float temp = core_val[i];
			for (k = 0; k < order; k++) {
				temp *= FactorM[pre3 + tmp2[k]*rank + core_index[pre4 + k]];
				pre3 += mult;
			}
			res += temp;
		}
		Error_T[pos] = res;
	}
}
