/*
* @file			GTA.cpp
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
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include "omp.h"
#include <time.h>
#include <Eigen/Dense>
using namespace std;
using namespace Eigen;
typedef Matrix<double, Dynamic, Dynamic> MatrixXdd;
#define lambda 0.001

struct GPU_objects {
	cl_platform_id platform;
	cl_device_id device;
	cl_device_id *devices;
	cl_context context;
	cl_command_queue queue;
	cl_command_queue *queues;
	cl_program program;
	char *kernel_source;
	size_t kernel_source_size;
	cl_kernel kernel;
	cl_kernel *kernels;
	cl_int err;
	cl_uint num_devices;
}TF,RECON;

struct Tensor{
	int order;											// Tensor order (e.g., 5)
	int nonzeros;										// Number of nonzeros in a tensor (e.g., 1000000)
	int gpu_mode;										// CPU:0, Single-GPU: 1, Multi-GPU: 2
	int partially_observed;								// 0: fully observed, 1: partially observed (default)
	int local_size;
	int *dimension;										// Tensor dimensionality (e.g., 100x100x100)
	//FOR COO format
	int *index;											// Indices of a tensor (e.g., (1,2,3))
	float *value;										// Values of a tensor (e.g., 4.5)
	//FOR CS-N format
	float *FactorM;
}X,CoreT;

#define CHECK_ERROR(err) \
  if (err != CL_SUCCESS) { \
    printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
    exit(EXIT_FAILURE); \
  }

double ffrand(double x, double y) {//return the random value in (x,y) interval
	return ((y - x)*((double)rand() / RAND_MAX)) + x;
}
double abss(double x) {
	return x > 0 ? x : -x;
}

double Fit, pFit = -1;
int max_dim;
int *WhereX, *CountX;					        // WhereX[n][I_n] contains all entries of a tensor X whose nth mode's index is I_n

double copy_time, gpu_time;

int rrank,l_size,gpu_mode,nnz_mode;
char* InputPath;
char* ResultPath;

/////////////////////////////////        OpenCL-related Variables            /////////////////////////////////
cl_context context;
cl_command_queue *queues;
cl_program program;
char *kernel_source;
size_t kernel_source_size;
cl_kernel *kernels;
cl_int err;
cl_uint num_devices;
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

//[Input] Source code
//[Output] Source code in a character format
//[Function] Convert source code into a string
char * get_source_code(const char *file_name, size_t *len) {
	char *source_code;
	size_t length;
	FILE *file = fopen(file_name, "rb");
	if (file == NULL) {
		printf("[%s:%d] Failed to open %s\n", __FILE__, __LINE__, file_name);
		exit(EXIT_FAILURE);
	}

	fseek(file, 0, SEEK_END);
	length = (size_t)ftell(file);
	rewind(file);

	source_code = (char *)malloc(length + 1);
	fread(source_code, length, 1, file);
	source_code[length] = '\0';

	fclose(file);

	*len = length;
	return source_code;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//[Input] Existing contents of a tensor
//[Output] Tensor in a structured form 
//[Function] Constructor function
void Construct_Core_Tensor(int ord, int nnz, int* dim, int* ind, float* val) {
	CoreT.order = ord; CoreT.nonzeros = nnz; CoreT.dimension = dim; CoreT.index = ind; CoreT.value = val;
}
//[Input] Path of a tensor and its metadata
//[Output] Tensor in a structured form 
//[Function] Constructor function
void Read_Tensor(char* Path) {
	printf("Reading Input Tensor......\n");
	FILE* fin = fopen(Path, "r");
	FILE* fin2 = fopen(Path,"r");
	char tmp[1005];
	int i, j,k;
	float v;
	int pos = 0,len;
	X.nonzeros=X.order=0;
	while(fgets(tmp,1005,fin2)){
		X.nonzeros++;
		len = strlen(tmp);
		if(X.nonzeros==1){
			for(i=0;i<len;i++){
				if(tmp[i]==' ' || tmp[i]=='\t'){
					X.order++;
				}
			}
		}
	}
	X.dimension = (int *)malloc(sizeof(int)*X.order);
	X.index = (int *)malloc(sizeof(int)*X.order*X.nonzeros);
	X.value = (float *)malloc(sizeof(float)*X.nonzeros);
	X.gpu_mode = 0;
	for (i = 0; i < X.order; i++) X.dimension[i] = 0;
	for (i = 0; i < X.nonzeros; i++) {
		fgets(tmp, 1005, fin);
		len = strlen(tmp);
		int k = 0, idx = 0, flag = 0;
		double mul = 0.1, val = 0;
		for (j = 0; j < len; j++) {
			if (tmp[j] == ' ' || tmp[j] == '\t') {
				X.index[pos++] = idx - 1;
				if (X.dimension[k] < idx) X.dimension[k] = idx;
				idx = 0;
				k++;
			}
			else if (tmp[j] >= '0' && tmp[j] <= '9') {
				if (flag == 1) {
					val += mul*(tmp[j] - '0');
					mul /= 10;
				}
				else idx = idx * 10 + tmp[j] - '0';
			}
			else if (tmp[j] == '.') {
				val += idx;
				flag = 1;
			}
		}
		if(flag==0) val = idx;
		X.value[i] = val;
	}
	X.partially_observed = nnz_mode; 
	X.local_size = l_size; 
	X.gpu_mode = gpu_mode;
	printf("Reading DONE!\n\n");
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//[Input] GPU mode and kernel information
//[Output] Initialized OpenCL variables
//[Function] Setup and initialized Opencl-related variables
GPU_objects GPU_INIT(int gpu_mode, char* kernel_code, char* kernel_name) {
	GPU_objects obj;
	err = clGetPlatformIDs(1, &obj.platform, NULL); CHECK_ERROR(err);
	err = clGetDeviceIDs(obj.platform, CL_DEVICE_TYPE_GPU, 0, NULL, &obj.num_devices); CHECK_ERROR(err);
	printf("%u devices\n", obj.num_devices);
	obj.devices = (cl_device_id*)malloc(sizeof(cl_device_id) * obj.num_devices);
	obj.queues = (cl_command_queue*)malloc(sizeof(cl_command_queue) * obj.num_devices);
	obj.kernels = (cl_kernel*)malloc(sizeof(cl_kernel) * obj.num_devices);
	err = clGetDeviceIDs(obj.platform, CL_DEVICE_TYPE_GPU, obj.num_devices, obj.devices, NULL); CHECK_ERROR(err);
	obj.context = clCreateContext(NULL, obj.num_devices, obj.devices, NULL, NULL, &err); CHECK_ERROR(err);
	for (int i = 0; i < obj.num_devices; i++) {
		obj.queues[i] = clCreateCommandQueue(obj.context, obj.devices[i], 0, &err);	CHECK_ERROR(err);
	}
	obj.kernel_source = get_source_code(kernel_code, &obj.kernel_source_size);
	obj.program = clCreateProgramWithSource(obj.context, 1, (const char**)&obj.kernel_source, &obj.kernel_source_size, &err); CHECK_ERROR(err);
	err = clBuildProgram(obj.program, obj.num_devices, obj.devices, "", NULL, NULL);
	if (err == CL_BUILD_PROGRAM_FAILURE) {
		size_t log_size;
		char *log;
		err = clGetProgramBuildInfo(obj.program, obj.devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size); CHECK_ERROR(err);
		log = (char*)malloc(log_size + 1);
		err = clGetProgramBuildInfo(obj.program, obj.devices[0], CL_PROGRAM_BUILD_LOG, log_size, log, NULL); CHECK_ERROR(err);
		log[log_size] = '\0';
		printf("Compiler error:\n%s\n", log);
		free(log);
		exit(0);
	}
	for (int i = 0; i < obj.num_devices; i++) {
		obj.kernels[i] = clCreateKernel(obj.program, kernel_name, &err); CHECK_ERROR(err);
	}
	return obj;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//[Input] GPU mode and OpenCL variables
//[Output] Released OpenCL variables
//[Function] Release the corresponding OpenCL variables
void GPU_DONE(int mode,GPU_objects obj) {
	if (mode == 0) return;
	int i = 0;
	for (i = 0; i < obj.num_devices; i++) clReleaseKernel(obj.kernels[i]);
	free(obj.kernels);
	for (i = 0; i < obj.num_devices; i++) clReleaseCommandQueue(obj.queues[i]);
	free(obj.queues);
	free(obj.devices);
	clReleaseProgram(obj.program);
	clReleaseContext(obj.context);
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
void assign_index() {

	int order = X.order, nonzeros = X.nonzeros;
	int* indices = X.index;
	int *tempX = (int *)malloc(sizeof(int)*max_dim*order);
	int pos = 0, i, j, k, l;
	for (i = 0; i < order; i++) {
		for (j = 0; j < X.dimension[i]; j++) {
			CountX[i*max_dim + j] = tempX[i*max_dim + j] = 0;
		}
	}
	for (i = 0; i < nonzeros; i++) {
		for (j = 0; j < order; j++) {
			k = indices[pos++];
			CountX[j*max_dim + k]++;
			tempX[j*max_dim + k]++;
		}
	}
	pos = 0;
	int now = 0;
	for (i = 0; i < order; i++) {
		pos = i*max_dim;
		for (j = 0; j < X.dimension[i]; j++) {
			k = CountX[pos];
			CountX[pos] = now;
			tempX[pos++] = now;
			now += k;
		}
		CountX[pos] = now;
		tempX[pos] = now;
	}
	pos = 0;
	for (i = 0; i < nonzeros; i++) {
		for (j = 0; j < order; j++) {
			k = indices[pos++];
			int now = tempX[j*max_dim + k];
			WhereX[now] = i;
			tempX[j*max_dim + k]++;
		}
	}
	free(tempX);
}

void partially_observed_pre_process(double *pre_check, int order, int rrank, int mult, float *FactorM, int* dim) {
	int i, j, k, l;
	for (i = 0; i < order; i++) {
		for (j = 0; j < rrank; j++) {
			for (k = 0; k < rrank; k++) {
				int pos1 = i*mult + j, pos2 = i*mult + k, now = i*rrank*rrank + j*rrank + k;
				pre_check[now] = 0;
				for (l = 0; l < dim[i]; l++) {
					pre_check[now] += FactorM[pos1 + l*rrank] * FactorM[pos2 + l*rrank];
				}
			}
		}
	}
}


void Computing_Delta(float* Delta, int rrank, int i) {
	int order = X.order, nnz = X.nonzeros, Core_N = CoreT.nonzeros, j, k, l, ii, jj;
	int *dim = X.dimension;
	int mult = max_dim*rrank;
	double st = omp_get_wtime(),st2=omp_get_wtime();
	if (X.gpu_mode != 0) {
			int NNZ_PER_DEVICE = nnz / num_devices + 1, last;
			if (nnz%num_devices == 0) NNZ_PER_DEVICE--;
			last = nnz - NNZ_PER_DEVICE*(num_devices - 1);
			cl_mem *bufA, *bufB, *bufC, *bufD, *bufE, *bufF;
			bufA = (cl_mem*)malloc(sizeof(cl_mem) * num_devices); bufB = (cl_mem*)malloc(sizeof(cl_mem) * num_devices); bufC = (cl_mem*)malloc(sizeof(cl_mem) * num_devices);  bufD = (cl_mem*)malloc(sizeof(cl_mem) * num_devices);  bufE = (cl_mem*)malloc(sizeof(cl_mem) * num_devices); bufF = (cl_mem*)malloc(sizeof(cl_mem) * num_devices);
			for (int j = 0; j < num_devices; j++) {
				if (j != num_devices - 1) {
					bufA[j] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * NNZ_PER_DEVICE*order, NULL, &err); CHECK_ERROR(err); err = clEnqueueWriteBuffer(queues[j], bufA[j], CL_FALSE, 0, sizeof(int) * NNZ_PER_DEVICE*order, X.index + j*NNZ_PER_DEVICE*order, 0, NULL, NULL); CHECK_ERROR(err);
					bufB[j] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*NNZ_PER_DEVICE, NULL, &err); CHECK_ERROR(err); err = clEnqueueWriteBuffer(queues[j], bufB[j], CL_FALSE, 0, sizeof(float) * NNZ_PER_DEVICE, X.value + j*NNZ_PER_DEVICE, 0, NULL, NULL); CHECK_ERROR(err);
					bufF[j] = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*rrank*NNZ_PER_DEVICE, NULL, &err); CHECK_ERROR(err); err = clEnqueueWriteBuffer(queues[j], bufF[j], CL_FALSE, 0, sizeof(float)*NNZ_PER_DEVICE*rrank, (Delta + j*NNZ_PER_DEVICE*rrank), 0, NULL, NULL);
				}
				else {
					bufA[j] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * last*order, NULL, &err); CHECK_ERROR(err); err = clEnqueueWriteBuffer(queues[j], bufA[j], CL_FALSE, 0, sizeof(int) * last*order, X.index + j*NNZ_PER_DEVICE*order, 0, NULL, NULL); CHECK_ERROR(err);
					bufB[j] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*last, NULL, &err); CHECK_ERROR(err); err = clEnqueueWriteBuffer(queues[j], bufB[j], CL_FALSE, 0, sizeof(float) * last, X.value + j*NNZ_PER_DEVICE, 0, NULL, NULL); CHECK_ERROR(err);
					bufF[j] = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*rrank*last, NULL, &err); CHECK_ERROR(err); err = clEnqueueWriteBuffer(queues[j], bufF[j], CL_FALSE, 0, sizeof(float)*last*rrank, (Delta + j*NNZ_PER_DEVICE*rrank), 0, NULL, NULL);
				}
				bufC[j] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int)*Core_N*order, NULL, &err); CHECK_ERROR(err); err = clEnqueueWriteBuffer(queues[j], bufC[j], CL_FALSE, 0, sizeof(int) * Core_N*order, CoreT.index, 0, NULL, NULL); CHECK_ERROR(err);
				bufD[j] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*Core_N, NULL, &err); CHECK_ERROR(err); err = clEnqueueWriteBuffer(queues[j], bufD[j], CL_FALSE, 0, sizeof(float) * Core_N, CoreT.value, 0, NULL, NULL); CHECK_ERROR(err);
				bufE[j] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*order*max_dim*rrank, NULL, &err); CHECK_ERROR(err); err = clEnqueueWriteBuffer(queues[j], bufE[j], CL_FALSE, 0, sizeof(float) * order*max_dim*rrank, X.FactorM, 0, NULL, NULL); CHECK_ERROR(err);

			}
			
			for(int j=0;j<num_devices;j++){
					clFinish(queues[j]);
			}
			printf("CPU->GPU COPY TIME: %lf\n",omp_get_wtime()-st);
			copy_time += omp_get_wtime()-st;
			st = omp_get_wtime();

			for (int j = 0; j < num_devices; j++) {
				err = clSetKernelArg(kernels[j], 0, sizeof(cl_mem), &bufA[j]); CHECK_ERROR(err);
				err = clSetKernelArg(kernels[j], 1, sizeof(cl_mem), &bufB[j]); CHECK_ERROR(err);
				err = clSetKernelArg(kernels[j], 2, sizeof(cl_mem), &bufC[j]); CHECK_ERROR(err);
				err = clSetKernelArg(kernels[j], 3, sizeof(cl_mem), &bufD[j]); CHECK_ERROR(err);
				err = clSetKernelArg(kernels[j], 4, sizeof(cl_mem), &bufE[j]); CHECK_ERROR(err);
				err = clSetKernelArg(kernels[j], 5, sizeof(cl_mem), &bufF[j]); CHECK_ERROR(err);
				err = clSetKernelArg(kernels[j], 6, sizeof(cl_int), &order); CHECK_ERROR(err);
				if (j == num_devices - 1) {
					err = clSetKernelArg(kernels[j], 7, sizeof(cl_int), &last); CHECK_ERROR(err);
				}
				else {
					err = clSetKernelArg(kernels[j], 7, sizeof(cl_int), &NNZ_PER_DEVICE); CHECK_ERROR(err);
				}
				err = clSetKernelArg(kernels[j], 8, sizeof(cl_int), &rrank); CHECK_ERROR(err);
				err = clSetKernelArg(kernels[j], 9, sizeof(cl_int), &i); CHECK_ERROR(err);
				err = clSetKernelArg(kernels[j], 10, sizeof(cl_int), &Core_N); CHECK_ERROR(err);
				err = clSetKernelArg(kernels[j], 11, sizeof(cl_int), &mult); CHECK_ERROR(err);
			}
			size_t global_size = NNZ_PER_DEVICE;
			size_t local_size = X.local_size;
			global_size = (global_size + local_size - 1) / local_size * local_size;
			for (int j = 0; j < num_devices; j++) {
				err = clEnqueueNDRangeKernel(queues[j], kernels[j], 1, NULL, &global_size, &local_size, 0, NULL, NULL); CHECK_ERROR(err);
			}

			for(int j=0;j<num_devices;j++){
					clFinish(queues[j]);
			}
			printf("GPU COMPUTING TIME: %lf\n",omp_get_wtime()-st);
			gpu_time += omp_get_wtime()-st;
			st = omp_get_wtime();


			for (int j = 0; j < num_devices; j++) {
				if (j == num_devices - 1) {
					err = clEnqueueReadBuffer(queues[j], bufF[j], CL_TRUE, 0, sizeof(float)*last*rrank, Delta + j*NNZ_PER_DEVICE*rrank, 0, NULL, NULL); CHECK_ERROR(err);
				}
				else {
					err = clEnqueueReadBuffer(queues[j], bufF[j], CL_TRUE, 0, sizeof(float)*NNZ_PER_DEVICE*rrank, Delta + j*NNZ_PER_DEVICE*rrank, 0, NULL, NULL); CHECK_ERROR(err);
				}
			}


			for(int j=0;j<num_devices;j++){
					clFinish(queues[j]);
			}
			printf("GPU->CPU COPY TIME: %lf\n",omp_get_wtime()-st);
			copy_time += omp_get_wtime()-st;
			st = omp_get_wtime();


			for (int j = 0; j < num_devices; j++) {
				clReleaseMemObject(bufA[j]); clReleaseMemObject(bufB[j]);  clReleaseMemObject(bufC[j]);  clReleaseMemObject(bufD[j]); clReleaseMemObject(bufE[j]); clReleaseMemObject(bufF[j]);
			}
			free(bufA); free(bufB); free(bufC); free(bufD); free(bufE); free(bufF);
		}

	else {
		#pragma omp parallel for schedule(static)
		for (j = 0; j < nnz; j++) {
			int pre_val = j*order, pre_val2 = j*rrank, k, l, ii;
			int *cach1 = (int *)malloc(sizeof(int)*order);
			for (l = 0; l < order; l++) cach1[l] = X.index[pre_val + l];
			for (l = 0; l < rrank; l++) Delta[pre_val2 + l] = 0;
			for (l = 0; l < CoreT.nonzeros; l++) {
				int pre1 = l*order, pre2 = 0;
				int CorePos = CoreT.index[pre1 + i];
				double res = CoreT.value[l];
				for (ii = 0; ii < order; ii++) {
					if (ii != i) {
						int mulrow = cach1[ii], mulcol = CoreT.index[pre1];
						res *= X.FactorM[pre2 + mulrow*rrank + mulcol];
					}
					pre1++;
					pre2 += mult;
				}
				Delta[pre_val2 + CorePos] += res;
			}
			free(cach1);
		}
	}
	printf("Elapsed time for Computing Delta: %lf\n", omp_get_wtime() - st2);
}

void Computing_BC(float* Delta, double* B, double* C, int rank, int i) {
	int iter, row_size = X.dimension[i],mult = max_dim*rank;
	double st = omp_get_wtime();
	//Initialize B and C
	#pragma omp parallel for schedule(static) //in parallel
	for (int j = 0; j < row_size; j++) {
		int pos_B = j*rank*rank, pos_C = j*rank,k,l;
		for (k = 0; k < rank; k++) {
			for (l = 0; l < rank; l++) {
				B[pos_B] = 0;
				if (k == l) B[pos_B] = lambda;
				pos_B++;
			}
			C[pos_C++] = 0;
		}
	}

	#pragma omp parallel for schedule(dynamic) //in parallel
	for (iter = 0; iter < row_size; iter++) {
		int j=iter, k, l, ii, jj,pos_B,pos_C;
		int pos = i*max_dim + j;
		int nnz = CountX[pos + 1] - CountX[pos];
		pos = CountX[pos];
		for (k = 0; k < nnz; k++) { //Updating Delta, B, and C
			int current_input_entry = WhereX[pos + k];
			int pre_val = current_input_entry*rank;
			int now = 0;
			double Entry_val = X.value[current_input_entry];
			pos_B = j*rank*rank, pos_C = j*rank;
			for (ii = 0; ii < rank; ii++) {
				double cach = Delta[pre_val + ii];
				for (jj = 0; jj < rank; jj++) {
					B[pos_B++] += cach * Delta[pre_val + jj];
				}
				C[pos_C++] += cach * Entry_val;
			}
		}
	}
	printf("Elapsed time for Computing B and C: %lf\n", omp_get_wtime() - st);
}


void Update_Factor_Matrices(int rrank) {
	int order = X.order, nnz = X.nonzeros, Core_N = CoreT.nonzeros, i, j, k, l;
	int *dim = X.dimension;
	int *crows, rowcount, Core_dim;
	int mult = max_dim*rrank;
	double st = omp_get_wtime();
	float *Delta = (float *)malloc(sizeof(float)*nnz*rrank);
	double *pre_check = (double *)malloc(sizeof(double)*order*rrank*rrank);
	double *B = (double *)malloc(sizeof(double)*max_dim*rrank*rrank);
	double *C = (double *)malloc(sizeof(double)*max_dim*rrank);
	double *Shared_B = (double *)malloc(sizeof(double)*rrank*rrank);
	if (X.partially_observed == 0) {
		partially_observed_pre_process(pre_check, order, rrank, mult, X.FactorM, dim);
	}

	kernels = TF.kernels; queues = TF.queues; context = TF.context; 
	printf("INIT TIME: %lf\n", omp_get_wtime() - st);

	copy_time = gpu_time = 0;

	for (i = 0; i < order; i++) { //Updating the ith Factor Matrix
		int row_size = dim[i];
		int column_size = rrank;
		int iter;

		st = omp_get_wtime();
		Computing_Delta(Delta, rrank, i);

		long long sizee = Core_N*Core_N;
		if (X.partially_observed == 0) {
			double st2 = omp_get_wtime();
			for (j = 0; j < column_size; j++) {
				for (k = 0; k < column_size; k++) {
					double temp = 0;
					if (j == k) temp = lambda;
					Shared_B[j*column_size + k] = temp;
				}
			}
			long long j;
			for (j = 0; j < sizee; j++) {
				int alpha = j / Core_N, beta = j - alpha*Core_N, k, aa;
				double totalval = CoreT.value[alpha] * CoreT.value[beta];
				for (k = 0; k < order; k++) {
					if (k != i) {
						int pos3 = CoreT.index[alpha*order + k], pos4 = CoreT.index[beta*order + k], pos5 = k*mult;
						totalval *= pre_check[k*rrank*rrank + pos3*rrank + pos4];
					}
				}
				int pos1 = CoreT.index[alpha*order + i], pos2 = CoreT.index[beta*order + i];
				Shared_B[pos1*column_size + pos2] += totalval;
			}
			printf("Shared_B calculation time : %lf\n", omp_get_wtime() - st2);
		}

		Computing_BC(Delta, B, C, rrank, i);
		
		double st2 = omp_get_wtime();
		#pragma omp parallel for schedule(static) //in parallel
		for (iter = 0; iter < row_size; iter++) {
			//Getting the inverse matrix of [B+lambda*I]
			int pos, j, k, l;
			j = iter;
			int pos_B = j*column_size*column_size, pos_C = j*column_size;
			MatrixXdd AA(column_size, column_size);
			pos = 0;
			for (k = 0; k < column_size; k++) {
				for (l = 0; l < column_size; l++) {
					if (X.partially_observed == 1) AA(k, l) = B[pos_B+k*column_size + l];
					else AA(k, l) = Shared_B[k*column_size + l];
				}
			}

			MatrixXdd BB = AA.inverse();

			//Update the jth row of ith Factor Matrix 
			int cach = i*mult + j*column_size;
			for (k = 0; k < column_size; k++) {
				double res = 0;
				for (l = 0; l < column_size; l++) {
					res += C[pos_C + l] * BB(l , k);
				}
				X.FactorM[cach + k] = res;
			}
			AA.resize(0,0); BB.resize(0,0);
		}

		printf("UPDATE TIME : %lf\n",omp_get_wtime()-st2);
		if (X.partially_observed == 0) {
			for (j = 0; j < rrank; j++) {
				for (k = 0; k < rrank; k++) {
					int pos1 = i*mult + j, pos2 = i*mult + k, now = i*rrank*rrank + j*rrank + k;
					pre_check[now] = 0;
					for (l = 0; l < dim[i]; l++) {
						double a1 =  X.FactorM[pos1 + l*rrank], a2 =   X.FactorM[pos2 + l*rrank];
						pre_check[now] += a1*a2;
					}
				}
			}
		}

		printf("[FACTOR MATRIX %d] UPDATE time : %lf\n\n", i + 1, (omp_get_wtime() - st));
		
	}
	printf("copy time: %lf\tgpu time: %lf\n",copy_time,gpu_time);

	free(Delta); free(pre_check); free(Shared_B); free(B); free(C);
}

void Reconstruction(int rrank) {
	int order = X.order, nnz = X.nonzeros, Core_N = CoreT.nonzeros, i, j, k, l;
	int* dim = X.dimension;
	float *Error_T, Error = 0, NormX = 0;
	Error_T = (float *)malloc(sizeof(float)*nnz);
	double st = omp_get_wtime();
	int mult = max_dim*rrank;

	if (X.gpu_mode == 0) {
		#pragma omp parallel for schedule(static)
		for (i = 0; i < nnz; i++) {
			int j, pre_val = i*order;
			double ans = 0;
			int *cach1 = (int *)malloc(sizeof(int)*order);
			for (j = 0; j < order; j++) cach1[j] = X.index[pre_val++];
			for (j = 0; j < Core_N; j++) {
				double temp = CoreT.value[j];
				int k;
				int pos = j*order;
				int val = 0;
				for (k = 0; k < order; k++) {
					int mulrow = cach1[k], mulcol = CoreT.index[pos++];
					temp *= X.FactorM[val + mulrow*rrank + mulcol];
					val += mult;
				}
				ans += temp;
			}
			free(cach1);
			Error_T[i] = ans;
		}
	}

	if (X.gpu_mode != 0) {
		
		kernels = RECON.kernels; queues = RECON.queues; context = RECON.context; 
		
		int NNZ_PER_DEVICE = nnz / num_devices + 1, last;
			if (nnz%num_devices == 0) NNZ_PER_DEVICE--;
			last = nnz - NNZ_PER_DEVICE*(num_devices - 1);
			cl_mem *bufA, *bufB, *bufC, *bufD, *bufE, *bufF;
			bufA = (cl_mem*)malloc(sizeof(cl_mem) * num_devices); bufB = (cl_mem*)malloc(sizeof(cl_mem) * num_devices);  bufC = (cl_mem*)malloc(sizeof(cl_mem) * num_devices);  bufD = (cl_mem*)malloc(sizeof(cl_mem) * num_devices);  bufE = (cl_mem*)malloc(sizeof(cl_mem) * num_devices); bufF = (cl_mem*)malloc(sizeof(cl_mem) * num_devices);
			for (int j = 0; j < num_devices; j++) {
				if (j != num_devices - 1) {
					bufA[j] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * NNZ_PER_DEVICE*order, NULL, &err); CHECK_ERROR(err); err = clEnqueueWriteBuffer(queues[j], bufA[j], CL_FALSE, 0, sizeof(int) * NNZ_PER_DEVICE*order, X.index + j*NNZ_PER_DEVICE*order, 0, NULL, NULL); CHECK_ERROR(err);
					bufB[j] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*NNZ_PER_DEVICE, NULL, &err); CHECK_ERROR(err); err = clEnqueueWriteBuffer(queues[j], bufB[j], CL_FALSE, 0, sizeof(float) * NNZ_PER_DEVICE, X.value + j*NNZ_PER_DEVICE, 0, NULL, NULL); CHECK_ERROR(err);
					bufF[j] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*NNZ_PER_DEVICE, NULL, &err); CHECK_ERROR(err); err = clEnqueueWriteBuffer(queues[j], bufF[j], CL_FALSE, 0, sizeof(float)*NNZ_PER_DEVICE, (Error_T + j*NNZ_PER_DEVICE), 0, NULL, NULL);
				}
				else {
					bufA[j] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * last*order, NULL, &err); CHECK_ERROR(err); err = clEnqueueWriteBuffer(queues[j], bufA[j], CL_FALSE, 0, sizeof(int) * last*order, X.index + j*NNZ_PER_DEVICE*order, 0, NULL, NULL); CHECK_ERROR(err);
					bufB[j] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*last, NULL, &err); CHECK_ERROR(err); err = clEnqueueWriteBuffer(queues[j], bufB[j], CL_FALSE, 0, sizeof(float) * last, X.value + j*NNZ_PER_DEVICE, 0, NULL, NULL); CHECK_ERROR(err);
					bufF[j] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*last, NULL, &err); CHECK_ERROR(err); err = clEnqueueWriteBuffer(queues[j], bufF[j], CL_FALSE, 0, sizeof(float)*last, (Error_T + j*NNZ_PER_DEVICE), 0, NULL, NULL);
				}
				bufC[j] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int)*Core_N*order, NULL, &err); CHECK_ERROR(err); err = clEnqueueWriteBuffer(queues[j], bufC[j], CL_FALSE, 0, sizeof(int) * Core_N*order, CoreT.index, 0, NULL, NULL); CHECK_ERROR(err);
				bufD[j] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*Core_N, NULL, &err); CHECK_ERROR(err); err = clEnqueueWriteBuffer(queues[j], bufD[j], CL_FALSE, 0, sizeof(float) * Core_N, CoreT.value, 0, NULL, NULL); CHECK_ERROR(err);
				bufE[j] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*order*max_dim*rrank, NULL, &err); CHECK_ERROR(err); err = clEnqueueWriteBuffer(queues[j], bufE[j], CL_FALSE, 0, sizeof(float) * order*max_dim*rrank, X.FactorM, 0, NULL, NULL); CHECK_ERROR(err);

			}


			for (int j = 0; j < num_devices; j++) {
				err = clSetKernelArg(kernels[j], 0, sizeof(cl_mem), &bufA[j]); CHECK_ERROR(err);
				err = clSetKernelArg(kernels[j], 1, sizeof(cl_mem), &bufB[j]); CHECK_ERROR(err);
				err = clSetKernelArg(kernels[j], 2, sizeof(cl_mem), &bufC[j]); CHECK_ERROR(err);
				err = clSetKernelArg(kernels[j], 3, sizeof(cl_mem), &bufD[j]); CHECK_ERROR(err);
				err = clSetKernelArg(kernels[j], 4, sizeof(cl_mem), &bufE[j]); CHECK_ERROR(err);
				err = clSetKernelArg(kernels[j], 5, sizeof(cl_mem), &bufF[j]); CHECK_ERROR(err);
				err = clSetKernelArg(kernels[j], 6, sizeof(cl_int), &order); CHECK_ERROR(err);
				if (j == num_devices - 1) {
					err = clSetKernelArg(kernels[j], 7, sizeof(cl_int), &last); CHECK_ERROR(err);
				}
				else {
					err = clSetKernelArg(kernels[j], 7, sizeof(cl_int), &NNZ_PER_DEVICE); CHECK_ERROR(err);
				}
				err = clSetKernelArg(kernels[j], 8, sizeof(cl_int), &rrank); CHECK_ERROR(err);
				err = clSetKernelArg(kernels[j], 9, sizeof(cl_int), &Core_N); CHECK_ERROR(err);
				err = clSetKernelArg(kernels[j], 10, sizeof(cl_int), &mult); CHECK_ERROR(err);
			}

			size_t global_size = NNZ_PER_DEVICE;
			size_t local_size = X.local_size/2;
			global_size = (global_size + local_size - 1) / local_size * local_size;

			for (int j = 0; j < num_devices; j++) {
				err = clEnqueueNDRangeKernel(queues[j], kernels[j], 1, NULL, &global_size, &local_size, 0, NULL, NULL); CHECK_ERROR(err);
			}

			for (int j = 0; j < num_devices; j++) {
				if (j == num_devices - 1) {
					err = clEnqueueReadBuffer(queues[j], bufF[j], CL_TRUE, 0, sizeof(float)*last, Error_T + j*NNZ_PER_DEVICE, 0, NULL, NULL); CHECK_ERROR(err);
				}
				else {
					err = clEnqueueReadBuffer(queues[j], bufF[j], CL_TRUE, 0, sizeof(float)*NNZ_PER_DEVICE, Error_T + j*NNZ_PER_DEVICE, 0, NULL, NULL); CHECK_ERROR(err);
				}
			}

			for (int j = 0; j < num_devices; j++) {
				clReleaseMemObject(bufA[j]); clReleaseMemObject(bufB[j]); clReleaseMemObject(bufC[j]); clReleaseMemObject(bufD[j]); clReleaseMemObject(bufE[j]); clReleaseMemObject(bufF[j]);
			}
			free(bufA); free(bufB); free(bufC); free(bufD); free(bufE); free(bufF);
		}

	if (X.partially_observed == 0) {
		double *pre_check = (double *)malloc(sizeof(double)*order*rrank*rrank);

		partially_observed_pre_process(pre_check, order, rrank, mult, X.FactorM, dim);

		long long sizee = Core_N*Core_N,j;
		for (j = 0; j < sizee; j++) {
			int alpha = j / Core_N, beta = j - alpha*Core_N, k, aa;
			double totalval = CoreT.value[alpha] * CoreT.value[beta];
			for (k = 0; k < order; k++) {
				int pos3 = CoreT.index[alpha*order + k], pos4 = CoreT.index[beta*order + k], pos5 = k*mult;
				totalval *= pre_check[k*rrank*rrank + pos3*rrank + pos4];
			}
			Error+=totalval;
		}
		free(pre_check);

#pragma omp parallel for schedule(static)
		for (i = 0; i < nnz; i++) {
			float temp = X.value[i] * (X.value[i] - 2 * Error_T[i]);
			Error_T[i] = temp;
		}
	}
	st = omp_get_wtime();
	if (X.partially_observed == 1) {
#pragma omp parallel for schedule(static) reduction(+:Error)
		for (i = 0; i < nnz; i++) {
			Error += (X.value[i] - Error_T[i]) * (X.value[i] - Error_T[i]);
		}
	}
	else {
#pragma omp parallel for schedule(static) reduction(+:Error)
		for (i = 0; i < nnz; i++) {
			Error += Error_T[i];
		}
	}

#pragma omp parallel for schedule(static) reduction(+:NormX)
	for (i = 0; i < nnz; i++) {
		NormX += X.value[i] * X.value[i];
	}

	NormX = sqrt(NormX);
	if (NormX == 0) Fit = 1;
	else Fit = 1 - sqrt(Error) / NormX;
	free(Error_T);
}


void tensor_factorization(int rrank, char* Path) {
	printf("INITIALIZING FOR TF......\n");
	int order = X.order, nnz = X.nonzeros, i, j, k, threads = omp_get_max_threads();
	omp_set_num_threads(threads);
	num_devices = X.gpu_mode;

	//INITIALIZE
	for(i=0;i<order;i++){
		if(max_dim<X.dimension[i]) max_dim = X.dimension[i];
	}
	max_dim++;
	
	X.FactorM = (float *)malloc(sizeof(float)*order*max_dim * rrank);
	for (i = 0; i < order; i++) {
		int row = X.dimension[i];
		for (j = 0; j < row; j++) {
			for (k = 0; k < rrank; k++) {
				X.FactorM[i*max_dim*rrank + j*rrank + k] = ffrand(0, 1);
			}
		}
	}
	int Core_N = 1;
	int* core_dim = (int *)malloc(sizeof(int)*order);
	for (i = 0; i < order; i++) {
		core_dim[i] = rrank;
		Core_N *= rrank;
	}
	int* core_index = (int *)malloc(sizeof(int)*order*Core_N);
	float* core_val = (float *)malloc(sizeof(float)*order*Core_N);
	for (i = 0; i < Core_N; i++) {
		core_val[i] = ffrand(0, 1);
		if (i == 0) {
			for (j = 0; j < order; j++) core_index[j] = 0;
		}
		else {
			for (j = 0; j < order; j++) {
				core_index[i*order + j] = core_index[(i - 1)*order + j];
			}
			core_index[i*order + order - 1]++;  k = order - 1;
			while (core_index[i*order + k] >= rrank) {
				core_index[i*order + k] -= rrank;
				core_index[i*order + k - 1]++; k--;
			}
		}
	}

	Construct_Core_Tensor(order, Core_N, core_dim, core_index, core_val);

	WhereX = (int *)malloc(sizeof(int)*order*nnz);
	CountX = (int *)malloc(sizeof(int)*max_dim*order);
	assign_index();
	TF = GPU_INIT(X.gpu_mode, "src/GTA_GPU_Delta.cl", "compute_delta");
	RECON = GPU_INIT(X.gpu_mode, "src/GTA_reconstruction.cl", "recon");

	//Main Iteration
	int iter = 0;
	double avertime = omp_get_wtime();
	printf("INITIALIZE DONE!\n\nTensor order: %d\tnonzeros: %d\trrank: %d\tthreads: %d\tLocal size: %d\tNumber of GPUs: %d\n\n", order, nnz, rrank, omp_get_max_threads(),X.local_size,num_devices);
	if (X.gpu_mode == 0 && X.partially_observed==0) printf("Starting Tucker Factorization for a Fully Observable Tensor in CPU-mode......\n\n");
	if (X.gpu_mode == 0 && X.partially_observed==1) printf("Starting Tucker Factorization for a Partially Observable Tensor in CPU-mode......\n\n");
	if (X.gpu_mode == 1 && X.partially_observed==0) printf("Starting Tucker Factorization for a Fully Observable Tensor in Single-GPU-mode......\n\n");
	if (X.gpu_mode == 1 && X.partially_observed==1) printf("Starting Tucker Factorization for a Partially Observable Tensor in Single-GPU-mode......\n\n");
	if (X.gpu_mode >= 2 && X.partially_observed==0) printf("Starting Tucker Factorization for a Fully Observable Tensor in Multi-GPU-mode......\n\n");
	if (X.gpu_mode >= 2 && X.partially_observed==1) printf("Starting Tucker Factorization for a Partially Observable Tensor in Multi-GPU-mode......\n\n");

	while (1) {

		double itertime = omp_get_wtime(), steptime;
		steptime = itertime;

		Update_Factor_Matrices(rrank);
		printf("Factor Time : %lf\n", omp_get_wtime() - steptime);
		steptime = omp_get_wtime();

		Reconstruction(rrank);
		printf("Recon Time : %lf\n\n", omp_get_wtime() - steptime);
		steptime = omp_get_wtime();

		printf("iter%d :      Fit : %lf\tElapsed Time : %lf\n\n", ++iter, Fit, omp_get_wtime() - itertime);

		if (iter>=10 || (pFit != -1 && abss(pFit - Fit) <= 0.0001)) break;
		pFit = Fit;
	}
	printf("Average Elapsed time per iteration: %lf\n\n", (omp_get_wtime() - avertime) / iter);

	printf("\nWriting factor matrices and core tensor to file...\n");
	char temp[50];
	int pos = 0;
	int mult = max_dim*rrank;
	for (i = 0; i < order; i++) {
		sprintf(temp, "%s/FACTOR%d", Path, i);
		FILE *fin = fopen(temp, "w");
		for (j = 0; j < X.dimension[i]; j++) {
			for (k = 0; k < rrank; k++) {
				fprintf(fin, "%e\t", X.FactorM[i*mult + j*rrank + k]);
			}
			fprintf(fin, "\n");
		}
	}
	sprintf(temp, "%s/CORETENSOR", Path);
	FILE *fcore = fopen(temp, "w");
	pos = 0;
	for (i = 0; i < Core_N; i++) {
		for (j = 0; j < order; j++) {
			fprintf(fcore, "%d\t", CoreT.index[pos++]);
		}
		fprintf(fcore, "%e\n", CoreT.value[i]);
	}

	GPU_DONE(X.gpu_mode,TF);
	GPU_DONE(X.gpu_mode, RECON);
}

int main(int argc, char* argv[]) {

	if (argc !=0) {
		InputPath = argv[1];
		ResultPath = argv[2];
		rrank = atoi(argv[3]);
		l_size = atoi(argv[4]);
		gpu_mode = atoi(argv[5]);
		nnz_mode = atoi(argv[6]);
		

		Read_Tensor(InputPath);

		tensor_factorization(rrank,ResultPath);
	}	
	
	return 0;
}
