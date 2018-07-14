# P-Tucker

Overview
---------------

**Scalable Tucker Factorization for Sparse Tensors - Algorithms and Discoveries (ICDE 2018)**  
[Sejoon Oh](https://www.sejoonoh.com/), [Namyong Park](http://namyongpark.com/), [Lee Sael](https://leesael.github.io/), and [U Kang](https://datalab.snu.ac.kr/~ukang/)

[[Paper](https://datalab.snu.ac.kr/ptucker/ptucker.pdf)] [[Supplementary Material](https://datalab.snu.ac.kr/ptucker/supple.pdf)]

Please refer to the following website for the details of P-Tucker (https://datalab.snu.ac.kr/ptucker/)

Usage
---------------

**P-Tucker requires OpenMP 2.0 or above version! (if you use gcc/g++ compiler, it is installed by default)**

"make" command will create a single executable file, which is "P-Tucker".

The executable file takes five arguments, which are the path of input tensor file, path of directory for storing results, tensor order, tensor rank, and number of threads. The arguments MUST BE valid and in the above order.

		ex) ./P-Tucker input.txt result/ 3 10 20

If you put the command properly, P-Tucker will write all values of factor matrices and a core tensor in the result directory set by an argument. (PLEASE MAKE SURE THAT YOU HAVE A WRITE PERMISSION TO THE RESULT DIRECTORY!).

		ex) result/FACTOR1, result/CORETENSOR

**We note that input tensors must follow base-1 indexing and outputs are based on base-0 indexing.**

Demo
---------------
To run the demo, please follow the following procedure. Sample tensor is created as 100x100x100 size with 1,000 observable entries.

	1. Type "make demo"
	2. Check "sample/result" directory for the demo factorization results
  
Orthogonalization of Factor Matrices
---------------

You can apply QR decompositions to output factor matrices and core tensor according to the main paper using MATLAB or other languages. Notice that current version of P-Tucker does not orthogonalize factor matrices and update a core tensor.

P-Tucker-Cache and P-Tucker-APPROX 
---------------

If you want to run P-Tucker-Cache or P-Tucker-APPROX, please contact the main author (Sejoon Oh, ohhenrie@snu.ac.kr).

Tested Environment
---------------
We tested our proposed method **P-Tucker** in a Linux Ubuntu 16.04.3 LTS machine equipped with 20 Intel Xeon E5-2630 v4 2.2GHz CPUs and 512GB RAM.
