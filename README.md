# High-Performance Tucker Factorization on Heterogeneous Platforms (GTA)

Overview
---------------

**High-Performance Tucker Factorization on Heterogeneous Platforms**  
[Sejoon Oh](https://sejoonoh.github.io/), [Namyong Park](http://namyongpark.com/), [Jungi Jang](https://datalab.snu.ac.kr/~jkjang/), [Lee Sael](https://leesael.github.io/), and [U Kang](https://datalab.snu.ac.kr/~ukang/)  
*[IEEE Transactions on Parallel and Distributed Systems (TPDS)](https://www.computer.org/csdl/journal/td), 2019*  
[[Paper](https://github.com/sejoonoh/GTA-Tensor/blob/master/papers/GTA.pdf)] [[Supplementary Material](https://github.com/sejoonoh/GTA-Tensor/blob/master/papers/supple.pdf)] [[Offical Page](https://ieeexplore.ieee.org/document/8678477)]

Usage
---------------

**GTA requires OpenMP (2.0+ version) and OpenCL (1.2+ version) libraries! (if you use gcc/g++ compiler, OpenMP is installed by default)**

"make" command will create a single executable file, which is "GTA".

The executable file takes five arguments, which are the path of input tensor file, path of directory for storing results, tensor rank (10~25 are recommended for Tucker), local size (64, 256, 512, and 1024 are recommended), number of GPUs (depends on your machine), binary number indicating POTF (1) or FOTF (0). The arguments MUST BE valid and in the above order.

		ex) ./GTA input.txt result/ 10 256 1 1

**Input tensor must follow tab- or space-separated format (see the demo tensor).**  
**It is highly recommended to modify the pre-defined size of tmp and tmp2 in kernel files (src/GTA_GPU_Delta.cl and src/GTA_reconstruction.cl).**  
**For tmp size (must be larger than the tensor rank), 10~25 is recommended.**  
**For tmp2 size (must be larger than the tensor order), 3~10 is recommended.**  


If you put the command properly, GTA will write all values of factor matrices and a core tensor in the result directory set by an argument. (PLEASE MAKE SURE THAT YOU HAVE A WRITE PERMISSION TO THE RESULT DIRECTORY!).

		ex) result/FACTOR1, result/CORETENSOR

**We note that input tensors must follow base-1 indexing and outputs are based on base-0 indexing.**

Demo
---------------
To run the demo, please follow the following procedure. Sample tensor is created as 100x100x100 size with 1,000 observable entries.

	1. Type "make demo"
	2. Check "result" directory for the demo factorization results
  
Dataset
---------------
| Name | Structure | Size | Number of Nonzeros | Download |
| :------------: | :-----------: | :-------------: |:------------: |:------------------: |
| Netflix     | (User, Movie, Year-month; Rating) | 480K &times; 18K &times; 74 | 100M | [DOWN](https://datalab.snu.ac.kr/data/GTA/netflix.zip) |
| MovieLens     | (User, Movie, Year, Hour; Rating) | 138K &times; 27K &times; 21 &times;  24 | 20M | [DOWN](https://datalab.snu.ac.kr/data/GTA/movielens.zip) |
| DBLP     | (Author, Conference, Year; Count) | 418K &times; 4K &times; 50 | 1.3M | [DOWN](https://datalab.snu.ac.kr/data/GTA/dblp.zip) |
| Facebook     | (User 1, User 2, Date; Friendship) | 64K &times; 64K &times; 870 | 1.5M | [DOWN](https://datalab.snu.ac.kr/data/GTA/facebook.zip) |
| Synthetic     | Synthetic random tensors | 10K &times; 10K &times;  10K &times; 10K | ~10M |  |

Orthogonalization of Factor Matrices
---------------

You can apply QR decompositions to output factor matrices according to the main paper using MATLAB or other languages. Notice that current version of GTA does not orthogonalize factor matrices by default.
