#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "query-g-libraray.h"
#include "book.h"

//#define DEBUG

/*
============================================
Describe GTX 560:

Multiprocessors:___________________________________________________7
CUDA Cores:______________________________________________________336
Compute Capability:______________________________________________2.1

Maximum dimensionality of grid of thread blocks:___________________3
Maximum x- y- or z- dimensions of a grid of a thread blocks:___65535
Maximum dimensionality of a thread block:__________________________3
Maximum x- or y- dimensions of a block:_________________________1024
Maximum z- dimensions of a block:_________________________________64
Maximum number of threads per block:____________________________1024
Warp size_________________________________________________________32
Maximum number of resident blocks per multiprocessor:______________8
Maximum number of resident warps per multiprocessor:______________48
Maximum number of resident threads per multiprocessor:__________1563
Number of 32-bit registers per multiprocessor:__________________32_K
Maximum number of shared memory per multiprocessor:_____________48_K
Number of shared memory banks:____________________________________32
Amount of local memory per thread:____________________________512_KB
Constant memory size:__________________________________________64_KB
Cache working set per multiprocessor for constant memory:_______8_KB
Maximum number of instructions per kernel:_________________2_million

===========================================
*/

/*
Performance constrains:
1)	The number of threads per block should be a round multiple of the warp size,
	which is 32 on all current hardware.

2)	Each streaming multiprocessor unit on the GPU must have enough active warps
	to sufficiently hide all of the different memory and instruction pipeline
	latency of the architecture and achieve maximum throughput. The orthodox
	approach here is to try achieving optimal hardware occupancy.

3)	By benchmarking, you will probably find that most non-trivial code has a
	"sweet spot" in the 128-512 threads per block range, but it will require some
	analysis on your part to find where that is. The good news is that because you
	are working in multiples of the warp size, the search space is very finite and
	the best configuration for a given piece of code relatively easy to find.

4)	Use this online calculator as a starting point:
	http://lxkarthi.github.io/cuda-calculator/

5)	
*/

namespace qgl {
	QGLibraray::QGLibraray() {
	}

	QGLibraray::~QGLibraray() {
	}

	/* kernel to increase book's id by given ammount */
	__global__ void increse_books_outhor_id_gpu(Book *books, int num_books, int ammount)
	{
		// Get our global thread ID
		int idx = blockIdx.x*blockDim.x + threadIdx.x;
		
		if (idx < num_books) {
			books[idx].author_id += ammount;
		}
	}

	void QGLibraray::increse_books_outhor_id(Book *books, int num_books, int amount) {
		// host
		Book *h_input;

		// device
		Book *d_input;

		// determine size in bytes for later allocation
		size_t size = sizeof(Book) * num_books;

		// alocate memory on host
		h_input = books;

		// alocate memory on device
		cudaMalloc(&d_input, size);

		// copy from host to device
		cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

		// define block size and threads per block
		int blockSize = 128;
		int gridSize = ceil(num_books / (float)blockSize );


#ifdef DEBUG
		printf("Number of blocks: %d\nNumber of threads per block: %d Total: %d", gridSize, blockSize, gridSize * blockSize);
#endif // DEBUG


		// execute kernel
		increse_books_outhor_id_gpu << <gridSize, blockSize >> >(d_input, num_books, amount);

		// copy from device to host
		cudaMemcpy(h_input, d_input, size, cudaMemcpyDeviceToHost);

		// release memory
		cudaFree(d_input);
	}
}
