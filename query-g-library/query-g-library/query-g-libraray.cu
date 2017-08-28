#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <math.h>

#include <time.h>
#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif

#include "query-g-libraray.h"
#include "book.h"
#include "author.h"
#include "author-book.h"

//#define DEBUG

/*
============================================
Describe GTX 560:

Multiprocessors:___________________________________________________7
CUDA Cores:______________________________________________________336
Compute Capability:______________________________________________2.1

Maximum dimensionality of grid of thread blocks:___________________3
Maximum x- y- or z- dimensions of a grid of a thread blocks:___65535 <--
Maximum dimensionality of a thread block:__________________________3
Maximum x- or y- dimensions of a block:_________________________1024
Maximum z- dimensions of a block:_________________________________64
Maximum number of threads per block:____________________________1024 <--
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

	void QGLibraray::agregate_sum_cpu(Book* books, unsigned long num_books) {
		unsigned long sum = 0;

		// on windows this is wall time
		// on linux this is cpu time
		clock_t begin = clock();

		unsigned long i;
		for (i = 0; i < num_books; i++) {
			sum += books[i].invertar_id;
		}
		printf("Sum CPU: %lu\n\n", sum);

		clock_t end = clock();
		double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

		printf("Kernel time: %.2f\n", time_spent);
	}

	__global__ void agregate_sum_gpu_kernel(Book *d_input, unsigned long *d_output, unsigned long num_books)
	{
		// tune the size of d_output data
		// it shoul have the same as blockSize... probably

		extern __shared__  unsigned long sdata[];

		unsigned long tid = threadIdx.x;									// idx of a thread inside of the blocck, [0, blockSize-1]
		unsigned int i = blockIdx.x*(blockDim.x * 2) + threadIdx.x;		// global idx of a thread, [0, gridSize * blockSize -1]

		// perform first level of reduction,
		// reading from global memory, writing to shared memory
		sdata[tid] = d_input[i].invertar_id + d_input[i + blockDim.x].invertar_id;
		__syncthreads();

		// do reduction in sahred memory
		for (unsigned long s = blockDim.x / 2; s>32; s >>= 1)
		{
			if (tid < s)
				sdata[tid] += sdata[tid + s];
			__syncthreads();
		}
		if (tid < 32)
		{
			sdata[tid] += sdata[tid + 32];
			sdata[tid] += sdata[tid + 16];
			sdata[tid] += sdata[tid + 8];
			sdata[tid] += sdata[tid + 4];
			sdata[tid] += sdata[tid + 2];
			sdata[tid] += sdata[tid + 1];
		}

		// write result for this block to global memory
		if (tid == 0) {
			d_output[blockIdx.x] = sdata[0];
		}
	}

	void QGLibraray::agregate_sum(Book *books, unsigned long num_books) {
		cudaError err;
		const int threads_per_block = 128;

		// host input data
		Book *h_input;

		// host output data
		unsigned long h_output[threads_per_block];

		// device input data
		Book *d_input;

		// device output data
		unsigned long *d_output;

		// determine alocaation sizes;
		unsigned long size_output = sizeof(unsigned long) * threads_per_block;
		unsigned long size_books = sizeof(Book) * num_books;

		// alocate memory on host
		h_input = books;

		// alocate memory on device
		cudaMalloc(&d_input, size_books);
		cudaMalloc(&d_output, size_output);

		// copy from host to device
		cudaMemcpy(d_input, h_input, size_books, cudaMemcpyHostToDevice);

		// determine grid size and block size
		unsigned long n = num_books / 2;

		unsigned long blockSize = threads_per_block;
		unsigned long gridSize = ceil((float) n / (float) blockSize);

		printf("\nThreads [needed]:   %lu\n", n);
		printf("Threads [executed]: %lu\n", gridSize * blockSize);

		printf("\nGrid Size: %lu\n", gridSize);
		printf("Block Size: %lu\n", blockSize);

		printf("\nMemory for output: %lu bytes\n", size_output);

		// execute kernel

		// on windows this is wall time
		// on linux this is cpu time
		clock_t begin = clock();

		agregate_sum_gpu_kernel << <gridSize, blockSize, sizeof(unsigned long) * blockSize >> >(d_input, d_output, num_books);
		
		err = cudaGetLastError();
		if (err != cudaSuccess) {
			printf("\nKernel failed: %s\n", cudaGetErrorString(err));
		}
		else {
			//printf("\nKernel Sucess!!!\n");
		}

		cudaDeviceSynchronize();

		clock_t end = clock();
		double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

		printf("Kernel time: %.2f\n", time_spent);

		// copy from device to host
		cudaMemcpy(h_output, d_output, size_output, cudaMemcpyDeviceToHost);

		unsigned long local_sum = 0;
		
		int i;
		for (i = 0; i < threads_per_block; i++) {
			local_sum += h_output[i];
		}

		printf("Sum GPU: %lu\n\n", local_sum);

		// free memory
		cudaFree(d_input);
		cudaFree(d_output);
		//free(h_output);
	}

	void QGLibraray::cartesian_product_cpu(Author *author_list, unsigned long num_authors, Book *books_list, unsigned long num_books) {
		unsigned long i, j;
		unsigned long current = 0;

		AuthorBook *cartesian;
		cartesian = new AuthorBook[num_authors * num_books];

		// on windows this is wall time
		// on linux this is cpu time
		clock_t begin = clock();

		for (i = 0; i < num_authors; i++)
		{
			for (j = 0; j < num_books; j++)
			{

				cartesian[current].author_author_id = author_list[i].author_id;

				strcpy(cartesian[current].first_name, author_list[i].first_name);
				strcpy(cartesian[current].last_name, author_list[i].last_name);

				cartesian[current].book_author_id = books_list[j].author_id;
				cartesian[current].invertar_br = books_list[j].invertar_id;
				strcpy(cartesian[current].naslov, books_list[j].title);

				current++;
			}
		}

		clock_t end = clock();
		double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

		printf("Kernel time: %.2f\n", time_spent);

		//print_cartesian_cpu(cartesian, num_authors, num_books);

		delete[]cartesian;
	}

	void QGLibraray::increse_books_outhor_id_cpu(Book *book, unsigned long num_books, int amount) {
		unsigned long i;

		// on windows this is wall time
		// on linux this is cpu time
		clock_t begin = clock();

		for (i = 0; i < num_books; i++) {
			book[i].author_id += amount;
		}

		clock_t end = clock();
		double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

		printf("Kernel time: %.2f\n", time_spent);
	}

	/* kernel to increase book's id by given ammount */
	__global__ void increse_books_outhor_id_gpu(Book *books, unsigned long num_books, int ammount)
	{
		// Get our global thread ID
		unsigned long idx = blockIdx.x * blockDim.x + threadIdx.x;
		
		if (idx < num_books) {
			books[idx].author_id += ammount;
		}
	}

	// my implementation of strcpy
	// becouse it is not provided by cuda!!!
	__device__ char * my_strcpy(char *dest, const char *src) {
		int i = 0;
		do {
			dest[i] = src[i];
		} while (src[i++] != 0);
		return dest;
	}

	/* kernel to create cartesian product */
	__global__ void cartesian_product_gpu(Author *d_input_authors, Book *d_input_books, AuthorBook *d_output_author_book, unsigned long num_authors, unsigned long num_books, unsigned long good_threads)
	{
		// blockIdx = numBlocks = gridSize [0, 5]
		// blockDim = blockSize = 5
		// threadIdx = [0, blockDim]
		//
		// 0 * 5 + 0 = 0
		// 0 * 5 + 1 = 1
		// 0 * 5 + 2 = 2
		// 0 * 5 + 3 = 3
		// 0 * 5 + 4 = 4
		// 1 * 5 + 0 = 5
		// 1 * 5 + 1 = 6
		// ...
		// 4 * 5 + 4 = 24
		
		unsigned long idx;				// thread idx

		unsigned long pos_authors;		// acces ith author acording to thread idx
		unsigned long pos_books;		// acces ith book acording to thread idx

		idx = blockIdx.x * blockDim.x + threadIdx.x;

		if (idx > good_threads) {
			//printf("[overflow] thread IDX: %d -> not executed\n", idx);
		}
		else {
			// larger of both get pos using mod
			// smaller of both get pos using div
			if (num_books >= num_authors) {
				pos_books = idx % num_books;
				pos_authors = idx / num_books;
			}
			else {
				// num_authors > num_books
				pos_authors = idx % num_books;		// might need to sweam num_books witk num_authors and vice versa couse of ilegals memory access
				pos_books = idx / num_books;
			}

			//printf("IDX:%d, authors: %d, books:%d\n", idx, pos_authors, pos_books);

			d_output_author_book[idx].author_author_id = d_input_authors[pos_authors].author_id;
			d_output_author_book[idx].book_author_id = d_input_books[pos_books].author_id;
			d_output_author_book[idx].invertar_br = d_input_books[pos_books].invertar_id;

			my_strcpy(d_output_author_book[idx].first_name, d_input_authors[pos_authors].first_name);
			my_strcpy(d_output_author_book[idx].last_name, d_input_authors[pos_authors].last_name);
			my_strcpy(d_output_author_book[idx].naslov, d_input_books[pos_books].title);
		}
	}

	void QGLibraray::cartesian_product(Author *authors, unsigned long num_authors, Book *books, unsigned long num_books) {
		//printf("cartesian_product_in_libraray\n");

		unsigned long i;
		cudaError_t err;

		// host
		Author *h_input_authors;
		Book *h_input_books;
		AuthorBook *h_output;

		// device
		Author *d_input_authors;
		Book *d_input_books;
		AuthorBook *d_output;

		// calculate size
		unsigned long size_books = sizeof(Book) * num_books;
		unsigned long size_authors = sizeof(Author) * num_authors;
		unsigned long size_cartesian = sizeof(AuthorBook) * num_authors * num_books;

		// alocate memory on host
		h_input_authors = authors;
		h_input_books = books;
		h_output = new AuthorBook[num_books * num_authors];

		// alocate memory on device
		cudaMalloc(&d_input_authors, size_authors);

		err = cudaGetLastError();
		if (err != cudaSuccess) {
			printf("\nError alocating d_input_authors: %s\n", cudaGetErrorString(err));
		}

		cudaMalloc(&d_input_books, size_books);
		if (err != cudaSuccess) {
			printf("\nError alocating d_input_books: %s\n", cudaGetErrorString(err));
		}

		cudaMalloc(&d_output, size_cartesian);
		if (err != cudaSuccess) {
			printf("\nError allocating d_output: %s\n", cudaGetErrorString(err));
		}

		// copy from device to host
		cudaMemcpy(d_input_authors, h_input_authors, size_authors, cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			printf("\nError copying h_input_authors from host to device d_input_authors: %s\n", cudaGetErrorString(err));
		}

		cudaMemcpy(d_input_books, h_input_books, size_books, cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			printf("\nError copying h_input_books from host to device d_input_books: %s\n", cudaGetErrorString(err));
		}

		cudaMemcpy(d_output, h_output, size_cartesian, cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			printf("\nError copying h_output from host to device d_output: %s\n", cudaGetErrorString(err));
		}

		// define grid sise and block size
		unsigned long n = num_authors * num_books;

		unsigned long blockSize = 1024;			            // block size // multiple of 32 -> threads per block
		unsigned long gridSize = ceil ((float)n / (float)blockSize);		// number of blocks, each containing blockSize threads
		
		printf("\nThreads [needed]:   %lu\n", num_books * num_authors);
		printf("Threads [executed]: %lu\n", gridSize * blockSize);

		printf("\nGrid Size: %lu\n", gridSize);
		printf("Block Size: %lu\n", blockSize);

		printf("\nMemory for output: %lu bytes\n", size_cartesian);
		
//		if (num_books <= 1024) {
//			blockSize = num_books;
//		}
//		else {
//			blockSize = 1024;
//		}
		

		// print output
//		printf("before cartesian\n");
//
//		for (i = 0; i < num_authors * num_books; i++) {
//			printf("%d\n", i);
//			h_output[i].print_details();
//			printf("\n");
//		}

		// execute kernel
		clock_t begin = clock();

		cartesian_product_gpu << <gridSize, blockSize >> >(d_input_authors, d_input_books, d_output, num_authors, num_books, num_authors * num_books);
		cudaDeviceSynchronize();

		clock_t end = clock();
		double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
		printf("\nKernel time: %.2f\n", time_spent);

		err = cudaGetLastError();
		if (err != cudaSuccess) {
			printf("\nKernel failed: %s\n", cudaGetErrorString(err));
		}
		else {
			//printf("\nKernel Sucess!!!\n");
		}

		// copy from device to host
		cudaMemcpy(h_output, d_output, size_cartesian, cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) {
			printf("\nError copying d_output from device to host h_output: %s\n", cudaGetErrorString(err));
		}

		// print output
		//printf("after cartesian\n");

		//printf("Print result:\n");
		//printf("Enter amount of result data to be printed: ");
		//int dammy;
		//scanf("%d", &dammy);

		//for (i = 0; i < dammy; i++) {
		//	printf("%d\n", i);
		//	h_output[i].print_details();
		//	printf("\n");
		//}

		// free memory
		cudaFree(d_input_authors);
		cudaFree(d_input_books);
		cudaFree(d_output);

		delete[] h_output;
	}

	void QGLibraray::increse_books_outhor_id(Book *books, unsigned long num_books, int amount) {
		cudaError_t err;
		
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
		unsigned long n = num_books;

		unsigned long blockSize = 512;
		unsigned long gridSize = ceil(float(n) / (float)blockSize);

		//printf("\nThreads [needed]:   %lu\n", n);
		//printf("Threads [executed]: %lu\n", gridSize * blockSize);

		//printf("\nGrid Size: %lu\n", gridSize);
		//printf("Block Size: %lu\n", blockSize);

		//printf("\nMemory for output: %lu bytes\n\n", size);

#ifdef DEBUG
		printf("Number of blocks: %d\nNumber of threads per block: %d Total: %d", gridSize, blockSize, gridSize * blockSize);
#endif // DEBUG

		// execute kernel

		// on windows this is wall time
		// on linux this is cpu time
		clock_t begin = clock();

		increse_books_outhor_id_gpu << <gridSize, blockSize >> >(d_input, num_books, amount);
		cudaDeviceSynchronize();

		err = cudaGetLastError();
		if (err != cudaSuccess) {
			printf("\nKernel failed: %s\n", cudaGetErrorString(err));
		}

		clock_t end = clock();
		double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
		//printf("Kernel time: %.2f\n", time_spent);

		// copy from device to host
		cudaMemcpy(h_input, d_input, size, cudaMemcpyDeviceToHost);

		// release memory
		cudaFree(d_input);
	}
}
