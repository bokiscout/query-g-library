#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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

	/* kernel to increase book's id by given ammount */
	__global__ void increse_books_outhor_id_gpu(Book *books, int num_books, int ammount)
	{
		// Get our global thread ID
		int idx = blockIdx.x*blockDim.x + threadIdx.x;
		
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
	__global__ void cartesian_product_gpu(Author *d_input_authors, Book *d_input_books, AuthorBook *d_output_author_book, int num_authors, int num_books)
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
		int idx = blockIdx.x * blockDim.x + threadIdx.x;

		int pos_authors;
		int pos_books;

		// larger of both get pos using mod
		// smaller of both get pos using div
		if (num_authors >= num_books) {
			pos_authors = idx % num_authors;
			pos_books = idx / num_authors;
		}
		else {
			pos_authors = idx / num_books;
			pos_books = idx % num_books;
		}

		printf("IDX:%d, authors: %d, books:%d\n", idx, pos_authors, pos_books);

//		char new_fn[50] = "modified";
//		my_strcpy(d_output_author_book[idx].first_name, new_fn);

		d_output_author_book[idx].author_author_id = d_input_authors[pos_authors].author_id;
		d_output_author_book[idx].book_author_id = d_input_books[pos_books].author_id;
		d_output_author_book[idx].invertar_br = d_input_books[pos_books].invertar_id;
		
		my_strcpy(d_output_author_book[idx].first_name, d_input_authors[pos_authors].first_name);
		my_strcpy(d_output_author_book[idx].last_name, d_input_authors[pos_authors].last_name);
		my_strcpy(d_output_author_book[idx].naslov, d_input_books[pos_books].title);
	}

	void QGLibraray::cartesian_product(Author *authors, int num_authors, Book *books, int num_books) {
		printf("cartesian_product_in_libraray");
		int i;

		// host
		Author *h_input_authors;
		Book *h_input_books;
		AuthorBook *h_output;

		// device
		Author *d_input_authors;
		Book *d_input_books;
		AuthorBook *d_output;

		// calculate size
		int size_books = sizeof(Book) * num_books;
		int size_authors = sizeof(Author) * num_authors;
		int size_cartesian = sizeof(AuthorBook) * num_authors * num_books;

		// alocate memory on host
		h_input_authors = authors;
		h_input_books = books;
		h_output = new AuthorBook[num_books * num_authors];

		// alocate memory on device
		cudaMalloc(&d_input_authors, size_authors);
		cudaMalloc(&d_input_books, size_books);
		cudaMalloc(&d_output, size_cartesian);

		// copy from device to host
		cudaMemcpy(d_input_authors, h_input_authors, size_authors, cudaMemcpyHostToDevice);
		cudaMemcpy(d_input_books, h_input_books, size_books, cudaMemcpyHostToDevice);
		cudaMemcpy(d_output, h_output, size_cartesian, cudaMemcpyHostToDevice);

		// define grid sise and block size
		int gridSize = num_authors;		// num blocks
		int blockSize = num_books;		// block size

		// print output
		printf("before cartesian\n");

		for (i = 0; i < num_authors * num_books; i++) {
			h_output[i].print_details();
			printf("\n");
		}

		// execute kernel
		cartesian_product_gpu << <gridSize, blockSize >> >(d_input_authors, d_input_books, d_output, num_authors, num_books);

		// copy from device to host
		cudaMemcpy(h_output, d_output, size_cartesian, cudaMemcpyDeviceToHost);

		// print output
		printf("after cartesian\n");

		for (i = 0; i < num_authors * num_books; i++) {
			h_output[i].print_details();
			printf("\n");
		}

		// free memory
		delete[] h_output;

		cudaFree(d_input_authors);
		cudaFree(d_input_books);
		cudaFree(d_output);
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
