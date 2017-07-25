#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "query-g-libraray.h"
#include "book.h"

namespace qgl {
	QGLibraray::QGLibraray() {
	}

	QGLibraray::~QGLibraray() {

	}

	/* kernel to increase book's id by given ammount */
	__global__ void increese_book_id(Book *book, int ammount)
	{
		// Get our global thread ID
		int id = blockIdx.x*blockDim.x + threadIdx.x;

		book->author_id += ammount;
		//book->
	}

	void QGLibraray::increse_book_id(Book *book, int amount) {

		// host
		Book *h_input;

		// device
		Book *d_input;

		// determine size in bytes for later allocation
		size_t size = sizeof(Book);

		// alocate memory on host
		h_input = book;

		// alocate memory on device
		cudaMalloc(&d_input, size);

		// copy from host to device
		cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

		// execute kernel
		increese_book_id << <1, 1 >> >(d_input, amount);

		// copy from device to host
		cudaMemcpy(h_input, d_input, size, cudaMemcpyDeviceToHost);

		// release memory
		cudaFree(d_input);
	}
}
