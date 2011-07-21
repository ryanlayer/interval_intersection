#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cutil.h>
#include "radixsort.h"
#include "interval.h"
#include "timer.h"
#include "interval_cuda.h"
#include "reduce.h"
#include "bsearch_cuda.h"
#include "bsearch_cuda.cu"

//{{{ unsigned int count_intersections_bsearch_cuda(struct interval *A,
unsigned int count_intersections_bsearch_cuda(struct interval *A,
										      unsigned int size_A,
											  struct interval *B,
										      unsigned int size_B)
{
	int block_size = 256;
	dim3 dimBlock(block_size);
	int grid_size = ( size_A + block_size - 1) / (block_size * 1);
	dim3 dimGridSearch( grid_size );
	unsigned int i;
	cudaError_t err;

	//{{{ Move intervals to unsigned int arrays
	unsigned int *A_starts_h = 
		(unsigned int *) malloc( (size_A) * sizeof(unsigned int));
	unsigned int *A_lens_h = 
		(unsigned int *) malloc( (size_A) * sizeof(unsigned int));

	unsigned int *B_starts_h = 
		(unsigned int *) malloc( (size_B) * sizeof(unsigned int));
	unsigned int *B_ends_h = 
		(unsigned int *) malloc( (size_B) * sizeof(unsigned int));

	for (i = 0; i < size_B; i++) {
		B_starts_h[i] = B[i].start;
		B_ends_h[i] = B[i].end;
	}

	for (i = 0; i < size_A; i++) {
		A_starts_h[i] = A[i].start;
		A_lens_h[i] = A[i].end - A[i].start;
	}
	//}}}

	//{{{ Move inteval arrays to device
	unsigned int *A_starts_d, *A_lens_d, *B_starts_d, *B_ends_d;
	cudaMalloc((void **)&A_starts_d, (size_A)*sizeof(unsigned int));
	cudaMalloc((void **)&A_lens_d, (size_A)*sizeof(unsigned int));
	cudaMalloc((void **)&B_starts_d, (size_B)*sizeof(unsigned int));
	cudaMalloc((void **)&B_ends_d, (size_B)*sizeof(unsigned int));

	cudaMemcpy(A_starts_d, A_starts_h, (size_A) * sizeof(unsigned int), 
			cudaMemcpyHostToDevice);
	cudaMemcpy(A_lens_d, A_lens_h, (size_A) * sizeof(unsigned int),
			cudaMemcpyHostToDevice);
	cudaMemcpy(B_starts_d, B_starts_h, (size_B) * sizeof(unsigned int), 
			cudaMemcpyHostToDevice);
	cudaMemcpy(B_ends_d, B_ends_h, (size_B) * sizeof(unsigned int),
			cudaMemcpyHostToDevice);

	cudaThreadSynchronize();
	err = cudaGetLastError();
	if(err != cudaSuccess)
		fprintf(stderr, "Interval move: %s.\n", cudaGetErrorString( err) );
	//}}}
	
	//{{{ Alocate space for result on device
	unsigned int *R_d;
	cudaMalloc((void **)&R_d, (size_A)*sizeof(unsigned int));
	unsigned long memup_time = report();

	cudaThreadSynchronize();
	err = cudaGetLastError();
	if(err != cudaSuccess)
		fprintf(stderr, "R_d malloc: %s.\n", cudaGetErrorString( err) );
	//}}}

	//{{{ Sort B_starts and B_ends
	// Sort B by start
	nvRadixSort::RadixSort radixsortB_starts(size_B, true);
	radixsortB_starts.sort((unsigned int*)B_starts_d, 0, size_B, 32);

	cudaThreadSynchronize();
	err = cudaGetLastError();
	if(err != cudaSuccess)
		fprintf(stderr, "Sort B_starts: %s.\n", cudaGetErrorString( err) );

	// Sort B by end
	nvRadixSort::RadixSort radixsortB_ends(size_B, true);
	radixsortB_ends.sort((unsigned int*)B_ends_d, 0, size_B, 32);

	cudaThreadSynchronize();
	err = cudaGetLastError();
	if(err != cudaSuccess)
		fprintf(stderr, "Sort B_ends: %s.\n", cudaGetErrorString( err) );
	//}}}

	//{{{ Compute and count intersections
	count_bsearch_cuda <<<dimGridSearch, dimBlock >>> (
			A_starts_d, A_lens_d, size_A,
			B_starts_d, B_ends_d, size_B,
			R_d,
			1);

	cudaThreadSynchronize();
	err = cudaGetLastError();
	if(err != cudaSuccess)
		fprintf(stderr, "Sort B_ends: %s.\n", cudaGetErrorString( err) );

	parallel_sum(R_d, block_size, size_A, 1024);

	cudaThreadSynchronize();
	err = cudaGetLastError();
	if(err != cudaSuccess)
		fprintf(stderr, "Parallel sum: %s.\n", cudaGetErrorString( err) );


	unsigned int R;
	cudaMemcpy(&R, R_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);

	cudaThreadSynchronize();
	err = cudaGetLastError();
	if(err != cudaSuccess)
		fprintf(stderr, "Result move: %s.\n", cudaGetErrorString( err) );

	//}}}
	
	cudaFree(A_starts_d);
	cudaFree(A_lens_d);
	cudaFree(B_starts_d);
	cudaFree(B_ends_d);
	cudaFree(R_d);

	return R;
}
//}}}

//{{{ unsigned int count_intersections_sort_bsearch_cuda(struct interval *A,
unsigned int count_intersections_sort_bsearch_cuda(struct interval *A,
										      unsigned int size_A,
											  struct interval *B,
										      unsigned int size_B)
{
	int block_size = 256;
	dim3 dimBlock(block_size);
	int grid_size = ( size_A + block_size - 1) / (block_size * 1);
	dim3 dimGridSearch( grid_size );
	unsigned int i;
	cudaError_t err;

	//{{{ Move intervals to unsigned int arrays
	unsigned int *A_starts_h = 
		(unsigned int *) malloc( (size_A) * sizeof(unsigned int));
	unsigned int *A_lens_h = 
		(unsigned int *) malloc( (size_A) * sizeof(unsigned int));

	unsigned int *B_starts_h = 
		(unsigned int *) malloc( (size_B) * sizeof(unsigned int));
	unsigned int *B_ends_h = 
		(unsigned int *) malloc( (size_B) * sizeof(unsigned int));

	for (i = 0; i < size_B; i++) {
		B_starts_h[i] = B[i].start;
		B_ends_h[i] = B[i].end;
	}

	for (i = 0; i < size_A; i++) {
		A_starts_h[i] = A[i].start;
		A_lens_h[i] = A[i].end - A[i].start;
	}
	//}}}

	//{{{ Move inteval arrays to device
	unsigned int *A_starts_d, *A_lens_d, *B_starts_d, *B_ends_d;
	cudaMalloc((void **)&A_starts_d, (size_A)*sizeof(unsigned int));
	cudaMalloc((void **)&A_lens_d, (size_A)*sizeof(unsigned int));
	cudaMalloc((void **)&B_starts_d, (size_B)*sizeof(unsigned int));
	cudaMalloc((void **)&B_ends_d, (size_B)*sizeof(unsigned int));

	cudaMemcpy(A_starts_d, A_starts_h, (size_A) * sizeof(unsigned int), 
			cudaMemcpyHostToDevice);
	cudaMemcpy(A_lens_d, A_lens_h, (size_A) * sizeof(unsigned int),
			cudaMemcpyHostToDevice);
	cudaMemcpy(B_starts_d, B_starts_h, (size_B) * sizeof(unsigned int), 
			cudaMemcpyHostToDevice);
	cudaMemcpy(B_ends_d, B_ends_h, (size_B) * sizeof(unsigned int),
			cudaMemcpyHostToDevice);

	cudaThreadSynchronize();
	err = cudaGetLastError();
	if(err != cudaSuccess)
		fprintf(stderr, "Interval move: %s.\n", cudaGetErrorString( err) );
	//}}}
	
	//{{{ Alocate space for result on device
	unsigned int *R_d;
	cudaMalloc((void **)&R_d, (size_A)*sizeof(unsigned int));
	unsigned long memup_time = report();

	cudaThreadSynchronize();
	err = cudaGetLastError();
	if(err != cudaSuccess)
		fprintf(stderr, "R_d malloc: %s.\n", cudaGetErrorString( err) );
	//}}}

	//{{{ Sort B_starts and B_ends
	// Sort B by start
	nvRadixSort::RadixSort radixsortB_starts(size_B, true);
	radixsortB_starts.sort((unsigned int*)B_starts_d, 0, size_B, 32);

	cudaThreadSynchronize();
	err = cudaGetLastError();
	if(err != cudaSuccess)
		fprintf(stderr, "Sort B_starts: %s.\n", cudaGetErrorString( err) );

	// Sort B by end
	nvRadixSort::RadixSort radixsortB_ends(size_B, true);
	radixsortB_ends.sort((unsigned int*)B_ends_d, 0, size_B, 32);

	cudaThreadSynchronize();
	err = cudaGetLastError();
	if(err != cudaSuccess)
		fprintf(stderr, "Sort B_ends: %s.\n", cudaGetErrorString( err) );
	//}}}
	
	//{{{ Sort A
	nvRadixSort::RadixSort sort_A_starts_lens_d(size_A, false);
	sort_A_starts_lens_d.sort((unsigned int*)A_starts_d, A_lens_d, size_A, 32);

	cudaThreadSynchronize();
	err = cudaGetLastError();
	if(err != cudaSuccess)
		fprintf(stderr, "Sort A_starts and lens: %s.\n", cudaGetErrorString( err) );

	stop();
	unsigned long sort_q_time = report();
	//}}}

	//{{{ Compute and count intersections
	count_bsearch_cuda <<<dimGridSearch, dimBlock >>> (
			A_starts_d, A_lens_d, size_A,
			B_starts_d, B_ends_d, size_B,
			R_d,
			1);

	cudaThreadSynchronize();
	err = cudaGetLastError();
	if(err != cudaSuccess)
		fprintf(stderr, "Sort B_ends: %s.\n", cudaGetErrorString( err) );

	parallel_sum(R_d, block_size, size_A, 1024);

	cudaThreadSynchronize();
	err = cudaGetLastError();
	if(err != cudaSuccess)
		fprintf(stderr, "Parallel sum: %s.\n", cudaGetErrorString( err) );


	unsigned int R;
	cudaMemcpy(&R, R_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);

	cudaThreadSynchronize();
	err = cudaGetLastError();
	if(err != cudaSuccess)
		fprintf(stderr, "Result move: %s.\n", cudaGetErrorString( err) );

	//}}}

	cudaFree(A_starts_d);
	cudaFree(A_lens_d);
	cudaFree(B_starts_d);
	cudaFree(B_ends_d);
	cudaFree(R_d);

	return R;
}
//}}}

//{{{ unsigned int count_intersections_i_bsearch_cuda(struct interval *A,
unsigned int count_intersections_i_bsearch_cuda(struct interval *A,
										      unsigned int size_A,
											  struct interval *B,
										      unsigned int size_B,
										      unsigned int size_I)
{
	int block_size = 256;
	dim3 dimBlock(block_size);
	int grid_size = ( size_A + block_size - 1) / (block_size);
	dim3 dimGridSearch( grid_size );
	unsigned int i;
	cudaError_t err;

	//{{{ Move intervals to unsigned int arrays
	unsigned int *A_starts_h = 
		(unsigned int *) malloc( (size_A) * sizeof(unsigned int));
	unsigned int *A_lens_h = 
		(unsigned int *) malloc( (size_A) * sizeof(unsigned int));

	unsigned int *B_starts_h = 
		(unsigned int *) malloc( (size_B) * sizeof(unsigned int));
	unsigned int *B_ends_h = 
		(unsigned int *) malloc( (size_B) * sizeof(unsigned int));

	for (i = 0; i < size_B; i++) {
		B_starts_h[i] = B[i].start;
		B_ends_h[i] = B[i].end;
	}

	for (i = 0; i < size_A; i++) {
		A_starts_h[i] = A[i].start;
		A_lens_h[i] = A[i].end - A[i].start;
	}
	//}}}

	//{{{ Move inteval arrays to device
	unsigned int *A_starts_d, *A_lens_d, *B_starts_d, *B_ends_d;
	cudaMalloc((void **)&A_starts_d, (size_A)*sizeof(unsigned int));
	cudaMalloc((void **)&A_lens_d, (size_A)*sizeof(unsigned int));
	cudaMalloc((void **)&B_starts_d, (size_B)*sizeof(unsigned int));
	cudaMalloc((void **)&B_ends_d, (size_B)*sizeof(unsigned int));

	cudaMemcpy(A_starts_d, A_starts_h, (size_A) * sizeof(unsigned int), 
			cudaMemcpyHostToDevice);
	cudaMemcpy(A_lens_d, A_lens_h, (size_A) * sizeof(unsigned int),
			cudaMemcpyHostToDevice);
	cudaMemcpy(B_starts_d, B_starts_h, (size_B) * sizeof(unsigned int), 
			cudaMemcpyHostToDevice);
	cudaMemcpy(B_ends_d, B_ends_h, (size_B) * sizeof(unsigned int),
			cudaMemcpyHostToDevice);

	cudaThreadSynchronize();
	err = cudaGetLastError();
	if(err != cudaSuccess)
		fprintf(stderr, "Interval move: %s.\n", cudaGetErrorString( err) );
	//}}}
	
	//{{{ Alocate space for result on device
	unsigned int *R_d;
	cudaMalloc((void **)&R_d, (size_A)*sizeof(unsigned int));
	unsigned long memup_time = report();

	cudaThreadSynchronize();
	err = cudaGetLastError();
	if(err != cudaSuccess)
		fprintf(stderr, "R_d malloc: %s.\n", cudaGetErrorString( err) );
	//}}}

	//{{{ Sort B_starts and B_ends
	// Sort B by start
	nvRadixSort::RadixSort radixsortB_starts(size_B, true);
	radixsortB_starts.sort((unsigned int*)B_starts_d, 0, size_B, 32);

	cudaThreadSynchronize();
	err = cudaGetLastError();
	if(err != cudaSuccess)
		fprintf(stderr, "Sort B_starts: %s.\n", cudaGetErrorString( err) );

	// Sort B by end
	nvRadixSort::RadixSort radixsortB_ends(size_B, true);
	radixsortB_ends.sort((unsigned int*)B_ends_d, 0, size_B, 32);

	cudaThreadSynchronize();
	err = cudaGetLastError();
	if(err != cudaSuccess)
		fprintf(stderr, "Sort B_ends: %s.\n", cudaGetErrorString( err) );
	//}}}

	//{{{ Generate index
	unsigned int *I_starts_d, *I_ends_d;
	cudaMalloc((void **)&I_starts_d, (size_I)*sizeof(unsigned int));
	cudaMalloc((void **)&I_ends_d, (size_I)*sizeof(unsigned int));

	int index_grid_size = ( size_I + block_size - 1) / (block_size);
	dim3 index_dimGrid( index_grid_size );

	gen_index <<<index_dimGrid, dimBlock>>> ( B_starts_d, size_B, I_starts_d, size_I);
	gen_index <<<index_dimGrid, dimBlock>>> ( B_ends_d, size_B, I_ends_d, size_I);

	cudaThreadSynchronize();
	err = cudaGetLastError();
	if(err != cudaSuccess)
		fprintf(stderr, "Count i bsearch: %s.\n", cudaGetErrorString( err) );
	//}}}

	//{{{ Compute and count intersections
	count_i_bsearch_cuda <<<dimGridSearch, dimBlock >>> (
			A_starts_d, A_lens_d, size_A,
			B_starts_d, B_ends_d, size_B,
			I_starts_d, I_ends_d, size_I,
			R_d,
			1);

	cudaThreadSynchronize();
	err = cudaGetLastError();
	if(err != cudaSuccess)
		fprintf(stderr, "Count i bsearch: %s.\n", cudaGetErrorString( err) );

	parallel_sum(R_d, block_size, size_A, 1024);

	cudaThreadSynchronize();
	err = cudaGetLastError();
	if(err != cudaSuccess)
		fprintf(stderr, "Parallel sum: %s.\n", cudaGetErrorString( err) );


	unsigned int R;
	cudaMemcpy(&R, R_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);

	cudaThreadSynchronize();
	err = cudaGetLastError();
	if(err != cudaSuccess)
		fprintf(stderr, "Result move: %s.\n", cudaGetErrorString( err) );

	//}}}
	
	cudaFree(A_starts_d);
	cudaFree(A_lens_d);
	cudaFree(B_starts_d);
	cudaFree(B_ends_d);
	cudaFree(I_starts_d);
	cudaFree(I_ends_d);
	cudaFree(R_d);

	return R;
}
//}}}

//{{{ __global__ void count_bsearch_cuda (	unsigned int *A_start,
/*
 * @param A_start list of start positions to query, does not need to be sorted
 * @param A_len list of lengths that correspond to A_start
 * @param A_size size of A_start and A_len
 * @param B_start list of sorted start positions to be queried
 * @param B_end list of sorted end positions to be queired 
 * @param B_size size of B_start and B_end
 * @param R number of intersections for each interval in A
 * @param n number of intervals per thread
 */
__global__
void count_bsearch_cuda (	unsigned int *A_start,
							unsigned int *A_len,
							int A_size,
							unsigned int *B_start,
							unsigned int *B_end,
							int B_size,
							unsigned int *R,
							int n)
{
	unsigned int id = (blockIdx.x * blockDim.x) + threadIdx.x;

	unsigned int i = id;
	unsigned int grid_size = blockDim.x * gridDim.x;

	while ( i < (n * grid_size) ) {

		if (i < A_size) {
			unsigned int start = A_start[i];
			unsigned int end = start + A_len[i];

			int cant_before = bound_binary_search(B_end,
														   B_size,
														   start,
														   -1,
														   B_size);

			int cant_after = bound_binary_search(B_start,
														  B_size,
														  end,
														  -1,
														  B_size);

			while ( end == B_start[cant_after] )
				++cant_after;

			cant_after = A_size - cant_after;	

			R[i] = A_size - cant_before - cant_after;
		}
		i += grid_size;
	}
}
//}}}

//{{{ __global__ void count_bsearch_cuda (	unsigned int *A_start,
/*
 * @param A_start list of start positions to query, does not need to be sorted
 * @param A_len list of lengths that correspond to A_start
 * @param A_size size of A_start and A_len
 * @param B_start list of sorted start positions to be queried
 * @param B_end list of sorted end positions to be queired 
 * @param B_size size of B_start and B_end
 * @param R number of intersections for each interval in A
 * @param n number of intervals per thread
 */
__global__
void count_i_bsearch_cuda (	unsigned int *A_start,
							unsigned int *A_len,
							int A_size,
							unsigned int *B_start,
							unsigned int *B_end,
							int B_size,
							unsigned int *I_start,
							unsigned int *I_end,
							int I_size,
							unsigned int *R,
							int n)
{
	unsigned int id = (blockIdx.x * blockDim.x) + threadIdx.x;

	unsigned int i = id;
	unsigned int grid_size = blockDim.x * gridDim.x;

	while ( i < (n * grid_size) ) {
		if (i < A_size) {
			unsigned int start = A_start[i];
			unsigned int end = start + A_len[i];

			int cant_before = i_binary_search(B_end,
											  B_size,
											  start,
											  I_end,
											  I_size);
	
			int cant_after = i_binary_search(B_start,
											 B_size,
											 end,
											 I_start,
											 I_size);

			while ( end == B_start[cant_after] )
				++cant_after;

			cant_after = A_size - cant_after;	

			R[i] = A_size - cant_before - cant_after;
		}
		i += grid_size;
	}
}
//}}}

