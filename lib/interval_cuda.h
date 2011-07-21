#ifndef __INTERVAL_CUDA_H__
#define __INTERVAL_CUDA_H__

unsigned int count_intersections_bsearch_cuda(struct interval *A,
										      unsigned int size_A,
											  struct interval *B,
										      unsigned int size_B);

unsigned int count_intersections_sort_bsearch_cuda(struct interval *A,
										      unsigned int size_A,
											  struct interval *B,
										      unsigned int size_B);

unsigned int count_intersections_i_bsearch_cuda(struct interval *A,
										      unsigned int size_A,
											  struct interval *B,
										      unsigned int size_B,
										      unsigned int size_I);


__global__
void count_bsearch_cuda (	unsigned int *A_start,
							unsigned int *A_len,
							int A_size,
							unsigned int *B_start,
							unsigned int *B_end,
							int B_size,
							unsigned int *R,
							int n);
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
							int n);

#endif
