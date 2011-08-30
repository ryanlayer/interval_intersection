#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <mpi.h>
#include <math.h>
#include <string.h>

#include "mt.h"
#include "timer.h"
#include "interval.h"

#define GET_REQ 0
#define SEND_A 1
#define SEND_B 2

#ifndef min
	#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif


int main(int argc, char *argv[]) {
	MPI_Status status;
	int rank, size;

	MPI_Init (&argc, &argv);    /* starts MPI */
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);  /* get current process id */
	MPI_Comm_size (MPI_COMM_WORLD, &size);  /* get number of processes */

	/*
	 * Load the set of quiereis from A, then search B in some number of rounds
	 */
	if (argc < 8) {
		fprintf(stderr, "usage:\t%s <size A> <len A> <size B> <len B> "
						"<seed> <P> <size I>\n",
				argv[0]);
		return 1;
	}

	unsigned int size_A = atoi(argv[1]);
	unsigned int len_A = atoi(argv[2]);
	unsigned int size_B = atoi(argv[3]);
	unsigned int len_B = atoi(argv[4]);
	unsigned int seed = atoi(argv[5]);
	unsigned int P = atoi(argv[6]);
	unsigned int size_I = atoi(argv[7]);
	unsigned int size_T = size_I;

	struct interval *A, *B;

	/*
	 *	Each node will take all of A, and a portion of B, then the final set
	 *	will be combined at 0
	 *	Static partition
	 */

	unsigned int int_work = size_B / size;
	unsigned int mod_work = size_B % size;
	unsigned int node_work = size_B % size;

	if (rank < mod_work)
		node_work = int_work + 1;
	else
		node_work = int_work;

	unsigned int work_start = rank * int_work + min(rank, mod_work);

	// Node 0 generates the random sets
	if (rank == 0) {
		A = (struct interval *)
				malloc(size_A * sizeof(struct interval));
		B = (struct interval *)
				malloc(size_B * sizeof(struct interval));

		init_genrand(seed);
		generate_interval_sets(A, size_A, len_A, B, size_B, len_B, P);
	} else {
		A = (struct interval *)
				malloc(size_A * sizeof(struct interval));
		B = (struct interval *)
				malloc(node_work * sizeof(struct interval));
	}

	unsigned int seen = 0;
	if (rank == 0) {
		while ( seen < (size - 1) ) {
			unsigned int send_work;
			// get the address of the client and how much data it wants
			MPI_Recv(&send_work, 1, MPI_UNSIGNED, MPI_ANY_SOURCE, GET_REQ,
					MPI_COMM_WORLD, &status);

			// send the full A
			MPI_Send(A, A_size, MPI_UNSIGNED, status.MPI_SOURCE, SEND_A,
					MPI_COMM_WORLD);

			// calculated where the next piece of work starts
			unsigned int remote_work_start = (status.MPI_SOURCE ) * int_work +
					min(status.MPI_SOURCE, mod_work);

			// send its portion of B
			MPI_Send(B + remote_work_start, send_work, MPI_UNSIGNED,
					status.MPI_SOURCE, SEND_B, MPI_COMM_WORLD);
		}
	} else { // rank != 0
		MPI_Send(&node_work, 1, MPI_UNSIGNED, 0, GET_REQ, MPI_COMM_WORLD);
		MPI_Recv(A, size_A, MPI_UNSIGNED, 0, SEND_A, MPI_COMM_WORLD, &status);
		MPI_Recv(B, 1, node_work, 0, SEND_B, MPI_COMM_WORLD, &status);
	}

	/*************************************************/
	/*
		int i;
		while (seen < (size - 1)) {
			MPI_Recv(R_r, A_size, MPI_UNSIGNED, MPI_ANY_SOURCE, 0,
					MPI_COMM_WORLD, &status);
			++seen;
			for (i = 0; i < A_size; i++) 
				R[i] += R_r[i];
		}
		for (i = 0; i < A_size; i++) 
			O += R[i];
	} else {
		MPI_Send(R, A_size, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD);
	}

	stop();
	unsigned long total = report();

	if (rank == 0)
		printf("%d,%d,%d\tO:%d\t\tt:%ld\tc:%d\ts:%d\n",
				A_size, line, A_size + line, O, total, chunk_size, size);

	*/
	MPI_Finalize();

	return 0;
}
