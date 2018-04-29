#include <stdio.h>
#include <assert.h>
#include "cnn_parallel.h"
#include "mpi.h"

int main(int argc, char **argv) {
    int test = 1;
//    test 0: test serial cnn
//    test 1: test mpi matrix
//    test 2: test forward

    if (test == 1){
        double start,end;
        int L, P, myrank, N;
//    double *unew, *u;
//    double *x;

        int kind = 0;           // 0 for increasing, 1 for decreasing
        int i,step;
        int tag = 1;
        MPI_Init(&argc, &argv);
        MPI_Comm_size(MPI_COMM_WORLD, &P);
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        MPI_Status status;
        MPI_Barrier(MPI_COMM_WORLD);
        start = MPI_Wtime();

        matrix* a = initMat(5, 5, 1);

        if (myrank==0){
            matrix* b = initMat(5, 5, 1);
            recvMat(b, 1, tag, status);
            printMat(b);
        }else if (myrank == 1){
            int r;
            for (r=0; r<a->row*a->column; r++){
                a->val[r] = myrank;
            }
            sendMat(a, 0, tag);
        }



        end =MPI_Wtime();
        MPI_Finalize();
    }

    return 0;
}
