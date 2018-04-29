#include <stdio.h>
#include <assert.h>
#include "cnn_parallel.h"
#include "mpi.h"

int main(int argc, char **argv) {
    int test = 2;
//    test 0: test serial cnn
//    test 1: test mpi matrix
//    test 2: test forward

    if (test == 1){
        double start,end;
        int P, myrank;
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


    }else if (test==2){
        int P, myrank;
        int tag = 1;
        MPI_Init(&argc, &argv);
        MPI_Comm_size(MPI_COMM_WORLD, &P);
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        MPI_Status status;
        MPI_Barrier(MPI_COMM_WORLD);
        LabelArr testLabel=read_Lable("Minst/train-labels.idx1-ubyte");
        ImgArr testImg=read_Img("Minst/train-images.idx3-ubyte");

        CNN* cnn=(CNN*)malloc(sizeof(CNN));
        cnn_setup(cnn,testImg->ImgMatPtr[0]->row,testImg->ImgMatPtr[0]->column,testLabel->LabelPtr[0].l, P, myrank);

        char saveFilePath[50];
        sprintf(saveFilePath, "cnnWeight_P%d_p%d.txt", P, myrank);
        cnnSaveWeight(cnn, saveFilePath);
        cnnfw(cnn, testImg->ImgMatPtr[0], P, myrank, status);

        sprintf(saveFilePath, "cnnOutput_P%d_p%d.txt", P, myrank);
        cnnSaveOutput(cnn, testImg->ImgMatPtr[0], saveFilePath);

    }

    return 0;
}
