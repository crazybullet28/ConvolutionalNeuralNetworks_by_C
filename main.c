#include <stdio.h>
#include <assert.h>
#include "cnn.h"
#include "mpi.c"

int main(int argc, char **argv) {
    int test = 1;
//    test 0: test serial cnn
//    test 1: test mpi matrix
//

    if (test==0){
        srand(100);

        LabelArr testLabel=read_Lable("E:/MyDocuments/Documents/CLionProjects/Parallel_Proj/Minst/train-labels.idx1-ubyte");
        ImgArr testImg=read_Img("E:/MyDocuments/Documents/CLionProjects/Parallel_Proj/Minst/train-images.idx3-ubyte");

//    int outSize=testLabel->LabelPtr[0].l;

        CNN* cnn=(CNN*)malloc(sizeof(CNN));
        cnn_setup(cnn,testImg->ImgMatPtr[0]->row,testImg->ImgMatPtr[0]->column,testLabel->LabelPtr[0].l);

        CNNOpts opts;
        opts.numepochs=1;
        opts.eta=0.1;
        int trainNum=50000;
        trainModel(cnn,testImg,testLabel,opts,trainNum);
        //    testModel()
    }
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

