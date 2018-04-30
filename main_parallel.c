#include <stdio.h>
#include <assert.h>
#include "cnn_parallel.h"
#include "mpi.h"

int main(int argc, char **argv) {
    int test = 4;
//    test 0: test serial cnn
//    test 1: test mpi matrix
//    test 2: test forward
//    test 3: test train
//    test 4: test batch

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
        printf("[test] P%d p%d\n", P, myrank);
        LabelArr testLabel=read_Lable("Minst/train-labels.idx1-ubyte");
        ImgArr testImg=read_Img("Minst/train-images.idx3-ubyte");
        CNNOpts opts;
        opts.numepochs=1;
        opts.eta=0.1;

//        printf("Load image finished.\n");

        CNN* cnn=(CNN*)malloc(sizeof(CNN));
        cnn_setup(cnn,testImg->ImgMatPtr[0]->row,testImg->ImgMatPtr[0]->column,testLabel->LabelPtr[0].l, P, myrank);

        char saveFilePath[50];
        sprintf(saveFilePath, "cnnWeight_P%d_p%d.txt", P, myrank);
        cnnSaveWeight(cnn, saveFilePath);
        cnnfw(cnn, testImg->ImgMatPtr[0], P, myrank, status);

        sprintf(saveFilePath, "cnnOutput_P%d_p%d.txt", P, myrank);
        cnnSaveOutput(cnn, testImg->ImgMatPtr[0], saveFilePath);

        cnnbp(cnn, testLabel->LabelPtr[0].LabelData, P, myrank, status);
        gradient_update(cnn, opts, testImg->ImgMatPtr[0]);

        sprintf(saveFilePath, "cnnWeight_P%d_p%d_after.txt", P, myrank);
        cnnSaveWeight(cnn, saveFilePath);
    }else if(test==3){
        int P, myrank;
        int tag = 1;
        MPI_Init(&argc, &argv);
        MPI_Comm_size(MPI_COMM_WORLD, &P);
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        MPI_Status status;
        MPI_Barrier(MPI_COMM_WORLD);
        printf("[test] P%d p%d\n", P, myrank);
        LabelArr testLabel=read_Lable("Minst/train-labels.idx1-ubyte");
        ImgArr testImg=read_Img("Minst/train-images.idx3-ubyte");
        CNNOpts opts;
        opts.numepochs=1;
        opts.eta=0.1;

//        printf("Load image finished.\n");

        CNN* cnn=(CNN*)malloc(sizeof(CNN));
        cnn_setup(cnn,testImg->ImgMatPtr[0]->row,testImg->ImgMatPtr[0]->column,testLabel->LabelPtr[0].l, P, myrank);
        int trainNum=5000;
        trainModel_modelPrallel(cnn,testImg,testLabel,opts,trainNum, P, myrank, status);
    }else if(test==4){
        int P, myrank;
        int tag = 1;
        MPI_Init(&argc, &argv);
        MPI_Comm_size(MPI_COMM_WORLD, &P);
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        MPI_Status status;
        MPI_Barrier(MPI_COMM_WORLD);
        printf("[test] P%d p%d\n", P, myrank);
        LabelArr testLabel=read_Lable("Minst/train-labels.idx1-ubyte");
        ImgArr testImg=read_Img("Minst/train-images.idx3-ubyte");
        CNNOpts opts;
        opts.numepochs=1;
        opts.eta=0.1;

//        printf("Load image finished.\n");

        CNN* cnn=(CNN*)malloc(sizeof(CNN));
        cnn_setup(cnn,testImg->ImgMatPtr[0]->row,testImg->ImgMatPtr[0]->column,testLabel->LabelPtr[0].l, P, myrank);
        int trainNum=5000;
        trainModel_batch(cnn,testImg,testLabel,opts,trainNum, P, myrank, status);
    }

    return 0;
}
