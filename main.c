#include <stdio.h>
#include <assert.h>
#include "cnn.h"
#include "mpi.h"

int main(int argc, char **argv) {
//    test 0: test serial cnn
//    test 1: test mpi matrix
//    test 2: test forward

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
    return 0;
}

