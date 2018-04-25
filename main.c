#include <stdio.h>
#include <assert.h>
#include "cnn.h"

int main() {
    srand(100);

    LabelArr testLabel=read_Lable("E:/MyDocuments/Documents/CLionProjects/Parallel_Proj/Minst/train-labels.idx1-ubyte");
    ImgArr testImg=read_Img("E:/MyDocuments/Documents/CLionProjects/Parallel_Proj/Minst/train-images.idx3-ubyte");

//    int outSize=testLabel->LabelPtr[0].l;

    CNN* cnn=(CNN*)malloc(sizeof(CNN));
    cnn_setup(cnn,testImg->ImgMatPtr[0]->row,testImg->ImgMatPtr[0]->column,testLabel->LabelPtr[0].l);

    CNNOpts opts;
    opts.numepochs=10;
    opts.eta=0.001;
    opts.batch = 10;
    int trainNum=10000;
    trainModel(cnn,testImg,testLabel,opts,trainNum);
//    testModel()

//    FILE  *fp=NULL;
//    fp=fopen("data/PicTrans/cnnL.ma","wb");
//    if(fp==NULL)
//        printf("write file failed\n");
//    fwrite(cnn->L,sizeof(float),trainNum,fp);
//    fclose(fp);

//    matrix *v, *inMat, *map;
//
//    v = initMat(5, 5, 1);
//    inMat = initMat(5, 5, 0);
//    map = initMat(5, 5, 0);
//
//    int i, j;
//
//    for (i=0; i<25; i++){
//        inMat->val[i] = i;
//        map->val[i] = 1;
//    }
//
//    for (i=0; i<5; i++){
//        for (j=0; j<5; j++){
//            printf("%.4f ", *getMatVal(inMat, i, j));
//        }
//        printf("\n");
//    }
//
//    printf("\n");
//
//    for (i=0; i<5; i++){
//        for (j=0; j<5; j++){
//            printf("%.4f ", *getMatVal(map, i, j));
//        }
//        printf("\n");
//    }
//
//    printf("\n");
//
//    covolution_once(v, inMat, map, 5, 5, 2);
//
//    for (i=0; i<5; i++){
//        for (j=0; j<5; j++){
//            printf("%.4f ", *getMatVal(v, i, j));
//        }
//        printf("\n");
//    }

    return 0;
}

