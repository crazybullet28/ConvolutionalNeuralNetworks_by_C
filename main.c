#include <stdio.h>
#include <assert.h>
#include "cnn.h"

int main() {
    srand(100);
//    const char parentPath[] = "E:/MyDocuments/Documents/CLionProjects/Parallel_Proj/";

    LabelArr testLabel=read_Lable("E:/MyDocuments/Documents/CLionProjects/Parallel_Proj/Minst/train-labels.idx1-ubyte");
    ImgArr testImg=read_Img("E:/MyDocuments/Documents/CLionProjects/Parallel_Proj/Minst/train-images.idx3-ubyte");

//    int outSize=testLabel->LabelPtr[0].l;

    CNN* cnn=(CNN*)malloc(sizeof(CNN));
    cnn_setup(cnn,testImg->ImgMatPtr[0]->row,testImg->ImgMatPtr[0]->column,testLabel->LabelPtr[0].l);

    CNNOpts opts;
    opts.numepochs=1;
    opts.eta=1;
    int trainNum=5000;
    trainModel(cnn,testImg,testLabel,opts,trainNum);
//    testModel()

    FILE  *fp=NULL;
    fp=fopen("E:/MyDocuments/Documents/CLionProjects/Parallel_Proj/PicTrans/cnnL.ma","wb");
    if(fp==NULL)
        printf("write file failed\n");
    fwrite(cnn->L,sizeof(float),trainNum,fp);
    fclose(fp);


    return 0;
}

