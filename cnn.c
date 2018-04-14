//
// Created by crazybullet on 2018/4/12.
//

#include <rpcndr.h>
#include <math.h>
#include "matrix.c"

typedef struct convolutional_layer{
    int inputWidth;
    int inputHeight;
    int mapSize;

    int inChannels;   //输入图像的数目
    int outChannels;  //输出图像的数目

    // 关于特征模板的权重分布，这里是一个四维数组, 其大小为inChannels * outChannels * mapSize * mapSize
    // 这里用四维数组，主要是为了表现全连接的形式，实际上卷积层并没有用到全连接的形式
//    double**** mapData;     //存放特征模块的数据
    matrix*** mapWeight;
//    double**** dmapData;    //存放特征模块的数据的局部梯度
    matrix*** dmapWeight;

    double* bias;   //偏置，偏置的大小，为outChannels
    boolean isFullConnect; //是否为全连接
    boolean *connectModel; //连接模式（默认为全连接）

    // 下面三者的大小同输出的维度相同
//    double*** v; // 进入激活函数的输入值          outChannels * (inputHeight - mapSize + 1) * (inputWidth - mapSize + 1)
    matrix** v;
//    double*** y; // 激活函数后神经元的输出
    matrix** y;

    // 输出像素的局部梯度
//    double*** d; // 网络的局部梯度,δ值
    matrix** d;
} CovLayer;

typedef struct pooling_layer{
    int inputWidth;
    int inputHeight;
    int mapSize;

    int inChannels;
    int outChannels;

    int poolType;       // max pooling / mean pooling
    double *bias;

//    double*** y; // output, without active
    matrix** y;
//    double*** d; // local gradient
    matrix** d;
} PoolLayer;

typedef struct nn_layer{
    int inputNum;   //输入数据的数目
    int outputNum;  //输出数据的数目

    double** weight; // 权重数据，为一个inputNum*outputNum大小
    double* bias;   //偏置，大小为outputNum大小

    // 下面三者的大小同输出的维度相同
    double* v; // 进入激活函数的输入值
    double* y; // 激活函数后神经元的输出
    double* d; // 网络的局部梯度,δ值

    boolean isFullConnect; //是否为全连接
} OutLayer;

typedef struct cnn_network{
    int layerNum;
    CovLayer* C1;
    PoolLayer* S2;
    CovLayer* C3;
    PoolLayer* S4;
    OutLayer* O5;

    double* e; // 训练误差
    double* L; // 瞬时误差能量
} CNN;

double activate(double num){
    return (num>0)?num:0;
};

CovLayer* initCovLayer(int inputHeight, int inputWidth, int mapSize, int inChannels, int outChannels){
    CovLayer *covLayer = (CovLayer*) malloc(sizeof(CovLayer));
    covLayer->inputHeight = inputHeight;
    covLayer->inputWidth = inputWidth;
    covLayer->mapSize = mapSize;
    covLayer->inChannels = inChannels;
    covLayer->outChannels = outChannels;

    covLayer->isFullConnect=1;

    covLayer->mapWeight = (matrix***) malloc(inChannels* sizeof(matrix**));
    covLayer->dmapWeight = (matrix***) malloc(inChannels*sizeof(matrix**));

    int i, j, k, l;

//    srand(100);

    for (i=0; i<inChannels; i++){
        covLayer->mapWeight[i] = (matrix**) malloc(outChannels* sizeof(matrix*));
        covLayer->dmapWeight[i] = (matrix**) malloc(outChannels* sizeof(matrix*));
        for (j=0; j<outChannels; j++){
            covLayer->mapWeight[i][j] = initMat(mapSize, mapSize);
            covLayer->dmapWeight[i][j] = initMat(mapSize, mapSize);
            for (k=0; k<mapSize; k++){
                for (l=0; l<mapSize; l++){
//                    covLayer->mapData[i][j][k][l] = (rand()/(double)(RAND_MAX+1)-0.5)*2 * sqrt(6.0/(mapSize*mapSize*inChannels+outChannels));             // xavier initialize
                    *getMat(covLayer->mapWeight[i][j], k, l) = (rand()/(double)(RAND_MAX+1)-0.5)*2 * sqrt(6.0/(mapSize*mapSize*inChannels+outChannels));
                }
            }
        }
    }

    covLayer->bias = (double*) malloc(inChannels*sizeof(double));

    int outW=inputWidth-mapSize+1;
    int outH=inputHeight-mapSize+1;

    covLayer->d=(matrix**)malloc(outChannels*sizeof(matrix*));
    covLayer->v=(matrix**)malloc(outChannels*sizeof(matrix*));
    covLayer->y=(matrix**)malloc(outChannels*sizeof(matrix*));
    for(j=0;j<outChannels;j++){
        covLayer->d[j]=initMat(outH, outW);
        covLayer->v[j]=initMat(outH, outW);
        covLayer->y[j]=initMat(outH, outW);
    }

    return covLayer;
};

PoolLayer* initPoolLayer(int inputWidth, int inputHeight, int mapSize, int inChannels, int outChannels, int poolType){
    PoolLayer* poolLayer = (PoolLayer*) malloc(sizeof(PoolLayer));
    poolLayer->inputWidth = inputWidth;
    poolLayer->inputHeight = inputWidth;
    poolLayer->mapSize = mapSize;
    poolLayer->inChannels = inChannels;
    poolLayer->outChannels = outChannels;
    poolLayer->poolType = poolType;

    poolLayer->bias = (double*)malloc(outChannels* sizeof(double));

    int outW=inputWidth/mapSize;
    int outH=inputHeight/mapSize;
    int i,j;
    poolLayer->d=(double***)malloc(outChannels*sizeof(double**));
    poolLayer->y=(double***)malloc(outChannels*sizeof(double**));
    for(i=0;i<outChannels;i++){
        poolLayer->d[i]=(double**)malloc(outH*sizeof(double*));
        poolLayer->y[i]=(double**)malloc(outH*sizeof(double*));
        for(j=0;j<outH;j++){
            poolLayer->d[i][j]=(double*)malloc(outW*sizeof(double));
            poolLayer->y[i][j]=(double*)malloc(outW*sizeof(double));
        }
    }

    return poolLayer;
};

OutLayer* initOutLayer(int inputNum,int outputNum){
    OutLayer* outLayer = (OutLayer*) malloc(sizeof(OutLayer));
    outLayer->inputNum = inputNum;
    outLayer->outputNum = outputNum;

    outLayer->d = (double*) malloc(outputNum*sizeof(double));
    outLayer->v = (double*) malloc(outputNum*sizeof(double));
    outLayer->y = (double*) malloc(outputNum*sizeof(double));

    int i,j;
    outLayer->bias = (double*) malloc(outputNum*sizeof(double));
    outLayer->weight = (double**) malloc(outputNum*sizeof(double*));
    srand(100);
    for (i=0; i<outputNum; i++){
        outLayer->weight[i] = (double*) malloc(inputNum*sizeof(double));
        for (j=0; j<inputNum; j++){
            outLayer->weight[i][j]=(rand()/(double)(RAND_MAX+1)-0.5)*2 * sqrt(6.0/(inputNum+outputNum));
        }
    }

    outLayer->isFullConnect=1;
    return outLayer;
};

matrix *covolution(matrix* inputMat, matrix* map){

};

void train(CNN* cnn, ImgArr )