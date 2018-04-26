//
// Created by crazybullet on 2018/4/15.
//

#ifndef PARALLEL_PROJ_CNN_H
#define PARALLEL_PROJ_CNN_H

#include "minst.h"
#include "stdbool.h"
#include "matrix.h"

typedef struct convolutional_layer{
    int inputWidth;
    int inputHeight;
    int mapSize;
    int paddingForward;
    int paddingBack;
    int outputWidth;
    int outputHeight;

    int inChannels;   //输入图像的数目
    int outChannels;  //输出图像的数目

    // 关于特征模板的权重分布，这里是一个四维数组, 其大小为inChannels * outChannels * mapSize * mapSize
    // 这里用四维数组，主要是为了表现全连接的形式，实际上卷积层并没有用到全连接的形式
    matrix*** mapWeight;
    matrix*** dmapWeight;

    float* bias;   //偏置，偏置的大小，为outChannels
    bool isFullConnect; //是否为全连接
    bool *connectModel; //连接模式（默认为全连接）

    // 下面三者的大小同输出的维度相同
    matrix** v;     // 进入激活函数的输入值           outChannels * outputHeight * outputWidth
    matrix** y;     // 激活函数后神经元的输出

    // 输出像素的局部梯度
    matrix** d;     // 网络的局部梯度,δ值              outCHannels * (outputHeight * outputWidth)
} CovLayer;

typedef struct pooling_layer{
    int inputWidth;
    int inputHeight;
    int mapSize;
    int outputWidth;
    int outputHeight;

//    inChannels = outChannels
    int inChannels;
    int outChannels;

    int poolType;       // 0 - max pooling / 1 - mean pooling
    float *bias;

//    float*** y; // output, without active
    matrix** y;
//    float*** d; // local gradient
    matrix** d;                          // outCHannels * (outputHeight * outputWidth)
} PoolLayer;

typedef struct nn_layer{
    int inputNum;   //输入数据的数目
    int outputNum;  //输出数据的数目

//    float** weight; // 权重数据，为一个inputNum*outputNum大小
    matrix* weight;
    float* bias;   //偏置，大小为outputNum大小

//    matrix* dweight;   // 权重梯度

    // 下面三者的大小同输出的维度相同
    float* v; // 进入激活函数的输入值
//    float* y; // 激活函数后神经元的输出
    float* p; // softMax(y)
    float* d; // 网络的局部梯度,δ值                 outputNum

    bool isFullConnect; //是否为全连接
} OutLayer;

typedef struct cnn_network{
    int layerNum;
    CovLayer* C1;
    PoolLayer* S2;
    CovLayer* C3;
    PoolLayer* S4;
    OutLayer* Out;

    float* e; // 训练误差
    float* L; // 瞬时误差能量
} CNN;

typedef struct train_opts{
    int numepochs; // 训练的迭代次数
    float eta; // 学习速率
}CNNOpts;

CovLayer* initCovLayer(int inputHeight, int inputWidth, int mapSize, int inChannels, int outChannels, int paddingForward);

PoolLayer* initPoolLayer(int inputHeight, int inputWidth, int mapSize, int inChannels, int outChannels, int poolType);

OutLayer* initOutLayer(int inputNum,int outputNum);

void cnn_setup(CNN* cnn, int inRow, int inCol, int outNum);

float activate(float num);

float acti_derivation(float y);

void softMax(float* outArr, const float* inArr, int outNum);

void convolution_once(matrix *v, matrix *inMat, matrix *map, int outH, int outW, int padding);

void convolution(CovLayer* C, matrix** inMat);

void pooling_max(matrix* res, matrix* inMat, int mapSize);

void pooling_mean(matrix* res, matrix* inMat, int mapSize);

void pooling(PoolLayer* S, matrix** inMat);

void nnForward(OutLayer* O, float* inArr);

float computeLoss(float* outArr, int labely);

void cnnfw(CNN* cnn, matrix* inMat);

matrix* UpSample(matrix* mat,int multiple_c,int multiple_r,int mapsize);

void cnnbp(CNN* cnn, float* outputData);

void gradient_update(CNN* cnn, CNNOpts opts, matrix* inMat);

void cnnclear(CNN* cnn);

void trainModel(CNN* cnn, ImgArr inputData, LabelArr outputData, CNNOpts opts, int trainNum);

float testModel(CNN* cnn, ImgArr inputData, LabelArr outputData, int testNum);

void cnnSaveOutput(CNN *cnn, matrix *inMat, const char *filename);

void cnnSaveWeight(CNN *cnn, const char *filename);

void cnnSaveD(CNN *cnn, const char *filename);

void cnnSaveDWeight(CNN *cnn, const char *filename);

#endif //PARALLEL_PROJ_CNN_H
