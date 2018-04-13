//
// Created by crazybullet on 2018/4/12.
//

#include <rpcndr.h>
#include "matrix.c"

typedef struct convolutional_layer{
    int inputWidth;
    int inputHeight;
    int stride;        // mapSize = stride * stride

    int inChannels;   //输入图像的数目
    int outChannels;  //输出图像的数目

    // 关于特征模板的权重分布，这里是一个四维数组, 其大小为inChannels * outChannels * stride * stride
    // 这里用四维数组，主要是为了表现全连接的形式，实际上卷积层并没有用到全连接的形式
    double**** mapData;     //存放特征模块的数据
    double**** dmapData;    //存放特征模块的数据的局部梯度

    double* basicData;   //偏置，偏置的大小，为outChannels
    boolean isFullConnect; //是否为全连接
    boolean * connectModel; //连接模式（默认为全连接）

    // 下面三者的大小同输出的维度相同
    double*** v; // 进入激活函数的输入值          outChannels * (inputHeight - stride + 1) * (inputWidth - stride + 1)
    double*** y; // 激活函数后神经元的输出

    // 输出像素的局部梯度
    double*** d; // 网络的局部梯度,δ值
} CovLayer;

typedef struct pooling_layer{
    int inputWidth;
    int inputHeight;
    int stride;        // mapSize = stride * stride

    int inChannels;
    int outChannels;

    int poolType;       // max pooling / mean pooling
    double *bisas;

    double*** y; // output, without active
    double*** d; // local gradient
} PoolLayer;

typedef struct nn_layer{
    int inputNum;   //输入数据的数目  
    int outputNum;  //输出数据的数目  

    double** wData; // 权重数据，为一个inputNum*outputNum大小  
    double* basicData;   //偏置，大小为outputNum大小  

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

    float* e; // 训练误差
    float* L; // 瞬时误差能量
} CNN;

double activate(double num){
    return (num>0)?num:0;
};

CovLayer *initialCovLayer(int inputHeight, int inputWeight, int stride, int inChannels, int outChannels){
    CovLayer *covLayer = (CovLayer*) malloc(sizeof(CovLayer));
    covLayer->inputHeight = inputHeight;
    covLayer->inputWidth = inputWeight;
    covLayer->stride = stride;
    covLayer->inChannels = inChannels;
    covLayer->outChannels = outChannels;

    covLayer->mapData = (double****) malloc(out)
};

matrix *covolution(CovLayer C, matrix *input){

};