//
// Created by crazybullet on 2018/4/12.
//

#include <rpcndr.h>
#include <math.h>
#include <mapi.h>
#include "matrix.c"

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

    double* bias;   //偏置，偏置的大小，为outChannels
    boolean isFullConnect; //是否为全连接
    boolean *connectModel; //连接模式（默认为全连接）

    // 下面三者的大小同输出的维度相同
    matrix** v;     // 进入激活函数的输入值           outChannels * outputHeight * outputWidth
    matrix** y;     // 激活函数后神经元的输出

    // 输出像素的局部梯度
    matrix** d;     // 网络的局部梯度,δ值
} CovLayer;

typedef struct pooling_layer{
    int inputWidth;
    int inputHeight;
    int mapSize;

//    inChannels = outChannels
    int inChannels;
    int outChannels;

    int poolType;       // 0 - max pooling / 1 - mean pooling
    double *bias;

//    double*** y; // output, without active
    matrix** y;
//    double*** d; // local gradient
    matrix** d;
} PoolLayer;

typedef struct nn_layer{
    int inputNum;   //输入数据的数目
    int outputNum;  //输出数据的数目

//    double** weight; // 权重数据，为一个inputNum*outputNum大小
    matrix* weight;
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
    OutLayer* Out;

    double* e; // 训练误差
    double* L; // 瞬时误差能量
} CNN;

double activate(double num){
//    ReLu
    return (num>0)?num:0;
};


double acti_derivation(double y){
    return (y>0)?1:0;
}


void softMax(double* outArr, const double* inArr, int outNum){
    double sum=0, tmp;
    int i;
    for (i=0; i<outNum; i++){
        tmp = exp(inArr[i]);
        outArr[i] = tmp;
        sum += tmp;
    }
    for (i=0; i<outNum; i++){
        outArr[i] /= sum;
    }
};

CovLayer* initCovLayer(int inputHeight, int inputWidth, int mapSize, int inChannels, int outChannels, int paddingForward){
    CovLayer *covLayer = (CovLayer*) malloc(sizeof(CovLayer));
    covLayer->inputHeight = inputHeight;
    covLayer->inputWidth = inputWidth;
    covLayer->mapSize = mapSize;
    covLayer->inChannels = inChannels;
    covLayer->outChannels = outChannels;
    covLayer->paddingForward = paddingForward;

    covLayer->isFullConnect=1;

    covLayer->mapWeight = (matrix***) malloc(inChannels* sizeof(matrix**));
    covLayer->dmapWeight = (matrix***) malloc(inChannels*sizeof(matrix**));

    int i, j, k, l;

//    srand(100);

    for (i=0; i<inChannels; i++){
        covLayer->mapWeight[i] = (matrix**) malloc(outChannels* sizeof(matrix*));
        covLayer->dmapWeight[i] = (matrix**) malloc(outChannels* sizeof(matrix*));
        for (j=0; j<outChannels; j++){
            covLayer->mapWeight[i][j] = initMat(mapSize, mapSize, 0);
            covLayer->dmapWeight[i][j] = initMat(mapSize, mapSize, 0);
            for (k=0; k<mapSize; k++){
                for (l=0; l<mapSize; l++){
//                    covLayer->mapData[i][j][k][l] = (rand()/(double)(RAND_MAX+1)-0.5)*2 * sqrt(6.0/(mapSize*mapSize*inChannels+outChannels));             // xavier initialize
                    *getMatVal(covLayer->mapWeight[i][j], k, l) = (rand()/(double)(RAND_MAX+1)-0.5)*2 * sqrt(6.0/(mapSize*mapSize*inChannels+outChannels));
                }
            }
        }
    }

    covLayer->bias = (double*) malloc(inChannels*sizeof(double));

    int outW=inputWidth-mapSize+1+2*paddingForward;
    int outH=inputHeight-mapSize+1+2*paddingForward;
    covLayer->outputWidth = outW;
    covLayer->outputHeight = outH;

    covLayer->d=(matrix**)malloc(outChannels*sizeof(matrix*));
    covLayer->v=(matrix**)malloc(outChannels*sizeof(matrix*));
    covLayer->y=(matrix**)malloc(outChannels*sizeof(matrix*));
    for(j=0;j<outChannels;j++){
        covLayer->d[j]=initMat(outH, outW, 0);
        covLayer->v[j]=initMat(outH, outW, 1);
        // may need modify
        covLayer->y[j]=initMat(outH, outW, 0);
    }

    return covLayer;
};

PoolLayer* initPoolLayer(int inputHeight, int inputWidth, int mapSize, int inChannels, int outChannels, int poolType){
    PoolLayer* poolLayer = (PoolLayer*) malloc(sizeof(PoolLayer));
    poolLayer->inputWidth = inputWidth;
    poolLayer->inputHeight = inputWidth;
    poolLayer->mapSize = mapSize;
    poolLayer->inChannels = inChannels;
    poolLayer->outChannels = outChannels;
    poolLayer->poolType = poolType;

//    poolLayer->bias = (double*)malloc(outChannels* sizeof(double));

    int outW=inputWidth/mapSize;
    int outH=inputHeight/mapSize;
    int i,j;
    poolLayer->d = (matrix**) malloc(outChannels* sizeof(matrix*));
    poolLayer->y = (matrix**) malloc(outChannels* sizeof(matrix*));
    for(i=0;i<outChannels;i++){
        poolLayer->d[i]=initMat(outH, outW, 0);
        poolLayer->y[i]=initMat(outH, outW, 0);
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
    outLayer->p = (double*) malloc(outputNum*sizeof(double));

    int i,j;
    outLayer->bias = (double*) malloc(outputNum*sizeof(double));
    outLayer->weight = initMat(inputNum, outputNum, 0);
//    outLayer->dweight = initMat(inputNum, outputNum, 0);
//    srand(100);
    for (i=0; i<outputNum; i++){
        for (j=0; j<inputNum; j++){
            *getMatVal(outLayer->weight, i, j)=(rand()/(double)(RAND_MAX+1)-0.5)*2 * sqrt(6.0/(inputNum+outputNum));
        }
    }

    outLayer->isFullConnect=1;
    return outLayer;
};

void cnn_setup(CNN* cnn, int inputHeight, int inputWidth, int outNum){
    int inH = inputHeight;
    int inW = inputWidth;
    cnn->C1 = initCovLayer(inH, inW, 5, 1, 6, 2);       // 28*28 -> 32*32 -> 28*28
    inH = cnn->C1->outputHeight;
    inW = cnn->C1->outputWidth;
    cnn->S2 = initPoolLayer(inH, inW, 2, 6, 6, 0);      // 28*28 -> 14*14
    inH /= cnn->S2->mapSize;
    inW /= cnn->S2->mapSize;
    cnn->C3 = initCovLayer(inH, inW, 5, 6, 12, 0);      // 14*14 -> 10*10
    inH = cnn->C3->outputHeight;
    inW = cnn->C3->outputWidth;
    cnn->S4 = initPoolLayer(inH, inW, 2, 12, 12, 0);    // 10*10 -> 5*5
    inH /= cnn->S4->mapSize;
    inW /= cnn->S4->mapSize;
    cnn->Out = initOutLayer(inH*inW * 12, outNum);        // 300 -> 10

    cnn->e=(double *)malloc(cnn->Out->outputNum*sizeof(double));
};




void covolution_once(matrix* v, matrix* inMat, matrix* map, int outH, int outW, int padding){
    int mapSize = map->row;
    int i,j;

    if (padding==0){
        matrix* tmp = initMat(mapSize, mapSize, 0);
        for (i=0; i<outH; i++){
            for (j=0; j<outW; j++){
                subMat(tmp, inMat, i, mapSize, j, mapSize);
                double val = dotMatSum(tmp, map);
                *getMatVal(v, i, j)=val;
            }
        }
        freeMat(tmp);
        return;
    }else{
        matrix* newInputMat = initMat(inMat->row+2*padding, inMat->column+2*padding, 1);
        for (i=0; i<inMat->row; i++){
            for (j=0; j<inMat->column; j++){
                *getMatVal(newInputMat, i+padding, j+padding) = *getMatVal(inMat, i, j);
            }
        }

        matrix* tmp = initMat(mapSize, mapSize, 0);
        for (i=0; i<outH; i++){
            for (j=0; j<outW; j++){
                subMat(tmp, newInputMat, i, mapSize, j, mapSize);
                double val = dotMatSum(tmp, map);
                *getMatVal(v, i, j)=val;
            }
        }
        freeMat(tmp);
        return;
    }
};

void convolution(CovLayer* C, matrix** inMat){
    int i, j, k, l;
    matrix* tmpv = initMat(C->outputHeight, C->outputWidth, 0);

    for (i=0; i<C->outChannels; i++){
        for (j=0; j<C->inChannels; j++){
            covolution_once(tmpv, inMat[j], C->mapWeight[j][i], C->outputHeight, C->outputWidth, C->paddingForward);
            addMat_replace(C->v[i], tmpv);
//            clearMat(tmpv);       // not necessary
        }
        for (k=0; k<C->outputHeight; k++){
            for (l=0; l<C->outputWidth; l++){
                *getMatVal(C->v[i], k, l) += C->bias[i];
                *getMatVal(C->y[i], k, l) = activate(*getMatVal(C->v[i], k, l));
            }
        }
    }
    freeMat(tmpv);
};

void pooling_max(matrix* res, matrix* inMat, int mapSize){
    int i, j, k, l;
    for (i=0; i<inMat->row/mapSize; i++){
        for (j=0; j<inMat->column/mapSize; j++){
            double max = *getMatVal(inMat, i*mapSize, j*mapSize);
            for (k=0; k<mapSize; k++){
                for (l=0; l<mapSize; l++){
                    max = (max>*getMatVal(inMat, i*mapSize+k, j*mapSize+l))?max:*getMatVal(inMat, i*mapSize+k, j*mapSize+l);
                }
            }
            *getMatVal(res, i, j) = max;
        }
    }
};

void pooling_mean(matrix* res, matrix* inMat, int mapSize){
    int i,j,k,l;
    for (i=0; i<inMat->row/mapSize; i++) {
        for (j = 0; j < inMat->column / mapSize; j++) {
            double sum = 0.0;
            for (k = 0; k < mapSize; k++) {
                for (l = 0; l < mapSize; l++) {
                    sum += *getMatVal(inMat, i * mapSize + k, j * mapSize + l);
                }
            }
            *getMatVal(res,i,j) = sum/(mapSize*mapSize);
        }
    }
};

void pooling(PoolLayer* S, matrix** inMat){
    int i, j, k;
    for (i=0; i<S->inChannels; i++){
        if (S->poolType == 0){
            pooling_max(S->y[i], inMat[i], S->mapSize);
        }else{
            pooling_mean(S->y[i], inMat[i], S->mapSize);
        }
    }
};

void nnForward(OutLayer* O, const double* inArr){
    matrix inMat, vMat;     // no need to free
    inMat.row=1;
    inMat.column=O->inputNum;
    inMat.val = inArr;

    vMat.row=1;
    vMat.column=O->outputNum;
    vMat.val = O->v;            // modify vMat->val should also modify O->v->val

    mulMat(&vMat, &inMat, O->weight);
    int i;
    for (i=0; i<O->outputNum; i++){
        vMat.val[i] += O->bias[i];
//        O->v[i] = vMat.val[i];
        O->y[i] = activate(vMat.val[i]);
    }
    softMax(O->p, O->y, O->outputNum);
};

void cnnfw(CNN* cnn, matrix* inMat){

}

void trainModel(CNN* cnn){

}

