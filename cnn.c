//
// Created by crazybullet on 2018/4/12.
//

#include <rpcndr.h>
#include <math.h>
#include <mapi.h>
#include "matrix.c"
#include "cnn.h"

double activate(double num){
//    ReLu
    return (num>0)?num:0;
};


double acti_derivation(double y){
    return (y>0)?1:0;
}

matrix* softmax_classifier(matrix* input){
    int i,j;
    double sum;
    matrix *res = initMat(input->row, input->column, 1);
    for(i=0;i<input->row;i++){
        sum=0;
        for(j=0;j<input->row;j++){
            sum += exp(*getMatVal(input,j,0));
        }
        res->val[i] = exp(*getMatVal(input,i,0))/sum;
    }
    return res;
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

    int i,j;
    outLayer->bias = (double*) malloc(outputNum*sizeof(double));
    outLayer->weight = initMat(inputNum, outputNum, 0);
    outLayer->dweight = initMat(inputNum, outputNum, 0);
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
    cnn->C3 = initCovLayer(inH, inW, 5, 6, 16, 0);      // 14*14 -> 10*10
    inH = cnn->C3->outputHeight;
    inW = cnn->C3->outputWidth;
    cnn->S4 = initPoolLayer(inH, inW, 2, 16, 16, 0);    // 10*10 -> 5*5
    inH /= cnn->S4->mapSize;
    inW /= cnn->S4->mapSize;
    cnn->Out = initOutLayer(inH*inW*16, outNum);        // 400 -> 10

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
};


void nnBackward(CNN* cnn, double* outputData) {
    int i, j, c, r;
    for (i = 0; i < cnn->Out->outputNum; i++) {
        cnn->e[i] = cnn->Out->y[i] - outputData[i];
    }
    //2 fully connected layer
    for (i = 0;)

};
