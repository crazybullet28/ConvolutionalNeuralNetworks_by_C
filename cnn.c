//
// Created by crazybullet on 2018/4/12.
//

#include <rpcndr.h>
#include <math.h>
#include <mapi.h>
#include <mem.h>
#include "matrix.c"
#include "cnn.h"
#include "minst.h"


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
    poolLayer->outputWidth = outW;
    poolLayer->outputHeight = outH;
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
    inH = cnn->S2->outputHeight;
    inW = cnn->S2->outputWidth;
    cnn->C3 = initCovLayer(inH, inW, 5, 6, 12, 0);      // 14*14 -> 10*10
    inH = cnn->C3->outputHeight;
    inW = cnn->C3->outputWidth;
    cnn->S4 = initPoolLayer(inH, inW, 2, 12, 12, 0);    // 10*10 -> 5*5
    inH = cnn->S4->outputHeight;
    inW = cnn->S4->outputWidth;
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

void nnForward(OutLayer* O, double* inArr){
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

double computeLoss(CNN* cnn, ){};

void cnnfw(CNN* cnn, matrix* inMat){       // only one matrix a time.           outArr has already been malloc
    matrix** input = (matrix**)malloc(sizeof(matrix*));
    input[0] = inMat;
    convolution(cnn->C1, input);
    pooling(cnn->S2, cnn->C1->y);
    convolution(cnn->C3, cnn->S2->y);
    pooling(cnn->S4, cnn->C3->y);

    double nn_input[cnn->Out->inputNum];
    int i;
    int tmpLength = cnn->S4->outputWidth * cnn->S4->outputHeight
    for (i=0; i<cnn->S4->outChannels; i++){
        memcpy(&nn_input[i*tmpLength], cnn->S4->y[i], tmpLength*sizeof(double));
    }
    nnForward(cnn->Out, nn_input);

    freeMat(input[0]);
    free(input);
}

matrix* UpSample(matrix* mat,int multiple_c,int multiple_r,int mapsize)
{
    int i,j,m,n;
    matrix* res= initMat(mapsize*multiple_r, mapsize*multiple_c, 1); // size initialization
    for(j=0;j<mapsize*multiple_r;j=j+mapsize){
        for(i=0;i<mapsize*multiple_c;i=i+mapsize)// 宽的扩充
            for(m=0;m<mapsize;m++)
                *getMatVal(res,j,i+m)=*getMatVal(mat,j/mapsize,i/mapsize)/(double)(mapsize*mapsize);

        for(n=1;n<mapsize;n++)      //  高的扩充
            for(i=0;i<multiple_c*mapsize;i++)
                *getMatVal(res,j+n,i)=*getMatVal(res,j,i)/(double)(mapsize*mapsize);
    }
    return res;
}


void cnnbp(CNN* cnn,double* outputData)
{
    int i,j,c,r; // 将误差保存到网络中
    for(i=0;i<cnn->Out->outputNum;i++)
        cnn->e[i]=cnn->Out->y[i]-outputData[i];

    // Output layer
    for(i=0;i<cnn->Out->outputNum;i++)
        cnn->Out->d[i]=cnn->e[i]*acti_derivation(cnn->Out->y[i]);

    // S4层，传递到S4层的误差
    // 这里没有激活函数
    int row = cnn->S4->inputHeight/cnn->S4->mapSize;
    int col = cnn->S4->inputWidth/cnn->S4->mapSize;
    for(i=0;i<cnn->S4->outChannels;i++)
        for(r=0;r<row;r++)
            for(c=0;c<col;c++)
                for(j=0;j<cnn->Out->outputNum;j++){
                    int wInt=i*col*row+r*col+c;
                    *getMatVal(cnn->S4->d[i], r, c) = *getMatVal(cnn->S4->d[i],r,c)+cnn->Out->d[j]*(*getMatVal(cnn->Out->weight,i,wInt));
                }

    // C3层
    // 由S4层传递的各反向误差,这里只是在S4的梯度上扩充一倍
    int mapdata=cnn->S4->mapSize;
    // 这里的Pooling是求平均，
    for(i=0;i<cnn->C3->outChannels;i++){
        matrix* C3e = UpSample(cnn->S4->d[i],cnn->S4->inputWidth/cnn->S4->mapSize,cnn->S4->inputHeight/cnn->S4->mapSize,cnn->S4->mapSize);
        for(r=0;r<cnn->S4->inputHeight;r++)
            for(c=0;c<cnn->S4->inputWidth;c++)
                *getMatVal(cnn->C3->d[i],r,c)=*getMatVal(C3e,r,c)*acti_derivation(*getMatVal(cnn->C3->y[i],r,c));
        freeMat(C3e);
    }

    // S2层，S2层没有激活函数，这里只有卷积层有激活函数部分
    // 由卷积层传递给采样层的误差梯度，这里卷积层共有6*12个卷积模板
    int output_c=cnn->C3->inputWidth;
    int output_r=cnn->C3->inputHeight;
    int input_c=cnn->S4->inputWidth;
    int input_r=cnn->S4->inputHeight;
    int mapsize_C3=cnn->C3->mapSize;
    for(i=0;i<cnn->S2->outChannels;i++){
        for(j=0;j<cnn->C3->outChannels;j++){
            float** corr=correlation(cnn->C3->mapWeight[i][j],mapSize,cnn->C3->d[j],inSize,full);
            addmat(cnn->S2->d[i],cnn->S2->d[i],outSize,corr,outSize);
            for(r=0;r<outSize.r;r++)
                free(corr[r]);
            free(corr);
        }
        /*
        for(r=0;r<cnn->C3->inputHeight;r++)
            for(c=0;c<cnn->C3->inputWidth;c++)
                // 这里本来用于采样的激活
        */
    }

    // C1层，卷积层
    mapdata=cnn->S2->mapSize;
    nSize S2dSize={cnn->S2->inputWidth/cnn->S2->mapSize,cnn->S2->inputHeight/cnn->S2->mapSize};
    // 这里的Pooling是求平均，所以反向传递到下一神经元的误差梯度没有变化
    for(i=0;i<cnn->C1->outChannels;i++){
        float** C1e=UpSample(cnn->S2->d[i],S2dSize,cnn->S2->mapSize,cnn->S2->mapSize);
        for(r=0;r<cnn->S2->inputHeight;r++)
            for(c=0;c<cnn->S2->inputWidth;c++)
                cnn->C1->d[i][r][c]=C1e[r][c]*sigma_derivation(cnn->C1->y[i][r][c])/(float)(cnn->S2->mapSize*cnn->S2->mapSize);
        for(r=0;r<cnn->S2->inputHeight;r++)
            free(C1e[r]);
        free(C1e);
    }
}

void trainModel(CNN* cnn, ImgArr inputData, LabelArr outputData, CNNOpts opts,int trainNum){
    cnn->L=(double *)malloc(trainNum*sizeof(double));
    int e;
    for (e=0; e<opts.numepochs; e++){
        int n;
        for (n=0; n<trainNum; n++){
//            matrix* inputMat = defMat(inputData->ImgPtr[n].ImgData, inputData->ImgPtr[n].r, inputData->ImgPtr[n].c);
//            freeMat(inputMat);
            cnnfw(cnn, inputData->ImgPtr[n]);



        }
    }
}

