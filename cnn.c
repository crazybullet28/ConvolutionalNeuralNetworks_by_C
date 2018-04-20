//
// Created by crazybullet on 2018/4/12.
//

#include <math.h>
//#include <mapi.h>
//#include <mem.h>
#include "cnn.h"


float activate(float num){
//    ReLu
    return (num>0)?num:0;
};

//float activate_sigmoid(float num){
//    return exp(num)/(exp(num)+1.0);
//}

float acti_derivation(float y){
    return (y>0)?1:0;
}


void softMax(float* outArr, const float* inArr, int outNum){
    float sum=0, tmp;
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
//                    covLayer->mapData[i][j][k][l] = (rand()/(float)(RAND_MAX+1)-0.5)*2 * sqrt(6.0/(mapSize*mapSize*inChannels+outChannels));             // xavier initialize
                    *getMatVal(covLayer->mapWeight[i][j], k, l) = (rand()/(float)(RAND_MAX+1)-0.5)*2 * sqrt(6.0/(mapSize*mapSize*inChannels+outChannels));
                }
            }
        }
    }

    covLayer->bias = (float*) malloc(inChannels*sizeof(float));

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

//    poolLayer->bias = (float*)malloc(outChannels* sizeof(float));

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

    outLayer->d = (float*) malloc(outputNum*sizeof(float));
    outLayer->v = (float*) malloc(outputNum*sizeof(float));
    outLayer->y = (float*) malloc(outputNum*sizeof(float));
    outLayer->p = (float*) malloc(outputNum*sizeof(float));

    int i,j;
    outLayer->bias = (float*) malloc(outputNum*sizeof(float));
    outLayer->weight = initMat(inputNum, outputNum, 0);
//    outLayer->dweight = initMat(inputNum, outputNum, 0);
//    srand(100);
    for (i=0; i<inputNum; i++){
        for (j=0; j<outputNum; j++){
            *getMatVal(outLayer->weight, i, j)=(rand()/(float)(RAND_MAX+1)-0.5)*2 * sqrt(6.0/(inputNum+outputNum));
        }
    }

    outLayer->isFullConnect=1;
    return outLayer;
};

void cnn_setup(CNN* cnn, int inputHeight, int inputWidth, int outNum){
    printf("start cnn_setup\n");
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

    cnn->e=(float *)malloc(cnn->Out->outputNum*sizeof(float));
    printf("end trainModel\n");
};




void covolution_once(matrix* v, matrix* inMat, matrix* map, int outH, int outW, int padding){
    int mapSize = map->row;
    int i,j;

    if (padding==0){
        matrix* tmp = initMat(mapSize, mapSize, 0);
        for (i=0; i<outH; i++){
            for (j=0; j<outW; j++){
                subMat(tmp, inMat, i, mapSize, j, mapSize);
                float val = dotMatSum(tmp, map);
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
                float val = dotMatSum(tmp, map);
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
            float max = *getMatVal(inMat, i*mapSize, j*mapSize);
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
            float sum = 0.0;
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

void nnForward(OutLayer* O, float* inArr){
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
//        O->y[i] = activate(vMat.val[i]);
        O->y[i] = vMat.val[i];
    }
    softMax(O->p, O->y, O->outputNum);
};

float computeLoss(float* outArr, int labely){
    return -log(outArr[labely]);
};

void cnnfw(CNN* cnn, matrix* inMat){       // only one matrix a time.           outArr has already been malloc
    printf("start cnnfw\n");
    matrix** input = (matrix**)malloc(sizeof(matrix*));
    input[0] = inMat;
    convolution(cnn->C1, input);
    pooling(cnn->S2, cnn->C1->y);
    convolution(cnn->C3, cnn->S2->y);
    pooling(cnn->S4, cnn->C3->y);

    float nn_input[cnn->Out->inputNum];
    int i;
    int tmpLength = cnn->S4->outputWidth * cnn->S4->outputHeight;
    for (i=0; i<cnn->S4->outChannels; i++){
        memcpy(&nn_input[i*tmpLength], cnn->S4->y[i], tmpLength*sizeof(float));
    }
    nnForward(cnn->Out, nn_input);

    freeMat(input[0]);
    free(input);
    printf("end cnnfw\n");
}

matrix* UpSample(matrix* mat,int multiple_c,int multiple_r,int mapsize){
    int i,j,m,n;
    matrix* res= initMat(mapsize*multiple_r, mapsize*multiple_c, 1); // size initialization
    for(j=0;j<mapsize*multiple_r;j=j+mapsize){
        for(i=0;i<mapsize*multiple_c;i=i+mapsize)// 宽的扩充
            for(m=0;m<mapsize;m++)
                *getMatVal(res,j,i+m)=*getMatVal(mat,j/mapsize,i/mapsize)/(float)(mapsize*mapsize);

        for(n=1;n<mapsize;n++)      //  高的扩充
            for(i=0;i<multiple_c*mapsize;i++)
                *getMatVal(res,j+n,i)=*getMatVal(res,j,i)/(float)(mapsize*mapsize);
    }
    return res;
}

void cnnbp(CNN* cnn, float* outputData){
    printf("start cnnbw\n");
    int i,j,c,r; // 将误差保存到网络中
    for(i=0;i<cnn->Out->outputNum;i++)
        cnn->e[i]=cnn->Out->y[i]-outputData[i];

    // Output layer
    for(i=0;i<cnn->Out->outputNum;i++)
        cnn->Out->d[i]=cnn->e[i];

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
    // 这里的Pooling是求平均，
    for(i=0;i<cnn->C3->outChannels;i++){
        matrix* C3e = UpSample(cnn->S4->d[i],cnn->S4->inputWidth/cnn->S4->mapSize,cnn->S4->inputHeight/cnn->S4->mapSize,cnn->S4->mapSize);
        for(r=0;r<cnn->S4->inputHeight;r++)
            for(c=0;c<cnn->S4->inputWidth;c++)
                *getMatVal(cnn->C3->d[i],r,c)=*getMatVal(C3e,r,c)*acti_derivation(*getMatVal(cnn->C3->v[i],r,c));
        freeMat(C3e);
    }

    // S2层，S2层没有激活函数，这里只有卷积层有激活函数部分
    // 由卷积层传递给采样层的误差梯度，这里卷积层共有6*16个卷积模板
    int output_c=cnn->C3->inputWidth;
    int output_r=cnn->C3->inputHeight;
    int input_c=cnn->S4->inputWidth;
    int input_r=cnn->S4->inputHeight;
    int mapsize_C3=cnn->C3->mapSize;
    int padding_size = mapsize_C3 -1;

    matrix* sum = initMat(output_r, output_c, 1);
    for(i=0;i<cnn->S2->outChannels;i++){
        for(j=0;j<cnn->C3->outChannels;j++){
            matrix* rot_weight = initMat(mapsize_C3,mapsize_C3,1);
            rotate180Mat(rot_weight,cnn->C3->mapWeight[i][j]);
            covolution_once(cnn->S2->d[i], cnn->C3->d[j], rot_weight, output_r, output_c, padding_size);
            addMat(sum,cnn->S2->d[i],sum);
            freeMat(rot_weight);
        }
        cnn->S2->d[i] = sum;
        /*
        for(r=0;r<cnn->C3->inputHeight;r++)
            for(c=0;c<cnn->C3->inputWidth;c++)
                // 这里本来用于采样的激活
        */
    }

    // C1层，卷积层
    // 这里的Pooling是求平均，所以反向传递到下一神经元的误差梯度没有变化
    for(i=0;i<cnn->C1->outChannels;i++){
        matrix* C1e = UpSample(cnn->S2->d[i],cnn->S2->inputWidth/cnn->S2->mapSize,cnn->S2->inputHeight/cnn->S2->mapSize,cnn->S2->mapSize);
        for(r=0;r<cnn->S2->inputHeight;r++)
            for(c=0;c<cnn->S2->inputWidth;c++)
                *getMatVal(cnn->C1->d[i],r,c)=*getMatVal(C1e,r,c)*acti_derivation(*getMatVal(cnn->C1->v[i],r,c));
        free(C1e);
    }

    printf("end cnnbw\n");
}


void gradient_update(CNN* cnn, CNNOpts opts, matrix* inMat){
    int i,j,r,c;

    //C1
    for(i=0;i<cnn->C1->outChannels;i++){
        for(j=0;j<cnn->C1->inChannels;j++){
            matrix* rot_input = initMat(cnn->C1->inputHeight,cnn->C1->inputWidth,1);
            rotate180Mat(rot_input,inMat);
            covolution_once(cnn->C1->dmapWeight[j][i], rot_input,cnn->C1->d[i], cnn->C1->mapSize, cnn->C1->mapSize, 0);
            matrix* minus_weight = initMat(cnn->C1->mapSize,cnn->C1->mapSize,1);
            mulMatVal(minus_weight, cnn->C1->dmapWeight[j][i], -1*opts.eta);
            addMat_replace(cnn->C1->mapWeight[j][i],minus_weight);

            freeMat(rot_input);
            freeMat(minus_weight);
        }
        cnn->C1->bias[i] = cnn->C1->bias[i] - opts.eta*sumMat(cnn->C1->d[i]);
    }

    //C3
    for(i=0;i<cnn->C3->outChannels;i++){
        for(j=0;j<cnn->C3->inChannels;j++){
            matrix* rot_input = initMat(cnn->S2->outputHeight,cnn->S2->outputWidth,1);
            rotate180Mat(rot_input,cnn->S2->y[j]);
            covolution_once(cnn->C3->dmapWeight[j][i], rot_input,cnn->C3->d[i], cnn->C3->mapSize, cnn->C3->mapSize, 0);
            matrix* minus_weight = initMat(cnn->C3->mapSize,cnn->C3->mapSize,1);
            mulMatVal(minus_weight, cnn->C3->dmapWeight[j][i], -1*opts.eta);
            addMat_replace(cnn->C3->mapWeight[j][i],minus_weight);

            freeMat(rot_input);
            freeMat(minus_weight);
        }
        cnn->C3->bias[i] = cnn->C3->bias[i] - opts.eta*sumMat(cnn->C3->d[i]);
    }

    //Output layer
    float* Out_input=(float*)malloc((cnn->Out->inputNum)*sizeof(float));
    int row = cnn->S4->inputHeight/cnn->S4->mapSize;
    int col = cnn->S4->inputWidth/cnn->S4->mapSize;
    for(i=0;i<cnn->S4->outChannels;i++)
        for(r=0;r<row;r++)
            for(c=0;c<col;c++)
                Out_input[i*row*col+r*col+c] = *getMatVal(cnn->S4->y[i],r,c);
    for(j=0;j<cnn->Out->outputNum;j++){
        for(i=0;i<cnn->Out->inputNum;i++)
            *getMatVal(cnn->Out->weight,j,i) = *getMatVal(cnn->Out->weight,j,i) - opts.eta*cnn->Out->d[j]*Out_input[i];
        cnn->Out->bias[j] = cnn->Out->bias[j] - opts.eta*cnn->Out->d[j];
    }
    free(Out_input);
}

void cnnclear(CNN* cnn){
    // Clear local error and output
    int j,c,r;
    // C1
    for(j=0;j<cnn->C1->outChannels;j++){
        clearMat(cnn->C1->d[j]);
        clearMat(cnn->C1->v[j]);
        clearMat(cnn->C1->y[j]);
//        for(r=0;r<cnn->S2->inputHeight;r++){
//            for(c=0;c<cnn->S2->inputWidth;c++){
//                *getMatVal(cnn->C1->d[j],r,c)=(float)0.0;
//                *getMatVal(cnn->C1->v[j],r,c)=(float)0.0;
//                *getMatVal(cnn->C1->y[j],r,c)=(float)0.0;
//            }
//        }
    }
    // S2
    for(j=0;j<cnn->S2->outChannels;j++){
        clearMat(cnn->S2->d[j]);
        clearMat(cnn->S2->y[j]);
//        for(r=0;r<cnn->C3->inputHeight;r++){
//            for(c=0;c<cnn->C3->inputWidth;c++){
//                *getMatVal(cnn->S2->d[j],r,c)=(float)0.0;
//                *getMatVal(cnn->S2->y[j],r,c)=(float)0.0;
//            }
//        }
    }
    // C3
    for(j=0;j<cnn->C3->outChannels;j++){
        clearMat(cnn->C3->d[j]);
        clearMat(cnn->C3->v[j]);
        clearMat(cnn->C3->y[j]);
//        for(r=0;r<cnn->S4->inputHeight;r++){
//            for(c=0;c<cnn->S4->inputWidth;c++){
//                *getMatVal(cnn->C3->d[j],r,c)=(float)0.0;
//                *getMatVal(cnn->C3->v[j],r,c)=(float)0.0;
//                *getMatVal(cnn->C3->y[j],r,c)=(float)0.0;
//            }
//        }
    }
    // S4
    for(j=0;j<cnn->S4->outChannels;j++){
        clearMat(cnn->S4->d[j]);
        clearMat(cnn->S4->y[j]);
//        for(r=0;r<cnn->S4->inputHeight/cnn->S4->mapSize;r++){
//            for(c=0;c<cnn->S4->inputWidth/cnn->S4->mapSize;c++){
//                *getMatVal(cnn->S4->d[j],r,c)=(float)0.0;
//                *getMatVal(cnn->S4->y[j],r,c)=(float)0.0;
//            }
//        }
    }
    // Output
    for(j=0;j<cnn->Out->outputNum;j++){
        cnn->Out->d[j]=(float)0.0;
        cnn->Out->v[j]=(float)0.0;
        cnn->Out->y[j]=(float)0.0;
    }
}

void trainModel(CNN* cnn, ImgArr inputData, LabelArr outputData, CNNOpts opts, int trainNum){           // may be slow?
    printf("start trainModel\n");
    cnn->L=(float *)malloc(opts.numepochs*sizeof(float));
    int epoch;
    for (epoch=0; epoch<opts.numepochs; epoch++){
        printf("Epoch %d \n", epoch);
        int n;
        for (n=0; n<trainNum; n++){
//            matrix* inputMat = defMat(inputData->ImgMatPtr[n].ImgData, inputData->ImgMatPtr[n].r, inputData->ImgMatPtr[n].c);
//            freeMat(inputMat);
            cnnfw(cnn, inputData->ImgMatPtr[n]);
//            cnn->L[epoch] = computeLoss(cnn->Out->p, outputData->LabelPtr[n].Labely);
            cnnbp(cnn, outputData->LabelPtr[n].LabelData);
            gradient_update(cnn, opts, inputData->ImgMatPtr[n]);

            cnnclear(cnn);
            float l=0.0;
            int i;
            for(i=0;i<cnn->Out->outputNum;i++)
                l=l+cnn->e[i]*cnn->e[i];
            if(n==0)
                cnn->L[n]=l/(float)2.0;
            else
                cnn->L[n]=cnn->L[n-1]*0.99+0.01*l/(float)2.0;
            printf("        n = %d,     loss = %f", n, cnn->L[n]);
        }
    }
    printf("end trainModel\n");
};


float testModel(CNN* cnn, ImgArr inputData, LabelArr outputData, int testNum){
    int sumOfCorrect = 0;
    int n;
    for (n=0; n<testNum; n++){
        cnnfw(cnn, inputData->ImgMatPtr[n]);
        int i, maxIndex=0;
        double max;
        for (i=0; i<cnn->Out->outputNum; i++){
            maxIndex = (cnn->Out->p[i]>0)?i:maxIndex;
        }
        if (maxIndex == outputData->LabelPtr[n].Labely) sumOfCorrect++;
    }
    return sumOfCorrect/(float) testNum;
};