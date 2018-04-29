//
// Created by crazybullet on 2018/4/20.
//

#ifndef PARALLEL_PROJ_MATRIX_H
#define PARALLEL_PROJ_MATRIX_H

#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include "mpi.h"

typedef struct matrix{
    int row;
    int column;
    float * val;
//    float* (*getMatVal)(struct matrix *self, int r, int c);
} matrix;

float* getMatVal(matrix *self, int r, int c);

matrix* initMat(int r, int c, int type);

void freeMat(matrix *self);

void freeMatVal(matrix *self);

void clearMat(matrix *self);

//void resetMat(matrix* self, int r, int c);

matrix* defMat(float** data, int r, int c);

matrix* copyMat(matrix* in);

void copyMat2exist(matrix* res, matrix* in);

void resizeMat(matrix* self, int r, int c);

void addMat(matrix* res, matrix* a, matrix* b);

void addMat_replace(matrix *a, matrix *b);

void addMatVal(matrix* res, matrix *a, float b);

void addMatVal_replace(matrix *a, float b);

void dotMat(matrix* res, matrix *a, matrix *b);

void dotMat_replace(matrix *a, matrix *b);

float dotMatSum(matrix *a, matrix *b);

void mulMat(matrix* res, matrix *a, matrix *b);

void mulMatVal(matrix* res, matrix *a, float b);

void mulMatVal_replace(matrix *a, float b);

float sumMat(matrix *a);

float maxMat(matrix* a);

void tranMat(matrix* res, matrix* a);

void subMat(matrix* res, matrix* a, int r_start, int height, int c_start, int width);

void rotate180Mat(matrix* res, matrix* a);

void rotate180Mat_replace(matrix* a);

void mat2arr(float* res, matrix* a);

void sendMat(matrix* sendObj, int toProc, int tag);

void recvMat(matrix* recvObj, int fromProc, int tag, MPI_Status status);

void sendrecvMat(matrix* sendObj, int toProc, int sendtag, matrix* recvObj, int fromProc, int recvtag, MPI_Status status);

void printMat(matrix* mat);

#endif //PARALLEL_PROJ_MATRIX_H
