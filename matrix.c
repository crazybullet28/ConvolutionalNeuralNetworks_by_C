//
// Created by crazybullet on 2018/4/13.
//

#include <stdio.h>
#include <malloc.h>

typedef struct matrix{
    int row;
    int column;
    double * val;
//    double* (*getMat)(struct matrix *self, int r, int c);
} matrix;

double* getMat(matrix *self, int r, int c){
    if (r>=self->row || c>=self->column){
        fprintf( stderr, "Error in getMat. Size out of range: %d - %d, %d - %d\n", self->row, self->column, r, c);
    }else{
        return self->val + (c+self->column*r);
    }
}

matrix* initMat(int r, int c){
    matrix *res = (matrix*) malloc(sizeof(matrix));
    res->row = r;
    res->column = c;
    res->val = (double*) malloc(r*c*sizeof(double));
    return res;
}

void destruct(matrix *self){
    free(self->val);
    free(self);
}

matrix* defMat(double** data, int r, int c){
    matrix* res = initMat(r, c);
    int i, j;
    for (i=0; i<r; i++){
        for (j=0; j<c; j++){
            *getMat(res, i, j) = data[i][j];
        }
    }
    return res;
};

matrix* addMat(matrix *a, matrix *b){
    if (a->row != b->row || a->column != b->column){
        fprintf( stderr, "Error in addMat. Matrix size not fit: %d - %d, %d - %d\n", a->row, a->column, b->row, b->column);
        return NULL;
    }else{
        matrix *res = initMat(a->row, a->column);
        int i=0, j=0;
        for (i=0; i<a->row; i++){
            for (j=0; j<a->column; j++){
                *getMat(res, i, j) = *getMat(a, i, j) + *getMat(b, i, j);
            }
        }
        return res;
    }
};

matrix* addMatVal(matrix *a, double b){
    matrix *res = initMat(a->row, a->column);
    int i=0, j=0;
    for (i=0; i<a->row; i++){
        for (j=0; j<a->column; j++){
            *getMat(res, i, j) = *getMat(a, i, j) + b;
        }
    }
    return res;
};

matrix* dotMat(matrix *a, matrix *b){
    if (a->row != b->row || a->column != b->column){
        fprintf( stderr, "Error in dotMat. Matrix size not fit: %d - %d, %d - %d\n", a->row, a->column, b->row, b->column);
        return NULL;
    }else{
        matrix *res = initMat(a->row, a->column);
        int i=0, j=0;
        for (i=0; i<a->row; i++){
            for (j=0; j<a->column; j++){
                *getMat(res, i, j) = *getMat(a, i, j) * *getMat(b, i, j);
            }
        }
        return res;
    }
};

double dotMatSum(matrix *a, matrix *b){
    if (a->row != b->row || a->column != b->column){
        fprintf( stderr, "Error in dotMatSum. Matrix size not fit: %d - %d, %d - %d\n", a->row, a->column, b->row, b->column);
        return NULL;
    }else{
        double sum=0;
        int i=0, j=0;
        for (i=0; i<a->row; i++){
            for (j=0; j<a->column; j++){
                sum += *getMat(a, i, j) * *getMat(b, i, j);
            }
        }
        return sum;
    }
};

matrix* mulMat(matrix *a, matrix *b){
    if (a->column != b->row){
        fprintf( stderr, "Error in mulMat. Matrix size not fit: %d - %d, %d - %d\n", a->row, a->column, b->row, b->column);
        return NULL;
    }else{
        matrix *res = initMat(a->row, b->column);
        int i=0, j=0, k=0;
        for (i=0; i<a->row; i++){
            for (j=0; j<b->column; j++){
                *getMat(res, i, j) = 0;
                for (k=0; k<a->column; k++){
                    *getMat(res, i, j) += *getMat(a, i, k) * *getMat(b, k, j);
                }
            }
        }
        return res;
    }
};

matrix* mulMatVal(matrix *a, double b){
    matrix *res = initMat(a->row, a->column);
    int i=0, j=0;
    for (i=0; i<a->row; i++){
        for (j=0; j<a->column; j++){
            *getMat(res, i, j) = *getMat(a, i, j) * b;
        }
    }
    return res;
};

double sumMat(matrix *a){
    double res=0;
    int i=0, j=0;
    for (i=0; i<a->row; i++){
        for (j=0; j<a->column; j++){
            res += *getMat(a, i, j);
        }
    }
    return res;
};

matrix* tranMat(matrix* a){
    matrix *res = initMat(a->column, a->row);
    int i=0, j=0;
    for (i=0; i<a->column; i++){
        for (j=0; j<a->val; j++){
            *getMat(res, i, j) = *getMat(a, j, i);
        }
    }
    return res;
};

