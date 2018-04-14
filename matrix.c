//
// Created by crazybullet on 2018/4/13.
//

#include <stdio.h>
#include <malloc.h>

typedef struct matrix{
    int row;
    int column;
    double * val;
//    double* (*getMatVal)(struct matrix *self, int r, int c);
} matrix;

double* getMatVal(matrix *self, int r, int c){
    if (r>=self->row || c>=self->column){
        fprintf( stderr, "Error in getMatVal. Size out of range: %d - %d, %d - %d\n", self->row, self->column, r, c);
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

void deleteMat(matrix *self){
    free(self->val);
    free(self);
}
void clearMat(matrix *self){
    free(self->val);
}

void resetMat(matrix* a, int r, int c){
    if (a!=NULL){
        clearMat(a);
        a->row = r;
        a->column = c;
        a->val = (double*) malloc(r*c*sizeof(double));
    }else{
        a = initMat(r, c);
    }
}

matrix* defMat(double** data, int r, int c){
    matrix* res = initMat(r, c);
    int i, j;
    for (i=0; i<r; i++){
        for (j=0; j<c; j++){
            *getMatVal(res, i, j) = data[i][j];
        }
    }
    return res;
};

void addMat(matrix* res, matrix* a, matrix* b){
    if (a->row != b->row || a->column != b->column){
        fprintf( stderr, "Error in addMat. Matrix size not fit: %d - %d, %d - %d\n", a->row, a->column, b->row, b->column);
        return;
    }else{
        resetMat(res, a->row, a->column);
        int i=0, j=0;
        for (i=0; i<a->row; i++){
            for (j=0; j<a->column; j++){
                *getMatVal(res, i, j) = *getMatVal(a, i, j) + *getMatVal(b, i, j);
            }
        }
        return;
    }
};

void addMat_relpace(matrix* a, matrix* b){
    if (a->row != b->row || a->column != b->column){
        fprintf( stderr, "Error in addMat. Matrix size not fit: %d - %d, %d - %d\n", a->row, a->column, b->row, b->column);
        return;
    }else{
        int i=0, j=0;
        for (i=0; i<a->row; i++){
            for (j=0; j<a->column; j++){
                *getMatVal(a, i, j) = *getMatVal(a, i, j) + *getMatVal(b, i, j);
            }
        }
        return;
    }
};

void addMatVal(matrix* res, matrix *a, double b){
    resetMat(res, a->row, a->column);
    int i=0, j=0;
    for (i=0; i<a->row; i++){
        for (j=0; j<a->column; j++){
            *getMatVal(res, i, j) = *getMatVal(a, i, j) + b;
        }
    }
    return;
};

void addMatVal_replace(matrix *a, double b){
    int i=0, j=0;
    for (i=0; i<a->row; i++){
        for (j=0; j<a->column; j++){
            *getMatVal(a, i, j) = *getMatVal(a, i, j) + b;
        }
    }
    return;
};

void dotMat(matrix* res, matrix *a, matrix *b){
    if (a->row != b->row || a->column != b->column){
        fprintf( stderr, "Error in dotMat. Matrix size not fit: %d - %d, %d - %d\n", a->row, a->column, b->row, b->column);
        return;
    }else{
        resetMat(res, a->row, a->column);
        int i=0, j=0;
        for (i=0; i<a->row; i++){
            for (j=0; j<a->column; j++){
                *getMatVal(res, i, j) = *getMatVal(a, i, j) * *getMatVal(b, i, j);
            }
        }
        return;
    }
};

void dotMat_replace(matrix *a, matrix *b){
    if (a->row != b->row || a->column != b->column){
        fprintf( stderr, "Error in dotMat. Matrix size not fit: %d - %d, %d - %d\n", a->row, a->column, b->row, b->column);
        return;
    }else{
        int i=0, j=0;
        for (i=0; i<a->row; i++){
            for (j=0; j<a->column; j++){
                *getMatVal(a, i, j) = *getMatVal(a, i, j) * *getMatVal(b, i, j);
            }
        }
        return;
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
                sum += *getMatVal(a, i, j) * *getMatVal(b, i, j);
            }
        }
        return sum;
    }
};

void mulMat(matrix* res, matrix *a, matrix *b){
    if (a->column != b->row){
        fprintf( stderr, "Error in mulMat. Matrix size not fit: %d - %d, %d - %d\n", a->row, a->column, b->row, b->column);
        return;
    }else{
        resetMat(res, a->row, b->column);
        int i=0, j=0, k=0;
        for (i=0; i<a->row; i++){
            for (j=0; j<b->column; j++){
                *getMatVal(res, i, j) = 0;
                for (k=0; k<a->column; k++){
                    *getMatVal(res, i, j) += *getMatVal(a, i, k) * *getMatVal(b, k, j);
                }
            }
        }
        return;
    }
};

void mulMatVal(matrix* res, matrix *a, double b){
    resetMat(res, a->row, a->column);
    int i=0, j=0;
    for (i=0; i<a->row; i++){
        for (j=0; j<a->column; j++){
            *getMatVal(res, i, j) = *getMatVal(a, i, j) * b;
        }
    }
    return;
};

void mulMatVal_replace(matrix *a, double b){
    int i=0, j=0;
    for (i=0; i<a->row; i++){
        for (j=0; j<a->column; j++){
            *getMatVal(a, i, j) = *getMatVal(a, i, j) * b;
        }
    }
    return;
};

double sumMat(matrix *a){
    double res=0;
    int i=0, j=0;
    for (i=0; i<a->row; i++){
        for (j=0; j<a->column; j++){
            res += *getMatVal(a, i, j);
        }
    }
    return res;
};

void tranMat(matrix* res, matrix* a){
    resetMat(res, a->column, a->row);
    int i=0, j=0;
    for (i=0; i<a->column; i++){
        for (j=0; j<a->row; j++){
            *getMatVal(res, i, j) = *getMatVal(a, j, i);
        }
    }
    return;
};

void subMat(matrix* res, matrix* a, int r_start, int height, int c_start, int width){
    if (a->row < r_start+height || a->column < c_start+width){
        fprintf( stderr, "Error in subMat. SubMatrix size out of range: %d - %d, %d %d %d %d\n", a->row, a->column, r_start, height, c_start, width);
        return;
    }else {
        resetMat(res, height, width);
        int i, j;
        for (i = 0; i < height; i++) {
            for (j = 0; j < width; j++) {
                *getMatVal(res, i, j) = *getMatVal(a, r_start + i, c_start + j);
            }
        }
        return;
    }
};

