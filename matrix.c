//
// Created by crazybullet on 2018/4/13.
//

#include <stdio.h>
#include <malloc.h>

typedef struct matrix{
    int row;
    int column;
    float * val;
//    float* (*getMat)(struct matrix *self, int r, int c);
} matrix;

float* getMat(matrix *self, int r, int c){
    if (r>=self->row || c>=self->column){
        fprintf( stderr, "Error in getMat. Size out of range: %d - %d, %d - %d\n", self->row, self->column, r, c);
    }else{
        return self->val + (c+self->column*r);
    }
}

matrix *initial(int r, int c){
    matrix *res = malloc(sizeof(matrix));
    res->row = r;
    res->column = c;
    res->val = malloc(r*c*sizeof(float));
//    res->getMat = getMat;
    return res;
}

void destruct(matrix *self){
    free(self->val);
    free(self);
}

struct matrix *addMat(matrix *a, matrix *b){
    if (a->row != b->row || a->column != b->column){
        fprintf( stderr, "Error in addMat. Matrix size not fit: %d - %d, %d - %d\n", a->row, a->column, b->row, b->column);
        return NULL;
    }else{
        matrix *res = initial(a->row, a->column);
        int i=0, j=0;
        for (i=0; i<a->row; i++){
            for (j=0; j<a->column; j++){
                *getMat(res, i, j) = *getMat(a, i, j) + *getMat(b, i, j);
            }
        }
        return res;
    }
};

struct matrix *addMatVal(matrix *a, float b){
    matrix *res = initial(a->row, a->column);
    int i=0, j=0;
    for (i=0; i<a->row; i++){
        for (j=0; j<a->column; j++){
            *getMat(res, i, j) = *getMat(a, i, j) + b;
        }
    }
    return res;
};

struct matrix *dotMat(matrix *a, matrix *b){
    if (a->column != b->row){
        fprintf( stderr, "Error in dotMat. Matrix size not fit: %d - %d, %d - %d\n", a->row, a->column, b->row, b->column);
        return NULL;
    }else{
        matrix *res = initial(a->row, b->column);
        int i=0, j=0, k=0;
        for (i=0; i<a->row; i++){
            for (j=0; j<b->column; j++){
                *getMat(res, i, j) = 0;
                for (k=0; k<a->)
            }
        }
        return res;
    }
}