#ifndef PARALLEL_PROJ_MINST_H
#define PARALLEL_PROJ_MINST_H
/*
MINST数据库是一个手写图像数据库，里面
*/

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
//#include <random>
#include <time.h>
#include "matrix.h"

//typedef struct MinstImg{
//    matrix* ImgMat;
////    int c;           // 图像宽
////    int r;           // 图像高
////    float** ImgData; // 图像数据二维动态数组
//}MinstImg;

typedef struct MinstImgArr{
    int ImgNum;        // 存储图像的数目
//    MinstImg* ImgMatPtr;  // 存储图像数组指针
    matrix** ImgMatPtr;
}*ImgArr;              // 存储图像数据的数组

typedef struct MinstLabel{
    int l;            // 输出标记的长
    float* LabelData; // 输出标记数据
    int Labely;
}MinstLabel;

typedef struct MinstLabelArr{
    int LabelNum;
    MinstLabel* LabelPtr;
}*LabelArr;              // 存储图像标记的数组

LabelArr read_Lable(const char* filename); // 读入图像标记

ImgArr read_Img(const char* filename); // 读入图像

void save_Img(ImgArr imgarr,char* filedir); // 将图像数据保存成文件

#endif //PARALLEL_PROJ_MINST_H
