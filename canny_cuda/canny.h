#ifndef CANNY_H
#define CANNY_H

#include <cuda.h>

typedef unsigned char byte;

// conv2d.cu
__global__ void conv2d(byte *d1, float *d2, byte *d3,
        int h1, int w1, int h2, int w2);

// blur.cu
__host__ void gaussian_filter(float blurStd, float **fltp, unsigned *fltSizep);

// gray.cu
__global__ void toGrayScale(byte *dImg, byte *dImgMono, int h, int w, int ch);
__global__ void fromGrayScale(byte *dImgMono, byte *dImg, int h, int w, int ch);

// clock types
enum { CLK_ALL, CLK_GRAY, CLK_BLUR, CLK_SOBEL, CLK_THIN, CLK_THRES, CLK_HYST };

#endif