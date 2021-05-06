#ifndef CANNY_H
#define CANNY_H

#include <cuda.h>
#include <unistd.h>
#include <iostream>

typedef unsigned char byte;

#define CUDAERR(fn, msg)\
	if ((err = fn) != cudaSuccess) {\
		std::cerr << "cuda error: " msg " ("\
			<< cudaGetErrorString(err) << ")" << std::endl;\
		_exit(-1);\
	}

extern cudaError_t err;

#define tx	threadIdx.x
#define ty	threadIdx.y
#define bs	32		// normal block size

#define lbs	64		// long block size (1D conv.)
#define sbs	16		// short block size (1D conv.)

// conv2d.cu
__host__ void setFilter(float *flt, unsigned size);
__global__ void conv2d(byte *dIn, byte *dOut,
        int h, int w, int hFlt, int wFlt);
__global__ void conv1dRows(byte *dIn, byte *dOut,
	int h, int w, int fltSize);
__global__ void conv1dCols(byte *dIn, byte *dOut,
	int h, int w, int fltSize);

// blur.cu
__host__ void gaussian_filter(float blurStd, float **fltp, unsigned *fltSizep);
__host__ void gaussian_filter_1d(float blurStd, float **fltp,
	unsigned *fltSizep);

// gray.cu
__global__ void toGrayScale(byte *dImg, byte *dImgMono, int h, int w, int ch);
__global__ void fromGrayScale(byte *dImgMono, byte *dImg, int h, int w, int ch);

// clock types
enum { CLK_ALL, CLK_GRAY, CLK_BLUR, CLK_SOBEL, CLK_THIN, CLK_THRES, CLK_HYST };

#endif