#ifndef BLUR_H
#define BLUR_H
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })
#define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _b : _a; })

typedef unsigned char byte;
void gaussian_filter(float blurStd, float **fltp, unsigned *fltSizep);
void conv2d(byte *input, float *filt, byte *output, int h1, int w1, int h2, int w2);
#endif