#ifndef SOBEL
#define SOBEL
#include <stdio.h>
typedef unsigned char byte;
void sobel(byte *input, byte *output, int h, int w);
void sobelv2(byte *input, byte *output,byte* output2, int h, int w);
void fromGreyScale(byte* input, byte* output, int h, int w,int ch);
void toGreyScale(byte *input, byte *output, int h, int w, int ch);
#endif
