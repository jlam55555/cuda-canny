#ifndef SOBEL
#define SOBEL
typedef unsigned char byte;
void sobel(byte *input, byte *output, int h, int w);
void fromGreyScale(byte* input, byte* output, int h, int w,int ch);
void toGreyScale(byte *input, byte *output, int h, int w, int ch);
#endif
