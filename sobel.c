#include "./sobel/sobel.h"
#include <math.h>
#include <stdlib.h>
#include <float.h>
typedef unsigned char byte;
//only the convolution filter
void sobel(byte *input, byte *output, int h, int w){
    int i,j,vKer,hKer,x,y;
    byte *img = input;
    for(i = 0; i < h; i++){
        for(j = 0; j < w; j++){
            y = i;
            x = j;
            vKer = img[(y-1)*w+(x-1)]*1 + img[(y-1)*w+x]*2 + img[(y-1)*w+(x+1)]*1 +
		        img[(y+1)*w+(x-1)]*-1 + img[(y+1)*w+x]*-2 + img[(y+1)*w+(x+1)]*-1;
	        hKer = img[(y-1)*w+(x-1)]*1 + img[(y-1)*w+(x+1)]*-1 + img[y*w+(x-1)]*2 + img[y*w+(x+1)]*-2 +
		        img[(y+1)*w+(x-1)]*1 + img[(y+1)*w+(x+1)]*-1;
            output[y*w+x] = hKer*hKer + vKer*vKer > 40000 ? 255 : 0;
        }
    }
}
void sobelv2(byte *input, byte *output,byte* output2, int h, int w){
    int x,y;
    byte *img = input;
    for(y = 0; y < h; y++){
        for(x = 0; x < w; x++){
            int vKer,hKer;
            vKer = img[(y-1)*w+(x-1)]*1 + img[(y-1)*w+x]*2 + img[(y-1)*w+(x+1)]*1 +
		        img[(y+1)*w+(x-1)]*-1 + img[(y+1)*w+x]*-2 + img[(y+1)*w+(x+1)]*-1;
	        hKer = img[(y-1)*w+(x-1)]*1 + img[(y-1)*w+(x+1)]*-1 + img[y*w+(x-1)]*2 + img[y*w+(x+1)]*-2 +
		        img[(y+1)*w+(x-1)]*1 + img[(y+1)*w+(x+1)]*-1;

            // Gradient strength
            output[y*w+x] = fmin(sqrtf(hKer*hKer + vKer*vKer), 255.);
            // Direction
	        output2[y*w+x] = ((byte)roundf((atan2f(vKer, hKer)+M_PI) / (M_PI/4))) % 4;
        }
    }
}

void fromGreyScale(byte* input, byte* output, int h, int w,int ch)
{
    int i,j;
    for(i = 0; i < h; i++){
        for(j = 0; j < w;j++){
            int ind = i*w*ch + j*ch;
            //printf("ind: %d",ind);
            output[ind] = output[ind+1] = output[ind+2] = input[i*w+j];
        }
    }

}
void toGreyScale(byte *input, byte *output, int h, int w, int ch)
{
    int i,j;
    for(i = 0; i < h; i++){
        for(j = 0; j < w; j++){
            byte res = 0;
            int ind = i*w*ch + j*ch;
            res = input[ind+0]*0.2989 + input[ind+1]*0.5870 + input[ind+2]*0.1140;
            output[i*w+j] = res;
        }
    }
}