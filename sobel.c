#include "sobel.h"
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
void fromGreyScale(byte* input, byte* output, int h, int w,int ch){
    int i,j;
    for(i = 0; i < h; i++){
        for(j = 0; j < w;j++){
            int ind = i*w*ch + j*ch;
            output[ind] = output[ind+1] = output[ind+2] = input[i*w+j];
        }
    }

}
void toGreyScale(byte *input, byte *output, int h, int w, int ch){
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