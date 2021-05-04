#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include "blur.h"
#include "../image_prep.h"
#include "../sobel.h"

#define threshold_1 64
#define threshold_2 0x90

typedef unsigned char byte;

void blur(float blurSize, byte *dImg, byte *dImgOut)
{
        printf("blurSize: %lf \n",blurSize);
        //printf("1111\n");
        float *hFlt, *dFlt;
        unsigned fltSize;
        //printf("filter\n");
        gaussian_filter(blurSize, &hFlt, &fltSize);

        dFlt = malloc(fltSize*fltSize*sizeof(float));
        memcpy(dFlt,hFlt,fltSize*fltSize*sizeof(float));
        printf("Performing conv2d...");
        conv2d(dImg, dFlt, dImgOut, height, width, fltSize, fltSize);
        free(hFlt);

}
void edge_thin(byte *mag, byte *angle, byte *out, int h, int w)
{
        int y,x;
        for(y = 0; y < h; y++){
                for(x = 0; x < w; x++){
                        int y1, x1, y2, x2;
                        switch (angle[y*w + x]) {
                        case 0:
                                // horizontal
                                y1 = y2 = y;
                                x1 = x-1;
                                x2 = x+1;
                                break;
                        case 3:
                                // 135
                                y1 = y-1;
                                x1 = x+1;
                                y2 = y+1;
                                x2 = x-1;
                                break;
                        case 2:
                                // vertical
                                x1 = x2 = x;
                                y1 = y-1;
                                y2 = y+1;
                                break;
                        case 1:
                                // 45
                                y1 = y-1;
                                x1 = x-1;
                                y2 = y+1;
                                x2 = x+1;
                        }

                        if (mag[y1*w + x1] >= mag[y*w + x] || mag[y2*w + x2] >= mag[y*w + x]) {
                                out[y*w + x] = 0;
                        } else {
                                out[y*w + x] = mag[y*w + x];
                        }
                }
        }
}
void edge_thin_double(byte* input, byte* output, int h, int w)
{
        int y, x, ind, grad;
        byte t1 = 255*0.2;
        byte t2 = 255*0.5;
        for (y = 0; y < h; y ++){
                for(x = 0; x < w; x++){
                        ind = y*w + x;
                        grad = input[ind];
                        if (grad < t1) {
                                output[ind] = 0;
                        } else if (grad < t2) {
                                output[ind] = t2;
                        } else {
                                output[ind] = 255;
                        }
                }
        }
}	

void bfs(byte* input, int x, int y, int width, int height, int* changes){
        
        if(x >= width || y >= height || x < 0 || y < 0){
                return;
        }
        int index = y*width + x;
        if(input[index] == threshold_1){
                input[index] = threshold_2;
                *changes = 1;
        }

}

void hysteresis(byte *gradiant,int height, int width){
        int x, y;
        int changes;
        for(y = 0; y < height; y++){
                for(x = 0; x < width; x++){
                        changes = 0;
                        int index = y * width + x;
                        do{
                                if(gradiant[index] == threshold_2){
                                        gradiant[index] = 255;
                                        bfs(gradiant, x+1,y+1,height,width,&changes);
                                        bfs(gradiant, x+1,y,height,width,&changes);
                                        bfs(gradiant, x+1,y-1,height,width,&changes);
                                        bfs(gradiant, x,y+1,height,width,&changes);
                                        bfs(gradiant, x,y-1,height,width,&changes);
                                        bfs(gradiant, x-1,y+1,height,width,&changes);
                                        bfs(gradiant, x-1,y,height,width,&changes);
                                        bfs(gradiant, x-1,y-1,height,width,&changes);
                                }
                                printf("changes: %d \n",changes);
                        }while (changes);
                        
                }
        }
        for(y = 0; y < height; y++){
                for(x = 0; x < width; x++){
                        int index = y * width + x;
                        if(gradiant[index] != 255){
                                gradiant[index] = 0;
                        }
                }
        }
}
int main(int argc, char** argv){
    if(argc < 2){
        printf("ERR: %d MSG: %s",-1,"Usage: ./canny [input.png] [output.png]");
        exit(-1);
        //ERRMSG(-1,"USAGE:./main [input.png] [output.png]")
    }
    int i,j;
    byte *Img, *ImgMono, *ImgMonoOut;
    unsigned channels, rowStride, blockSize;
    printf("Reading the image...\n");
    read_png_file("/Users/leonfang/Sobel_filter/canny/star.png");
    channels = color_type==PNG_COLOR_TYPE_RGBA ? 4 : 3;
    rowStride = width*channels;

    printf("channel is :%d\n", channels);
    printf("Allocating space...\n");
    Img = (byte*)malloc(width * height * channels);
    ImgMono = (byte*)malloc(width * height);
    ImgMonoOut = (byte*)malloc(width * height);
    byte* ImgTemp = (byte*)malloc(width * height);
    
    for (i = 0; i < height; ++i) {
        memcpy(Img + i*rowStride, row_pointers[i], rowStride);
    }
    clock_t begin = clock();
    // convert to grayscale
    printf("Converting to grayscale...\n");
    toGreyScale(Img,ImgMono,height,width,channels);

    printf("Performing Gaussian blurring...........\n");
    blur(1.4,ImgMono,ImgMonoOut);
    
    printf("Performing Sobel filter...\n");
    sobelv2(ImgMonoOut,ImgMono,ImgTemp,height,width);
//ImgMono: net Gradient 
//ImgTemp: direction

    printf("Performing Edge thinning...\n");
    edge_thin(ImgMono,ImgTemp,ImgMonoOut,height,width);
    
    printf("Performing Double thresholding...\n");
    edge_thin_double(ImgMonoOut,ImgTemp,height,width);
    
    printf("Performing Hysteresis Thresholding...\n");
    // using ImgTemp
    hysteresis(ImgTemp,height,width);

    // convert back from grayscale
    printf("Convert image back to multi-channel...\n");
    fromGreyScale(ImgTemp,Img,height,width,channels);
    
    clock_t end = clock();

/*
        int x,y;
        for(y = 0; y < height; y++){
                for(x = 0; x < width; x++){
                        int ind = y*width + x;
                        printf("%d ",ImgMonoOut[ind]);
                }
                printf("\n");
        }
        */
       


    // copy image back to row_pointers
    printf("Copy image back to row_pointers...\n");
    for (i = 0; i < height; i++) {
	memcpy(row_pointers[i],Img + i*rowStride, rowStride);
    }
    printf("Writing image back to file...\n");
    write_png_file("/Users/leonfang/Sobel_filter/canny/test.png");
    printf("Done...\n");
    printf("time cost: %f\n",(double)(end-begin)/CLOCKS_PER_SEC);
    return 0;
}