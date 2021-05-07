#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <stdbool.h>
#include "blur.h"

#include "image_prep.h"
#include "sobel.h"

#define ThresholdL 51
#define ThresholdH 102

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

bool checkInRang(int r,int c, int rows, int cols){
	if (r >= 0 && r < rows && c >= 0 && c < cols)
		return true;
	else{
                return false;
        }
}
// from edge, extend the edge
void trace(byte* edgeMag_noMaxsup, byte* edge, int TL, int y, int x, int height, int width){
        int index = y * width + x;
        if (edge[index] == 0){
		edge[index] = 255;
                int i, j;
		for (int i = -1; i <= 1; ++i){
			for (int j = -1; j <= 1; ++j){
                                int sub_index = (y+i)*width + (x+j);
				float mag = edgeMag_noMaxsup[sub_index];
				if (checkInRang(y + i, x + j, height, width) && mag >= TL)
					trace(edgeMag_noMaxsup, edge, TL, y + i, x + j, height, width);
			}
		}
	}
}
// 255 white
// 0 black
void hysteresis(byte *gradiant,byte* edge,int height, int width){
        int x, y;
        int changes;
        for(y = 0; y < height; y++){
                for(x = 0; x < width; x++){
                        int index = y * width + x;
                        if(gradiant[index] >= ThresholdH){
                                trace(gradiant,edge,ThresholdL,y,x,height,width);
                        }else if(gradiant[index] < ThresholdL){
                                edge[index] = 0;
                        }
                }
        }
}
int main(int argc, char** argv){
        if(argc < 2){
                printf("ERR: %d MSG: %s",-1,"Usage: ./canny [input.png] [output.png]");
                exit(-1);
        }
        int i,j;
        byte *Img, *ImgMono, *ImgMonoOut;
        unsigned channels, rowStride, blockSize;
        printf("Reading the image...\n");
        read_png_file(argv[1]);
        channels = color_type==PNG_COLOR_TYPE_RGBA ? 4 : 3;
        rowStride = width*channels;

        printf("channel is :%d\n", channels);
        // printf("Allocating space...\n");
        Img = (byte*)malloc(width * height * channels);
        ImgMono = (byte*)malloc(width * height);
        ImgMonoOut = (byte*)malloc(width * height);
        byte* ImgTemp = (byte*)malloc(width * height);

        for (i = 0; i < height; ++i) {
                memcpy(Img + i*rowStride, row_pointers[i], rowStride);
        }
        clock_t begin = clock();
        // convert to grayscale
        //printf("Converting to grayscale...\n");
        toGreyScale(Img,ImgMono,height,width,channels);

        //printf("Performing Gaussian blurring...........\n");
        clock_t blur_begin = clock();
        blur(2,ImgMono,ImgMonoOut);
        clock_t blur_end = clock();
        printf("blur time: %f\n",(double)(blur_end-blur_begin)/CLOCKS_PER_SEC);

        
        printf("Performing Sobel filter...\n");
        clock_t sobel_begin = clock();
        sobelv2(ImgMonoOut,ImgMono,ImgTemp,height,width);
                //ImgMono: net Gradient 
                //ImgTemp: direction
        clock_t sobel_end = clock();
        printf("sobel time: %f\n",(double)(sobel_end-sobel_begin)/CLOCKS_PER_SEC);

        
        //printf("Performing Edge thinning...\n");
        clock_t edge_thin_begin = clock();
        edge_thin(ImgMono,ImgTemp,ImgMonoOut,height,width);
        clock_t edge_thin_end = clock();
        printf("edge_thin time: %f\n",(double)(edge_thin_end-edge_thin_begin)/CLOCKS_PER_SEC);


        //printf("Performing Double thresholding...\n");
        clock_t edge_thin_double_begin = clock();
        edge_thin_double(ImgMonoOut,ImgTemp,height,width);
        clock_t edge_thin_double_end = clock();
        printf("edge_thin_double time: %f\n",(double)(edge_thin_double_end-edge_thin_double_begin)/CLOCKS_PER_SEC);


        printf("Performing Hysteresis Thresholding...\n");
        // using ImgTemp

        clock_t hyster_begin = clock();
        byte* edge = (byte*)calloc(height*width,sizeof(byte));

        hysteresis(ImgTemp,edge,height,width);
        clock_t hyster_end = clock();
        printf("hyster time: %f\n",(double)(hyster_end-hyster_begin)/CLOCKS_PER_SEC);


        // convert back from grayscale
        printf("Convert image back to multi-channel...\n");
        fromGreyScale(edge,Img,height,width,channels);

        clock_t end = clock();


        // copy image back to row_pointers
        printf("Copy image back to row_pointers...\n");
        for (i = 0; i < height; i++) {
                memcpy(row_pointers[i],Img + i*rowStride, rowStride);
        }
        printf("Writing image back to file...\n");
        write_png_file(argv[2]);
        printf("Done...\n");
        printf("time cost: %f\n",(double)(end-begin)/CLOCKS_PER_SEC);
        return 0;
}