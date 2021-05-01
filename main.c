#include "image_prep.h"
#include "sobel.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define ERRMSG(code,msg)\
    if(code != 0){\
        printf("ERR: %d MSG: %s",code,msg);\
        _exit(code);\
    }\

typedef unsigned char byte;
int main(int argc, char** argv){
    if(argc < 2){
        printf("ERR: %d MSG: %s",-1,"Usage: ./sobel [input.png] [output.png]");
        exit(-1);
        //ERRMSG(-1,"USAGE:./main [input.png] [output.png]")
    }
    unsigned i, j, channels, rowStride, blockSize;
    byte *Img, *ImgMono, *ImgOut;

	// get image
	printf("Reading image from file...\n");
    read_png_file(argv[1]);
	channels = color_type==PNG_COLOR_TYPE_RGBA ? 4 : 3;
	rowStride = width*channels;

    printf("Channels: %d\n",channels);
    printf("width is %d\n",height);
    printf("height is %d\n",height);
    printf("rowstride is %d\n",rowStride);

    // assign the memory
    Img = (byte *)malloc(width*height*channels);
    ImgMono = (byte *)malloc(width*height);
    ImgOut = (byte *)malloc(width*height);

    // copy from row_pointers to Img
    for (i = 0; i < height; i++) {
		memcpy(Img + i*rowStride, row_pointers[i], rowStride);
	}
    clock_t begin = clock();
	// convert to grayscale
    printf("Converting to grayscale...\n");
    toGreyScale(Img,ImgMono,height,width,channels);
	// sobel filter
    printf("Performing sobel filter edge-detection...\n");
    sobel(ImgMono,ImgOut,height,width);

	// convert back from grayscale
    printf("Convert image back to multi-channel...\n");
    fromGreyScale(ImgOut,Img,height,width,channels);
    clock_t end = clock();
    
	// copy image back to row_pointers
	printf("Copy image back to row_pointers...\n");
    for (i = 0; i < height; i++) {
		memcpy(row_pointers[i],Img + i*rowStride, rowStride);
	}
	printf("Writing image back to file...\n");
	write_png_file(argv[2]);
	printf("Done.");
    printf("time cost: %f",(double)(end-begin)/CLOCKS_PER_SEC);

    return 0;
}
