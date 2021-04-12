#include "image_prep.h"
#include "sobel.h"
#include <time.h>
#include <stdio.h>
int main(int argc, char** argv){
    if(argc < 2){
        printf("USAGE:./main [input.png] [output.png]");
        return -1;
    }
    read_png_file(argv[1]);

    //write_png_file(argv[2]);
    //TODO: image preprossing
    //TODO: apply sobel filter
    //TODO: output image

    return 0;
}