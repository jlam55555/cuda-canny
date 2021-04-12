#ifndef IMAGE_PREP
#define IMAGE_PREP
#include <png.h>
//open the image and conver in to char array
// var: png_bytep* row_pointers
int width, height;
png_bytep * row_pointers;

void read_png_file(char* file_name);
void write_png_file(char* file_name);

#endif
