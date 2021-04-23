#ifndef IMAGE_PREP
#define IMAGE_PREP
#include <png.h>

//open the image and conver in to char array
// var: png_bytep* row_pointers
extern int width, height;
extern png_bytep * row_pointers;
extern png_byte color_type;

void read_png_file(char* file_name);
void write_png_file(char* file_name);

#endif
