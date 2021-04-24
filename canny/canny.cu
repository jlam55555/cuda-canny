#include "image_prep.h"
#include "canny.h"
#include <iostream>
#include <unistd.h>
#include <string>

// TODO: optimize multiplications!
// TODO: worry about memory locality later
// TODO: signed or unsigned? chars or shorts?
// TODO: try separable filters

cudaError_t err = cudaSuccess;
dim3 dimGrid, dimBlock;

#define CUDAERR(fn, msg)\
	if ((err = fn) != cudaSuccess) {\
		std::cerr << "cuda error: " msg " ("\
			<< cudaGetErrorString(err) << ")" << std::endl;\
		_exit(-1);\
	}

// performs a gaussian blur on an image
__host__ void blur(float blurSize, byte *dImg, byte *dImgOut)
{
        float *hFlt, *dFlt;
        unsigned fltSize;

        gaussian_filter(blurSize, &hFlt, &fltSize);

        // allocate and copy filter to device
	CUDAERR(cudaMalloc((void **) &dFlt, fltSize*fltSize*sizeof(float)),
		"allocating dFlt");

	CUDAERR(cudaMemcpy(dFlt, hFlt, fltSize*fltSize*sizeof(float),
		cudaMemcpyHostToDevice), "copying hFlt to dFlt");

        // blur image (for testing)
        conv2d<<<dimGrid, dimBlock>>>(dImg, dFlt, dImgOut,
                height, width, fltSize, fltSize);

        // cleanup
        free(hFlt);
        CUDAERR(cudaFree(dFlt), "freeing dFlt");
}

// basic sobel kernel
// out is the magnitude of the gradient
// out2 is the angle of the gradient
__global__ void sobel(byte *img, byte *out, byte *out2, int h, int w)
{
	int vKer, hKer, y, x;

	y = blockDim.y*blockIdx.y + threadIdx.y;
	x = blockDim.x*blockIdx.x + threadIdx.x;

	if (y <= 0 || y >= h-1 || x <= 0 || x >= w-1) {
		return;
	}

	vKer = img[(y-1)*w+(x-1)]*1 + img[(y-1)*w+x]*2 + img[(y-1)*w+(x+1)]*1 +
		img[(y+1)*w+(x-1)]*-1 + img[(y+1)*w+x]*-2 + img[(y+1)*w+(x+1)]*-1;

	hKer = img[(y-1)*w+(x-1)]*1 + img[(y-1)*w+(x+1)]*-1 +
		img[y*w+(x-1)]*2 + img[y*w+(x+1)]*-2 +
		img[(y+1)*w+(x-1)]*1 + img[(y+1)*w+(x+1)]*-1;

	out[y*w+x] = min(sqrtf(hKer*hKer + vKer*vKer), 255.);
	out2[y*w+x] = ((byte)roundf((atan2f(vKer, hKer)+M_PI) / (M_PI/4))) % 4;
}

// perform edge thinning
__global__ void edge_thin(byte *mag, byte *angle, byte *out, int h, int w)
{
	int y, x, y1, x1, y2, x2;

	y = blockDim.y*blockIdx.y + threadIdx.y;
	x = blockDim.x*blockIdx.x + threadIdx.x;

	if (y <= 0 || y >= h-1 || x <= 0 || x >= w-1) {
		return;
	}

	// if not greater than angles in both directions, then zero
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

// perform double thresholding
__global__ void edge_thin(byte *dImg, byte *out, int h, int w, byte t1, byte t2)
{
	int y, x, ind, grad;

	y = blockDim.y*blockIdx.y + threadIdx.y;
	x = blockDim.x*blockIdx.x + threadIdx.x;

	if (y <= 0 || y >= h-1 || x <= 0 || x >= w-1) {
		return;
	}

	ind = y*w + x;
	grad = dImg[ind];
	if (grad < t1) {
		out[ind] = 0;
	} else if (grad < t2) {
		out[ind] = t2;
	} else {
		out[ind] = 255;
	}
}

// perform canny edge detection
__host__ void canny(byte *dImg, byte *dImgOut)
{
	byte *dTmp, *dImgTmp;

	CUDAERR(cudaMalloc((void**)&dImgTmp, width*height), "alloc dImgTmp");

	std::cout << "Performing Gaussian blurring..." << std::endl;
	blur(1.4, dImg, dImgOut);

	std::cout << "Performing Sobel filter..." << std::endl;
	sobel<<<dimGrid, dimBlock>>>(dImgOut, dImg, dImgTmp, height, width);
	CUDAERR(cudaGetLastError(), "launch sobel kernel");

	std::cout << "Performing edge thinning..." << std::endl;
	edge_thin<<<dimGrid, dimBlock>>>(dImg, dImgTmp, dImgOut, height, width);
	CUDAERR(cudaGetLastError(), "launch edge thinning kernel");

	std::cout << "Performing double thresholding..." << std::endl;
	edge_thin<<<dimGrid, dimBlock>>>(dImgOut, dImgTmp, height, width,
		255*0.2, 255*0.5);
	CUDAERR(cudaGetLastError(), "launch double thresholding kernel");

	// TODO: remove this
	CUDAERR(cudaMemcpy(dImgOut, dImgTmp, width*height, cudaMemcpyDeviceToDevice),
		"TESTING");

	// dTmp = dImg;
	// dImg = dImgOut;
	// dImgOut = dTmp;

	CUDAERR(cudaFree(dImgTmp), "freeing dImgTmp");
}

__host__ int main(int argc, char **argv)
{
	std::string filename;
	unsigned i, channels, rowStride, blockSize;
	byte *hImg, *dImg, *dImgMono, *dImgMonoOut;

	// get image name
	std::cout << "Enter filename of image (*.png): ";
	std::cin >> filename;

	// get image
	std::cout << "Reading image from file..." << std::endl;
	read_png_file(const_cast<char *>(filename.c_str()));
	channels = color_type==PNG_COLOR_TYPE_RGBA ? 4 : 3;
	rowStride = width*channels;

	std::cout << "Channels: " << channels << std::endl;

	// allocate memory
	std::cout << "Allocating host and device buffers..." << std::endl;
	hImg = (byte *)malloc(width*height*channels);
	CUDAERR(cudaMalloc((void **)&dImg, width*height*channels),
		"cudaMalloc dImg");
	CUDAERR(cudaMalloc((void **)&dImgMono, width*height),
		"cudaMalloc dImgMono");
	CUDAERR(cudaMalloc((void **)&dImgMonoOut, width*height),
		"cudaMalloc dImgMonoOut");

	// copy image from row-pointers to device
	for (i = 0; i < height; ++i) {
		memcpy(hImg + i*rowStride, row_pointers[i], rowStride);
	}

	// copy image to device
	std::cout << "Copying image to device..." << std::endl;
	CUDAERR(cudaMemcpy(dImg, hImg, width*height*channels,
		cudaMemcpyHostToDevice), "cudaMemcpy to device");

	// set kernel parameters (same for all future kernel invocations)
	// TODO: calculate best grid/block dim depending on the device
	blockSize = 32;
	dimGrid = dim3(ceil(rowStride*1./blockSize),
		ceil(height*1./blockSize), 1);
	dimBlock = dim3(blockSize, blockSize, 1);

	// convert to grayscale
	std::cout << "Converting to grayscale..." << std::endl;
	toGrayScale<<<dimGrid, dimBlock>>>(dImg, dImgMono, height, width,
		channels);
	CUDAERR(cudaGetLastError(), "launch toGrayScale kernel");

	// canny edge detection
	std::cout << "Performing canny edge-detection..." << std::endl;
	canny(dImgMono, dImgMonoOut);

	// convert back from grayscale
	std::cout << "Convert image back to multi-channel..." << std::endl;
	fromGrayScale<<<dimGrid, dimBlock>>>(dImgMonoOut, dImg,
		height, width, channels);
	CUDAERR(cudaGetLastError(), "launch fromGrayScale kernel");

	// copy image back to host
	std::cout << "Copy image back to host..." << std::endl;
	CUDAERR(cudaMemcpy(hImg, dImg, width*height*channels,
		cudaMemcpyDeviceToHost), "cudaMemcpy to host");

	// copy image back to row_pointers
	std::cout << "Copy image back to row_pointers..." << std::endl;
	for (i = 0; i < height; ++i) {
		memcpy(row_pointers[i], hImg + i*rowStride, rowStride);
	}

	// copy image back from device
	std::cout << "Writing image back to file..." << std::endl;
	write_png_file("test.png");

	// freeing pointers
	std::cout << "Freeing device memory..." << std::endl;
	CUDAERR(cudaFree(dImg), "freeing dImg");
	CUDAERR(cudaFree(dImgMono), "freeing dImgMono");
	CUDAERR(cudaFree(dImgMonoOut), "freeing dImgMonoOut");

	std::cout << "Done." << std::endl;
}
