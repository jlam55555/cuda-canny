#include "image_prep.h"
#include <iostream>
#include <unistd.h>

// TODO: optimize multiplications!
// TODO: worry about memory locality later
// TODO: signed or unsigned? chars or shorts?
// TODO: try separable filters

typedef unsigned char byte;
cudaError_t err = cudaSuccess;

#define CUDAERR(fn, msg)\
	if ((err = fn) != cudaSuccess) {\
		std::cerr << "cuda error: " msg " ("\
			<< cudaGetErrorString(err) << ")" << std::endl;\
		_exit(-1);\
	}

// basic sobel kernel
__global__ void sobel(byte *img, byte *out, int h, int w)
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

	// TODO: tweak this threshold?
	out[y*w+x] = hKer*hKer + vKer*vKer > 40000 ? 255 : 0;
}

// grayscale operator from IEEE paper; transform multi-channel into grayscale
// assume input image is 3- or 4-channel (i.e., not already grayscale)
__global__ void toGrayScale(byte *dImg, byte *dImgMono, int h, int w, int ch)
{
	int ind, y, x;

	y = blockDim.y*blockIdx.y + threadIdx.y;
	x = blockDim.x*blockIdx.x + threadIdx.x;

	if (y >= h || x >= w) {
		return;
	}

	ind = y*w*ch + x*ch;
	dImgMono[y*w + x] = 0.2989*dImg[ind] + 0.5870*dImg[ind+1]
		+ 0.1140*dImg[ind+2];
}

// convert back from single channel to multi-channel
__global__ void fromGrayScale(byte *dImgMono, byte *dImg, int h, int w, int ch)
{
	int ind, y, x;

	y = blockDim.y*blockIdx.y + threadIdx.y;
	x = blockDim.x*blockIdx.x + threadIdx.x;

	if (y >= h || x >= w) {
		return;
	}

	ind = y*w*ch + x*ch;
	dImg[ind] = dImg[ind+1] = dImg[ind+2] = dImgMono[y*w + x];
}

__host__ int main(int argc, char **argv)
{
	unsigned i, channels, rowStride, blockSize;
	byte *hImg, *dImg, *dImgMono, *dImgMonoOut;
	dim3 dimGrid, dimBlock;

	// get image
	std::cout << "Reading image from file..." << std::endl;
	read_png_file((char *)"star.png");
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

	// sobel filter
	std::cout << "Performing sobel filter edge-detection..." << std::endl;
	sobel<<<dimGrid, dimBlock>>>(dImgMono, dImgMonoOut, height, width);
	CUDAERR(cudaGetLastError(), "launch sobel kernel");

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

	std::cout << "Done." << std::endl;
}
