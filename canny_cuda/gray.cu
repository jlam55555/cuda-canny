#include "canny.h"

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

	// >30% speedup in this function when not using fp
//	dImgMono[y*w + x] = 0.2989*dImg[ind] + 0.5870*dImg[ind+1]
//		+ 0.1140*dImg[ind+2];

	dImgMono[y*w + x] = (2989*dImg[ind] + 5870*dImg[ind+1]
		+ 1140*dImg[ind+2])/10000;
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