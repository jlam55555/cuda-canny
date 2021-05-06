#include <string>
#include "canny.h"
#include "image_prep.h"
#include "clock.h"

// TODO: optimize multiplications!
// TODO: worry about memory locality later
// TODO: signed or unsigned? chars or shorts?
// TODO: try separable filters

cudaError_t err = cudaSuccess;
dim3 dimGrid, dimBlock;
bool doSync = true;

// performs a gaussian blur on an image
__host__ void blur(float blurSize, byte *dImg, byte *dImgOut)
{
        float *hFlt;
        unsigned fltSize;
        clock_t *t;

        gaussian_filter(blurSize, &hFlt, &fltSize);
	setFilter(hFlt, fltSize*fltSize*sizeof(float));

        // allocate and copy filter to device
//	CUDAERR(cudaMalloc((void **) &dFlt, fltSize*fltSize*sizeof(float)),
//		"allocating dFlt");

//	CUDAERR(cudaMemcpy(dFlt, hFlt, fltSize*fltSize*sizeof(float),
//		cudaMemcpyHostToDevice), "copying hFlt to dFlt");

        // blur image
        t = clock_start();
        conv2d<<<dimGrid, dimBlock>>>(dImg, dImgOut,
                height, width, fltSize, fltSize);
        if (doSync) {
		CUDAERR(cudaDeviceSynchronize(), "cudaDeviceSynchronize()");
		clock_lap(t, CLK_BLUR);
	}

        // cleanup
        free(hFlt);
//        CUDAERR(cudaFree(dFlt), "freeing dFlt");
}

// performs a separable gaussian blur on an image (also using shm)
__host__ void blur_sep(float blurSize, byte *dImg, byte *dImgOut)
{
	float *hFlt;
	unsigned fltSize, as;
	clock_t *t;

	gaussian_filter_1d(blurSize, &hFlt, &fltSize);
	setFilter(hFlt, fltSize*sizeof(float));
	as = fltSize/2;
	std::cout << "Blur filter size: " << fltSize << std::endl;

	dim3 dimGrid2 = dim3(ceil(width*1./(lbs-2*as)), ceil(height*1./sbs), 1);
	dim3 dimBlock2 = dim3(lbs, sbs, 1);

	dim3 dimGrid3 = dim3(ceil(width*1./sbs), ceil(height*1./(lbs-2*as)), 1);
	dim3 dimBlock3 = dim3(sbs, lbs, 1);

	// blur image
	t = clock_start();
	conv1dRows<<<dimGrid2, dimBlock2>>>(dImg, dImgOut,
		height, width, fltSize);
	if (doSync) {
		CUDAERR(cudaDeviceSynchronize(), "cudaDeviceSynchronize()");
	}

	conv1dCols<<<dimGrid3, dimBlock3>>>(dImgOut, dImg,
		height, width, fltSize);
	if (doSync) {
		CUDAERR(cudaDeviceSynchronize(), "cudaDeviceSynchronize()");
		clock_lap(t, CLK_BLUR);
	}

	// TODO: remove this
	CUDAERR(cudaMemcpy(dImgOut, dImg, width*height,
		cudaMemcpyDeviceToDevice), "TESTING");

	// cleanup
	free(hFlt);
}

// basic sobel kernel
// out is the magnitude of the gradient
// out2 is the angle of the gradient
__global__ void sobel(byte *img, byte *out, byte *out2, int h, int w)
{
	int vKer, hKer, y, x;

	y = blockDim.y*blockIdx.y + threadIdx.y;
	x = blockDim.x*blockIdx.x + threadIdx.x;

	// make sure not on edge
	if (y <= 0 || y >= h-1 || x <= 0 || x >= w-1) {
		return;
	}

	vKer = img[(y-1)*w+(x-1)]*1 + img[(y-1)*w+x]*2 + img[(y-1)*w+(x+1)]*1 +
		img[(y+1)*w+(x-1)]*-1 + img[(y+1)*w+x]*-2 + img[(y+1)*w+(x+1)]*-1;

	hKer = img[(y-1)*w+(x-1)]*1 + img[(y-1)*w+(x+1)]*-1 +
		img[y*w+(x-1)]*2 + img[y*w+(x+1)]*-2 +
		img[(y+1)*w+(x-1)]*1 + img[(y+1)*w+(x+1)]*-1;

	out[y*w+x] = out[y*w+x] = sqrtf(hKer*hKer + vKer*vKer);
	out2[y*w+x] = (byte)((atan2f(vKer,hKer)+9/8*M_PI)*4/M_PI)&0x3;
}

// shared memory sobel filter
__global__ void sobel_shm(byte *img, byte *out, byte *out2, int h, int w)
{
	int y, x;
	int vKer, hKer;
	__shared__ int tmp[bs*bs];

	y = (bs-2)*blockIdx.y + threadIdx.y-1;
	x = (bs-2)*blockIdx.x + threadIdx.x-1;

	// load data from image
	if (y>=0 && y<h && x>=0 && x<w) {
		tmp[ty*bs+tx] = img[y*w+x];
	}

	__syncthreads();

	// convolution and write-back
	if (ty>=1 && ty<bs-1 && tx>=1 && tx<bs-1 && y<h && x<w) {
		vKer = tmp[(ty-1)*bs+(tx-1)]*1 + tmp[(ty-1)*bs+tx]*2
			+ tmp[(ty-1)*bs+(tx+1)]*1 + tmp[(ty+1)*bs+(tx-1)]*-1
			+ tmp[(ty+1)*bs+tx]*-2 + tmp[(ty+1)*bs+(tx+1)]*-1;

		hKer = tmp[(ty-1)*bs+(tx-1)]*1 + tmp[(ty-1)*bs+(tx+1)]*-1 +
		       tmp[ty*bs+(tx-1)]*2 + tmp[ty*bs+(tx+1)]*-2 +
		       tmp[(ty+1)*bs+(tx-1)]*1 + tmp[(ty+1)*bs+(tx+1)]*-1;

		out[y*w+x] = sqrtf(hKer*hKer + vKer*vKer);
		out2[y*w+x] = (byte)((atan2f(vKer,hKer)+9/8*M_PI)*4/M_PI)&0x3;
	}
}

// separable (and shared memory) sobel filter
__global__ void sobel_sep(byte *img, byte *out, byte *out2, int h, int w)
{
	int y, x;

	// using int instead of byte for the following offers a 0.01s (5%)
	// speedup on the 16k image -- coalesced memory?
	int vKer, hKer;
	__shared__ int tmp1[bs*bs], tmp2[bs*bs], tmp3[bs*bs];

	y = (bs-2)*blockIdx.y + threadIdx.y-1;
	x = (bs-2)*blockIdx.x + threadIdx.x-1;

	// load data from image
	if (y>=0 && y<h && x>=0 && x<w) {
		tmp1[ty*bs+tx] = img[y*w+x];
	}

	__syncthreads();

	// first convolution
	if (ty>=1 && ty<bs-1 && tx && tx<bs) {
		tmp2[ty*bs+tx] = tmp1[(ty-1)*bs+tx]
			+ (tmp1[ty*bs+tx]<<1) + tmp1[(ty+1)*bs+tx];
	}

	if (ty && ty<bs && tx>=1 && tx<bs-1) {
		tmp3[ty*bs+tx] = tmp1[ty*bs+(tx-1)]
			+ (tmp1[ty*bs+tx]<<1) + tmp1[ty*bs+(tx+1)];
	}

	__syncthreads();

	// second convolution and write-back
	if (ty>=1 && ty<bs-1 && tx>=1 && tx<bs-1 && y<h && x<w) {
		hKer = tmp2[ty*bs+(tx-1)] - tmp2[ty*bs+(tx+1)];
		vKer = tmp3[(ty-1)*bs+tx] - tmp3[(ty+1)*bs+tx];

		out[y*w+x] = sqrtf(hKer*hKer + vKer*vKer);
		out2[y*w+x] = (byte)((atan2f(vKer,hKer)+9/8*M_PI)*4/M_PI)&0x3;
	}
}

// perform edge thinning
__global__ void edge_thin(byte *mag, byte *angle, byte *out, int h, int w)
{
	int y, x, y1, x1, y2, x2;

	y = blockDim.y*blockIdx.y + threadIdx.y;
	x = blockDim.x*blockIdx.x + threadIdx.x;

	// make sure not on the border
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

// definitions for the below two functions
#define MSK_LOW		0x0	// below threshold 1
#define MSK_THR		0x60	// at threshold 1
#define MSK_NEW		0x90	// at threshold 2, newly discovered
#define MSK_DEF		0xff	// at threshold 2 and already discovered

// perform double thresholding
__global__ void edge_thin(byte *dImg, byte *out, int h, int w, byte t1, byte t2)
{
	int y, x, ind, grad;

	y = blockDim.y*blockIdx.y + threadIdx.y;
	x = blockDim.x*blockIdx.x + threadIdx.x;

	if (y >= h || x >= w) {
		return;
	}

	ind = y*w + x;
	grad = dImg[ind];
	if (grad < t1) {
		out[ind] = MSK_LOW;
	} else if (grad < t2) {
		out[ind] = MSK_THR;
	} else {
		out[ind] = MSK_NEW;
	}
}

// check and set neighbor
#define CAS(buf, cond, x2, y2, width) \
	if ((cond) && buf[(y2)*(width)+(x2)] == MSK_THR) { \
		buf[(y2)*(width)+(x2)] = MSK_NEW; \
	}

// perform one iteration of hysteresis
__global__ void hysteresis(byte *dImg, int h, int w, bool final)
{
	int y, x;
	__shared__ byte changes;

	// infer y, x, from block/thread index
	y = blockDim.y * blockIdx.y + threadIdx.y;
	x = blockDim.x * blockIdx.x + threadIdx.x;

	// check if pixel is connected to its neighbors; continue until
	// no changes remaining
	do {
		__syncthreads();
		changes = 0;
		__syncthreads();

		// make sure inside bounds -- need this here b/c we can't have
		// __syncthreads() cause a branch divergence in a warp;
		// see https://stackoverflow.com/a/6667067/2397327

		// if newly-discovered edge, then check its neighbors
		if ((x<w && y<h) && dImg[y*w+x] == MSK_NEW) {
			// promote to definitely discovered
			dImg[y*w+x] = MSK_DEF;
			changes = 1;

			// check neighbors
			CAS(dImg,	x>0&&y>0,	x-1,	y-1,	w);
			CAS(dImg,	y>0,		x,	y-1,	w);
			CAS(dImg,	x<w-1&&y>0,	x+1,	y-1,	w);
			CAS(dImg,	x<w-1,		x+1,	y,	w);
			CAS(dImg,	x<w-1&&y<h-1,	x+1,	y+1,	w);
			CAS(dImg,	y<h-1,		x,	y+1,	w);
			CAS(dImg,	x>0&&y<h-1,	x-1,	y+1,	w);
			CAS(dImg,	x>0,		x-1,	y,	w);
		}

		__syncthreads();
	} while (changes);

	// set all threshold1 values to 0
	if (final && (x<w && y<h) && dImg[y*w+x] != MSK_DEF) {
		dImg[y*w+x] = 0;
	}
}

// shared memory version of hysteresis
__global__ void hysteresis_shm(byte *dImg, int h, int w, bool final)
{
	int y, x;
	bool in_bounds;
	__shared__ byte changes, tmp[bs*bs];

	// infer y, x, from block/thread index
	y = (bs-2)*blockIdx.y + ty-1;
	x = (bs-2)*blockIdx.x + tx-1;

	in_bounds = (x<w && y<h) && (tx>=1 && tx<bs-1 && ty>=1 && ty<bs-1);

	if (y>=0 && y<h && x>=0 && x<w) {
		tmp[ty*bs+tx] = dImg[y*w+x];
	}

	__syncthreads();

	// check if pixel is connected to its neighbors; continue until
	// no changes remaining
	do {
		__syncthreads();
		changes = 0;
		__syncthreads();

		// make sure inside bounds -- need this here b/c we can't have
		// __syncthreads() cause a branch divergence in a warp;
		// see https://stackoverflow.com/a/6667067/2397327

		// if newly-discovered edge, then check its neighbors
		if (in_bounds && tmp[ty*bs+tx] == MSK_NEW) {
			// promote to definitely discovered
			tmp[ty*bs+tx] = MSK_DEF;
			changes = 1;

			// check neighbors
			CAS(tmp, 1,		tx-1,	ty-1,	bs);
			CAS(tmp, 1,		tx,	ty-1,	bs);
			CAS(tmp, x<w-1,		tx+1,	ty-1,	bs);
			CAS(tmp, x<w-1,		tx+1,	ty,	bs);
			CAS(tmp, x<w-1&&y<h-1,	tx+1,	ty+1,	bs);
			CAS(tmp, y<h-1,		tx,	ty+1,	bs);
			CAS(tmp, y<h-1,		tx-1,	ty+1,	bs);
			CAS(tmp, 1,		tx-1,	ty,	bs);
		}

		__syncthreads();
	} while (changes);

	if (y>=0 && y<h && x>=0 && x<w) {
		if (final) {
			if (in_bounds) {
				dImg[y*w+x] = MSK_DEF*(tmp[ty*bs+tx]==MSK_DEF);
			}
		} else {
			dImg[y*w+x] = max(dImg[y*w+x], tmp[ty*bs+tx]);
		}
	}
}

// perform canny edge detection
__host__ void canny(byte *dImg, byte *dImgOut,
	float blurStd, float threshold1, float threshold2, int hystIters)
{
	byte *dImgTmp;
	clock_t *t;
	int i;

	CUDAERR(cudaMalloc((void**)&dImgTmp, width*height), "alloc dImgTmp");

//	blur(blurStd, dImg, dImgOut);
	blur_sep(blurStd, dImg, dImgOut);

	// different grid with 1-width apron for shared-memory schemes
	dim3 dimGrid2 = dim3(ceil(width*1./(bs-2)), ceil(height*1./(bs-2)), 1);
	dim3 dimBlock2 = dim3(bs, bs, 1);

	t = clock_start();
	std::cout << "Performing Sobel filter..." << std::endl;
//	sobel<<<dimGrid, dimBlock>>>(dImgOut, dImg, dImgTmp,
//		height, width);
	sobel_shm<<<dimGrid2, dimBlock2>>>(dImgOut, dImg, dImgTmp,
		height, width);
//	sobel_sep<<<dimGrid2, dimBlock2>>>(dImgOut, dImg, dImgTmp,
//		height, width);
	CUDAERR(cudaGetLastError(), "launch sobel kernel");
	if (doSync) {
		CUDAERR(cudaDeviceSynchronize(), "cudaDeviceSynchronize()");
		clock_lap(t, CLK_SOBEL);
	}

	std::cout << "Performing edge thinning..." << std::endl;
	edge_thin<<<dimGrid, dimBlock>>>(dImg, dImgTmp, dImgOut,
		height, width);
	CUDAERR(cudaGetLastError(), "launch edge thinning kernel");
	if (doSync) {
		CUDAERR(cudaDeviceSynchronize(), "cudaDeviceSynchronize()");
		clock_lap(t, CLK_THIN);
	}

	std::cout << "Performing double thresholding..." << std::endl;
	edge_thin<<<dimGrid, dimBlock>>>(dImgOut, dImgTmp, height, width,
		255*threshold1, 255*threshold2);
	CUDAERR(cudaGetLastError(), "launch double thresholding kernel");
	if (doSync) {
		CUDAERR(cudaDeviceSynchronize(), "cudaDeviceSynchronize()");
		clock_lap(t, CLK_THRES);
	}

	if (hystIters) {
		std::cout << "Performing hysteresis..." << std::endl;
		for (i = 0; i < hystIters; ++i) {
//			hysteresis<<<dimGrid, dimBlock>>>(dImgTmp,
//				height, width, i==hyst_iters-1);
			hysteresis_shm<<<dimGrid2, dimBlock2>>>(dImgTmp,
				height, width, i==hystIters-1);
			CUDAERR(cudaGetLastError(),
	   			"launch hysteresis kernel");
			if (doSync) {
				CUDAERR(cudaDeviceSynchronize(),
					"cudaDeviceSynchronize()");
				clock_lap(t, CLK_HYST);
			}
		}
	}

	// TODO: remove this
	CUDAERR(cudaMemcpy(dImgOut, dImgTmp, width*height,
		cudaMemcpyDeviceToDevice), "TESTING");

	// dTmp = dImg;
	// dImg = dImgOut;
	// dImgOut = dTmp;

	CUDAERR(cudaFree(dImgTmp), "freeing dImgTmp");
}

// print timings
__host__ void print_timings(void)
{
	std::cout << "overall:\t" << clock_ave[CLK_ALL] << "s" << std::endl;

	// doSync off means only overall time counted
	if (!doSync) {
		return;
	}

	std::cout << "grayscale:\t" << clock_ave[CLK_GRAY] << "s" << std::endl
		<< "blur:\t\t" << clock_ave[CLK_BLUR] << "s" << std::endl
		<< "sobel\t\t" << clock_ave[CLK_SOBEL] << "s" << std::endl
		<< "edgethin:\t" << clock_ave[CLK_THIN] << "s" << std::endl
		<< "threshold:\t" << clock_ave[CLK_THRES] << "s" << std::endl
		<< "hysteresis:\t" << clock_ave[CLK_HYST] << "s" << std::endl
		<< "hyst total:\t" << clock_total[CLK_HYST] << "s" << std::endl;
}

__host__ int main(void)
{
	std::string inFile, outFile;
	unsigned i, channels, rowStride, hystIters;
	byte *hImg, *dImg, *dImgMono, *dImgMonoOut;
	float blurStd, threshold1, threshold2;
	clock_t *tGray, *tOverall;

	// get image name
	std::cout << "Enter infile (without .png): ";
	std::cin >> inFile;

	std::cout << "Enter outfile (without .png): ";
	std::cin >> outFile;

	std::cout << "Blur stdev: ";
	std::cin >> blurStd;

	std::cout << "Threshold 1: ";
	std::cin >> threshold1;

	std::cout << "Threshold 2: ";
	std::cin >> threshold2;

	std::cout << "Hysteresis iters: ";
	std::cin >> hystIters;

	std::cout << "Sync after each kernel? ";
	std::cin >> doSync;

	inFile += ".png";
	outFile += "_bs" + std::to_string(blurStd)
		+ "_th" + std::to_string(threshold1)
		+ "_th" + std::to_string(threshold2)
		+ (hystIters ? "" : "_nohyst") + ".png";

	// get image
	std::cout << "Reading image from file..." << std::endl;
	read_png_file(inFile.c_str());
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
	dimGrid = dim3(ceil(rowStride*1./bs), ceil(height*1./bs), 1);
	dimBlock = dim3(bs, bs, 1);

	// convert to grayscale
	cudaDeviceSynchronize();
	tOverall = clock_start();
	tGray = clock_start();
	std::cout << "Converting to grayscale..." << std::endl;
	toGrayScale<<<dimGrid, dimBlock>>>(dImg, dImgMono, height, width,
		channels);
	CUDAERR(cudaGetLastError(), "launch toGrayScale kernel");
	if (doSync) {
		cudaDeviceSynchronize();
		clock_lap(tGray, CLK_GRAY);
	}

	// canny edge detection
	std::cout << "Performing canny edge-detection..." << std::endl;
	canny(dImgMono, dImgMonoOut, blurStd, threshold1, threshold2,
		hystIters);

	// convert back from grayscale
	tGray = clock_start();
	std::cout << "Convert image back to multi-channel..." << std::endl;
	fromGrayScale<<<dimGrid, dimBlock>>>(dImgMonoOut, dImg,
		height, width, channels);
	CUDAERR(cudaGetLastError(), "launch fromGrayScale kernel");
	cudaDeviceSynchronize();
	if (doSync) {
		clock_lap(tGray, CLK_GRAY);
	}
	clock_lap(tOverall, CLK_ALL);

	// copy image back to host
	std::cout << "Copy image back to host..." << std::endl;
	CUDAERR(cudaMemcpy(hImg, dImg, width*height*channels,
		cudaMemcpyDeviceToHost), "cudaMemcpy to host");

	// copy image back to row_pointers
	std::cout << "Copy image back to row_pointers..." << std::endl;
	for (i = 0; i < height; ++i) {
		memcpy(row_pointers[i], hImg + i*rowStride, rowStride);
	}

	print_timings();

	// copy image back from device
	std::cout << "Writing image back to file..." << std::endl;
	write_png_file(outFile.c_str());

	// freeing pointers
	std::cout << "Freeing device memory..." << std::endl;
	CUDAERR(cudaFree(dImg), "freeing dImg");
	CUDAERR(cudaFree(dImgMono), "freeing dImgMono");
	CUDAERR(cudaFree(dImgMonoOut), "freeing dImgMonoOut");

	std::cout << "Done." << std::endl;
}
