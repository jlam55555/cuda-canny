#include "canny.h"

// see note about constant memory: https://stackoverflow.com/a/2452813/2397327
__constant__ float dFlt[1024];
__host__ void setFilter(float *flt, unsigned size)
{
	CUDAERR(cudaMemcpyToSymbol(dFlt, flt, size),
		"copying flt to dFlt");
}

/**
 * performs a 2D convolution of an single-channel image d1 with a single-channel
 * filter filt, resulting in an image with the same dimensions as and centered
 * around d1
 */
__global__ void conv2d(byte *d1, byte *d3,
        int h1, int w1, int h2, int w2)
{
        int y, x, i, j, imin, imax, jmin, jmax, ip, jp;
        float sum;

        // infer y, x, from block/thread index
        y = blockDim.y * blockIdx.y + threadIdx.y;
        x = blockDim.x * blockIdx.x + threadIdx.x;

        // out of bounds, no work to do
        if (x >= w1 || y >= h1) {
                return;
        }

	// appropriate ranges for convolution
	imin = max(0, y+h2/2-h2+1);
	imax = min(h1, y+h2/2+1);
	jmin = max(0, x+w2/2-w2+1);
	jmax = min(w1, x+w2/2+1);

	// convolution
	sum = 0;
	for (i = imin; i < imax; ++i) {
		for (j = jmin; j < jmax; ++j) {
			ip = i - h2/2;
			jp = j - w2/2;

			sum += d1[i*w1 + j] * dFlt[(y-ip)*w2 + (x-jp)];
		}
	}

	// set result
	d3[y*w1 + x] = sum;
}

// 1D convolution along x-direction with shared memory: loads in an apron
// assumed that block size in x direction is lbs (long blocksize = 64),
// and block size in y direction is sbs (short blocksize = 16) for performance;
// also best to have filt be in constant memory for performance purposes
__global__ void conv1dRows(byte *dIn, byte *dOut,
	int h, int w, int fltSize)
{
	int y, x, as, i, j;
	float sum;
	__shared__ byte tmp[lbs*sbs];

	as = fltSize>>1; // apron size

	// infer y, x, from block/thread index
	// note extra operations based on apron for x
	y = sbs * blockIdx.y + ty;
	x = (lbs-(as<<1)) * blockIdx.x + tx-as;

	// load data
	if (y<h && x>=0 && x<w) {
		tmp[ty*lbs+tx] = dIn[y*w+x];
	}

	__syncthreads();

	// perform 1-D convolution
	if (tx>=as && tx<lbs-as && y<h && x<w) {
		for (i = ty*lbs+tx-as, j = 0, sum = 0; j < fltSize; ++i, ++j) {
			sum += dFlt[j] * tmp[i];
		}

		// set result
		dOut[y*w+x] = sum;
	}
}

// same as above but with columns; assumes that blocksize is 64 in the column
// direction and blocksize is 16 in the row dimension
__global__ void conv1dCols(byte *dIn, byte *dOut,
	int h, int w, int fltSize)
{
	int y, x, as, i, j;
	float sum;
	__shared__ byte tmp[lbs*sbs];

	as = fltSize>>1; // apron size

	// infer y, x, from block/thread index
	// note extra operations based on apron for x
	y = (lbs-(as<<1)) * blockIdx.y + ty-as;
	x = sbs * blockIdx.x + tx;

	// load data
	if (y>=0 && y<h && x<w) {
		tmp[ty*sbs+tx] = dIn[y*w+x];
	}

	__syncthreads();

	// perform 1-D convolution
	if (ty>=as && ty<lbs-as && y<h && x<w) {
		for (i = (ty-as)*sbs+tx, j = 0, sum = 0; j < fltSize;
			i+=sbs, ++j) {
			sum += dFlt[j] * tmp[i];
		}

		// set result
		dOut[y*w+x] = sum;
	}
}