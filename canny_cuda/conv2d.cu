#include "canny.h"

/**
 * performs a 2D convolution of an single-channel image d1 with a single-channel
 * filter filt, resulting in an image with the same dimensions as and centered
 * around d1
 */
__global__ void conv2d(byte *d1, float *filt, byte *d3,
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

			sum += d1[i*w1 + j] * filt[(y-ip)*w2 + (x-jp)];
		}
	}

	// set result
	d3[y*w1 + x] = sum;
}