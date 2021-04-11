// TODO: optimize multiplications!
// TODO: worry about memory locality later
// TODO: signed or unsigned? chars or shorts?
// TODO: try separable filters

__global__ void sobelVKernel(char *img, char *out, int y, int x, int h, int w)
{
	int vKer, hKer;

	if (y <= 0 || y >= h-1 || x <= 0 || x >= w-1) {
		out[y*w+x] = 0;
		return;
	}

	vKer = img[(y-1)*w+(x-1)]*1 + img[(y-1)*w+x]*2 + img[(y-1)*w+(x+1)]*1 +
		img[(y+1)*w]*-1 + img[(y+1)*w+x]*-2 + img[(y+1)*w+(x+1)]*-1;

	hKer = img[(y-1)*w+(x-1)]*1 + img[(y-1)*w+(x+1)]*-1 +
		img[y*w+(x-1)]*2 + img[y*w+(x+1)]*-2 +
		img[(y+1)*w]*1 + img[(y+1)*w+(x+1)]*-1;

	out[y*w+x] = sqrt(vKer*vKer + hKer*hKer);
}

// grayscale operator from IEEE paper
// assume input image is 3- or 4-channel (i.e., not already grayscale)
__global__ void toGrayScale(char *img, int y, int x, int h, int w, int ch)
{
	int ind;

	if (y >= h || x >= w) {
		return;
	}

	ind = y*w*ch;
	return 0.2989*img[ind] + 0.5870*img[ind] + 0.1140*img[ind];
}

__host__ int main(int argc, char **argv)
{
	// TODO: 

	// TODO: calculate best grid/block dim
}