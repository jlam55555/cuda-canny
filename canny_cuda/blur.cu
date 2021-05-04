#include "canny.h"

// returns a circular gaussian filter with standard deviation blurStd; returns
// the dimensions of the filter in fltSize
__host__ void gaussian_filter(float blurStd, float **fltp, unsigned *fltSizep)
{
        float fltSum, cent, *flt;
        unsigned i, j, fltSize;

        // initialize filter; 3x3 circular gaussian filter
        // https://en.wikipedia.org/wiki/Gaussian_blur
        fltSize = 6*blurStd+1;  // for factor of 6 see Wikipedia
                                // +1 to make it odd for better centering
	cent = (fltSize-1.)/2;  // center of filter

        flt = (float *) malloc(fltSize*fltSize*sizeof(float));

        fltSum = 0;
        for (i = 0; i < fltSize; ++i) {
                for (j = 0; j < fltSize; ++j) {
                        flt[i*fltSize+j] = exp(-(pow(i-cent,2)+pow(j-cent,2))
                                /(2*blurStd*blurStd))/(2*M_PI*blurStd*blurStd);
                        fltSum += flt[i*fltSize+j];
                }
        }

        // normalize the filter
        for (i = 0; i < fltSize*fltSize; ++i) {
                flt[i] /= fltSum;
        }

        *fltSizep = fltSize;
        *fltp = flt;
}

// returns a 1D gaussian filter with standard deviation blurStd; returns
// the dimensions of the filter in fltSize
__host__ void gaussian_filter_1d(float blurStd, float **fltp,
	unsigned *fltSizep)
{
	float fltSum, cent, *flt;
	unsigned i, fltSize;

	// initialize filter; 3x3 circular gaussian filter
	// https://en.wikipedia.org/wiki/Gaussian_blur
	fltSize = 6*blurStd+1;  // for factor of 6 see Wikipedia
	// +1 to make it odd for better centering
	cent = (fltSize-1.)/2;  // center of filter

	flt = (float *) malloc(fltSize*sizeof(float));

	fltSum = 0;
	for (i = 0; i < fltSize; ++i) {
		flt[i] = exp(-pow((i-cent)/blurStd,2)/2)/(blurStd*sqrt(2*M_PI));
		fltSum += flt[i];
	}

	// normalize the filter
	for (i = 0; i < fltSize; ++i) {
		flt[i] /= fltSum;
	}

	*fltSizep = fltSize;
	*fltp = flt;
}