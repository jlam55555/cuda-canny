#include "blur.h"
typedef unsigned char byte;
void gaussian_filter(float blurStd, float **fltp, unsigned *fltSizep)
{
        float fltSum, cent, *flt;
        unsigned i, j, fltSize;

        
        fltSize = 6*blurStd+1;  

	cent = (fltSize-1.)/2;

        flt = (float *) malloc(fltSize*fltSize*sizeof(float));

        fltSum = 0;
        for (i = 0; i < fltSize; ++i) {
                for (j = 0; j < fltSize; ++j) {
                        flt[i*fltSize+j] = exp(-(pow(i-cent,2)+pow(j-cent,2))
                                /(2*blurStd*blurStd))/(2*M_PI*blurStd*blurStd);
                        fltSum += flt[i*fltSize+j];
                }
        }

        for (i = 0; i < fltSize*fltSize; ++i) {
                flt[i] /= fltSum;
        }

        *fltSizep = fltSize;
        *fltp = flt;
}

void conv2d(byte *input, float *filt, byte *output, int h1, int w1, int h2, int w2)
{
        int y, x, i, j, imin, imax, jmin, jmax, ip, jp;
        float sum;

        for(y = 0; y < h1; y++){
                for(x = 0; x < w1; x++){
                        // appropriate ranges for convolution
                        imin = max(0, y+h2/2-h2+1);
                        imax = min(h1, y+h2/2+1);
                        jmin = max(0, x+w2/2-w2+1);
                        jmax = min(w1, x+w2/2+1);

                        // convolution
                        // TODO: this only deals with the case where filt has a single channel
                        //      (i.e., like a filter)
                        sum = 0;
                        for (i = imin; i < imax; ++i) {
                                for (j = jmin; j < jmax; ++j) {
                                        ip = i - h2/2;
                                        jp = j - w2/2;

                                        sum += input[i*w1 + j] * filt[(y-ip)*w2 + (x-jp)];
                                }
                        }

                        // set result
                        output[y*w1 + x] = sum;
                }
        }
	
}

void dfs(byte* input, byte* output){
        
        return;
}