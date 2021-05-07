# Canny Edge Detection in CUDA C++

Based on the work in [Canny edge detection on NVIDIA CUDA][paper]

---

### Paper
(Coming soon!)

---

### Build Instructions
##### CUDA Application
For the CUDA application, you may need to specify the appropriate
flags for the Makefile (see an example in [buildx86.sh][buildx86])
```bash
$ make -C canny_cuda CUDA_PATH=/usr TARGET_ARCH=x86_64 SMS=30 [TARGET]
```
where the target is one of `all`, `clean`, etc. Specify the `dbg=1` Makefile flag to enable debugging. See the [Makefile](canny_cuda/Makefile) for more build flags.

##### CPU Application
For the CPU verison, go to the canny dir first
``` bash
$ make
``` 
If need to clean the old excutable
```bash
$ make clean
```
where the target is one of `all`, `clean`, etc. See the [Makefile](canny/Makefile) for more build flags.

---

### Run Instructions
##### CUDA Application
The Makefile should build the CUDA application to `canny_cuda/canny`. When you run the application, it will prompt you for several options (this assumes you have a file called `res/lizard.png` and a directory called `out` relative to the current working directory):
```bash
$ canny_cuda/canny 
Enter infile (without .png): res/lizard
Enter outfile (without .png): out/lizard
Blur stdev: 2
Threshold 1: 0.2
Threshold 2: 0.4
Hysteresis iters: 5
Sync after each kernel? 1
Reading image from file...
Channels: 3
Allocating host and device buffers...
Copying image to device...
Converting to grayscale...
Performing canny edge-detection...
Blur filter size: 13
Performing Sobel filter...
Performing edge thinning...
Performing double thresholding...
Performing hysteresis...
Convert image back to multi-channel...
Copy image back to host...
Copy image back to row_pointers...
overall:        0.007235s
grayscale:      0.0006155s
blur:           0.001099s
sobel           0.000663s
edgethin:       0.000579s
threshold:      0.000435s
hysteresis:     0.0005142s
hyst total:     0.002571s
Writing image back to file...
Freeing device memory...
Done.
```
This will generate a PNG file with the filename:
```text
[OUTFILE]_bs[BLURSIZE]_th[THRESHOLD1]_th[THRESHOLD2].png
```

##### CPU Version Application
The Makefile should build the C application to `canny/canny`. When you run the application, it will prompt you for several options 
```bash
$ ./canny [output.png] [input.png]
```
The prompts should be like something in the following
```bash
user-name@host-name canny % ./canny star.png t.png
Reading the image...
channel is :3
blurSize: 2.000000 
Performing conv2d...blur time: 0.301269
Performing Sobel filter...
sobel time: 0.011020
edge_thin time: 0.003468
edge_thin_double time: 0.001077
Performing Hysteresis Thresholding...
hyster time: 0.003468
Convert image back to multi-channel...
Copy image back to row_pointers...
Writing image back to file...
Done...
time cost: 0.323120


```



[paper]: https://ieeexplore.ieee.org/abstract/document/4563088
