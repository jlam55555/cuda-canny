# Canny Edge Detection in CUDA C++

Based on the work in [Canny edge detection on NVIDIA CUDA][paper]

---

### Paper
(Coming soon!)

---
### TODO

implement Sobel filter in cuda c &amp; c and try to compare the efficiency

implement Canny filter in cuda c &amp; c

implement Laplacian filter in cuda c &amp; c

TODO:
- Implement basic Sobel filter w/ 3x3 horizontal and vertical kernels
    - Implement on CUDA
    - Implement on CPU
- Implement generic larger convolution method
- Implement Sobel filter with larger convolution
- Attempt to optimize larger convolution using overlap/save
- Attempt Sobel filters at different angles to see if we can get better results
- Compile results!
- Write report!

[paper]: https://ieeexplore.ieee.org/abstract/document/4563088