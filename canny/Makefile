all:	canny.c image_prep.c blur.c ../sobel.c
	gcc canny.c ../sobel.c image_prep.c blur.c -o canny -lpng
clean: 
	$(RM) canny
