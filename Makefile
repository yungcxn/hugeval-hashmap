NVCC = nvcc
CFLAGS = -O3 -arch=sm_80

all: test

test: test.cu 
	$(NVCC) $(CFLAGS) -o test test.cu

run: test
	./test

clean:
	rm -f test