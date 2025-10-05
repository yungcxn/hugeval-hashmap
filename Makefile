NVCC = nvcc
CFLAGS = -O3 -arch=sm_80

all: clean test run

test: test.cu 
	$(NVCC) $(CFLAGS) -o test test.cu
	chmod +x ./test

run: test
	./test

clean:
	rm -f test