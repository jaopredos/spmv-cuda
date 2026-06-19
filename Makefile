CC        = gcc
NVCC      = nvcc
ARCH      = -arch=sm_89

CFLAGS    = -O3 -fopenmp -Wall -Iinclude
NVCCFLAGS = -O3 $(ARCH) -Xcompiler -fopenmp -Iinclude
LDFLAGS   = -Xcompiler -fopenmp -lm

BUILD_DIR = build

C_SRCS_COMMON = src/matrix.c src/csr.c src/sp24.c src/macko.c src/spmv_cpu.c
CU_SRCS       = src/spmv_cuda.cu src/spmv_cuda_macko.cu

C_OBJS_COMMON = $(C_SRCS_COMMON:src/%.c=$(BUILD_DIR)/%.o)
CU_OBJS       = $(CU_SRCS:src/%.cu=$(BUILD_DIR)/%.o)

BENCH_OBJ = $(BUILD_DIR)/benchmark.o
GEN_OBJ   = $(BUILD_DIR)/generator.o

.PHONY: all clean

all: benchmark generator

benchmark: $(BENCH_OBJ) $(C_OBJS_COMMON) $(CU_OBJS)
	$(NVCC) $(ARCH) -o $@ $^ $(LDFLAGS)

generator: $(GEN_OBJ) $(BUILD_DIR)/matrix.o $(BUILD_DIR)/csr.o $(BUILD_DIR)/sp24.o
	$(CC) -o $@ $^ -lm

$(BUILD_DIR)/%.o: src/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: src/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR) benchmark generator
