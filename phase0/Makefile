CC ?= gcc
# -fopenmp parallelizes the packed GEMV across cores; if the compiler drops the
# flag the `#pragma omp` pragmas are simply ignored (graceful single-thread).
CFLAGS ?= -O3 -mavx2 -mfma -fopenmp -Wall -Wextra -Iinclude -fPIC

SRCS = src/ternary_ref.c src/ternary_avx2.c src/nano_elementwise.c
OBJS = $(SRCS:.c=.o)
BENCH_SRC = bench/benchmark.c
BENCH_BIN = bin/benchmark
# T1 roofline probe: read floor vs GEMV, warm vs DRAM. Links the shipped .so
# so it measures the kernels the runner actually calls.
PROBE_SRC = bench/stream_probe.c
PROBE_BIN = bin/stream_probe
LIB = bin/libternary.so

all: $(BENCH_BIN) $(LIB)

bin:
	mkdir -p bin

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

$(BENCH_BIN): bin $(OBJS) $(BENCH_SRC)
	$(CC) $(CFLAGS) $(OBJS) $(BENCH_SRC) -o $(BENCH_BIN) -lm

# Shared library for the Python inference runner (ctypes).
$(LIB): bin $(OBJS)
	$(CC) $(CFLAGS) -shared $(OBJS) -o $(LIB) -lm

lib: $(LIB)

$(PROBE_BIN): bin $(LIB) $(PROBE_SRC)
	$(CC) $(CFLAGS) $(PROBE_SRC) -L bin -lternary -o $(PROBE_BIN) -lm

probe: $(PROBE_BIN)
	LD_LIBRARY_PATH=bin OMP_NUM_THREADS=3 ./$(PROBE_BIN)

run: $(BENCH_BIN)
	./$(BENCH_BIN)

clean:
	rm -rf bin $(OBJS)

.PHONY: all lib probe run clean
