CC ?= gcc
# -fopenmp parallelizes the packed GEMV across cores; if the compiler drops the
# flag the `#pragma omp` pragmas are simply ignored (graceful single-thread).
CFLAGS ?= -O3 -mavx2 -mfma -fopenmp -Wall -Wextra -Iinclude -fPIC

SRCS = src/ternary_ref.c src/ternary_avx2.c src/nano_elementwise.c
OBJS = $(SRCS:.c=.o)
BENCH_SRC = bench/benchmark.c
BENCH_BIN = bin/benchmark
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

run: $(BENCH_BIN)
	./$(BENCH_BIN)

clean:
	rm -rf bin $(OBJS)

.PHONY: all lib run clean
