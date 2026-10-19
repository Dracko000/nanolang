CC ?= gcc
# -fopenmp is required at link time as well as compile time: the phase0 objects
# pulled in here (ternary_avx2.c) reference GOMP_parallel and omp_get_*.
CFLAGS ?= -O3 -mavx2 -mfma -fopenmp -Wall -Wextra -Iinclude -I../phase1/include -I../phase0/include

P0_SRCS = ../phase0/src/ternary_ref.c ../phase0/src/ternary_avx2.c
P1_SRCS = ../phase1/src/nanow_mmap.c ../phase1/src/ssm_step.c
P2_SRCS = src/nanolang_engine.c src/rmsnorm.c src/nanolang_qnn.c

ALL_SRCS = $(P0_SRCS) $(P1_SRCS) $(P2_SRCS)
OBJS = $(ALL_SRCS:.c=.o)

CLI_SRC = tools/nanolang_cli.c
CLI_BIN = bin/nanolang_cli

all: $(CLI_BIN)

bin:
	mkdir -p bin

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

$(CLI_BIN): bin $(OBJS) $(CLI_SRC)
	$(CC) $(CFLAGS) $(OBJS) $(CLI_SRC) -o $(CLI_BIN) -lm

run: $(CLI_BIN)
	./$(CLI_BIN) ../phase1/model_sample.nanow

clean:
	rm -rf bin src/*.o tools/*.o

.PHONY: all run clean
