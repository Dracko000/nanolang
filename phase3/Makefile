CC ?= gcc
CFLAGS ?= -O3 -mavx2 -mfma -Wall -Wextra -fPIC \
          -Iinclude -I../phase2/include -I../phase1/include -I../phase0/include

P0_OBJS = ../phase0/src/ternary_ref.o ../phase0/src/ternary_avx2.o
P1_OBJS = ../phase1/src/nanow_mmap.o ../phase1/src/ssm_step.o
P2_OBJS = ../phase2/src/nanolang_engine.o ../phase2/src/rmsnorm.o ../phase2/src/nanolang_qnn.o
P3_SRCS = src/semantic_dsl.c src/semantic_renderer.c src/nanolang_bridge.c
P3_OBJS = $(P3_SRCS:.c=.o)

ALL_OBJS = $(P0_OBJS) $(P1_OBJS) $(P2_OBJS) $(P3_OBJS)

DEMO_SRC = tools/semantic_demo.c
DEMO_BIN = bin/semantic_demo
LIB_SO   = bin/libnanolang.so

all: bin $(LIB_SO) $(DEMO_BIN)

bin:
	mkdir -p bin

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

$(LIB_SO): $(ALL_OBJS)
	$(CC) -shared $(ALL_OBJS) -o $(LIB_SO) -lm

$(DEMO_BIN): $(ALL_OBJS) $(DEMO_SRC)
	$(CC) $(CFLAGS) $(ALL_OBJS) $(DEMO_SRC) -o $(DEMO_BIN) -lm

run: $(DEMO_BIN)
	./$(DEMO_BIN)

clean:
	rm -rf bin src/*.o tools/*.o

.PHONY: all run clean
