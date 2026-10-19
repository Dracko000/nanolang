CC ?= gcc
CFLAGS ?= -O3 -Wall -Wextra -Iinclude

SRCS = src/nanow_writer.c src/nanow_mmap.c src/ssm_step.c src/nano_ir.c
OBJS = $(SRCS:.c=.o)
TEST_SRC = tools/test_loader.c
TEST_BIN = bin/test_loader

all: $(TEST_BIN)

bin:
	mkdir -p bin

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

$(TEST_BIN): bin $(OBJS) $(TEST_SRC)
	$(CC) $(CFLAGS) $(OBJS) $(TEST_SRC) -o $(TEST_BIN) -lm

run: $(TEST_BIN)
	./$(TEST_BIN)

clean:
	rm -rf bin $(OBJS) /root/llm/phase1/model_sample.nanow

.PHONY: all run clean
