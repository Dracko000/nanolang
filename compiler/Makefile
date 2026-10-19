# NanoLang compiler (nanoc)
#
# Self-contained: it compiles the phase1 container writer, IR and mmap reader
# directly instead of depending on ../phase1 having been built first, so
# `make -C compiler` works from a clean checkout.
#
# Vendored phase1 sources are built with -Wno-missing-prototypes because they
# declare their prototypes at the call sites (as test_loader.c does) rather than
# in a header. Our own sources keep the full warning set.

CC       ?= cc
CSTD     ?= -std=gnu11
WARN     ?= -Wall -Wextra -Wshadow -Wpointer-arith -Wstrict-prototypes -Wmissing-prototypes
OPT      ?= -O2
CFLAGS   ?= $(CSTD) $(WARN) $(OPT) -g
CPPFLAGS += -Iinclude -I../phase1/include
LDFLAGS  ?=
LDLIBS   ?=
# Kept out of CFLAGS on purpose so a command-line CFLAGS= cannot drop it.
# Without header dependencies, editing nano_lang.h leaves stale objects built
# against the old struct layout, which links cleanly and misbehaves at run time.
DEPFLAGS := -MMD -MP

VERSION  := 0.1.0
BUILD    := build
BIN      := $(BUILD)/nanoc
TESTBIN  := $(BUILD)/test_nanoc

CORE_SRC   := src/nano_lexer.c src/nano_parser.c src/nano_sema.c \
              src/nano_codegen.c src/nano_driver.c
VENDOR_SRC := ../phase1/src/nanow_writer.c ../phase1/src/nano_ir.c
CLI_SRC    := tools/nanoc.c
TEST_SRC   := tests/test_nanoc.c
# The round-trip test opens the container with the real runtime reader.
READER_SRC := ../phase1/src/nanow_mmap.c

CORE_OBJ   := $(patsubst src/%.c,$(BUILD)/%.o,$(CORE_SRC))
VENDOR_OBJ := $(patsubst ../phase1/src/%.c,$(BUILD)/vendor_%.o,$(VENDOR_SRC))
READER_OBJ := $(patsubst ../phase1/src/%.c,$(BUILD)/vendor_%.o,$(READER_SRC))
CLI_OBJ    := $(BUILD)/nanoc.o
TEST_OBJ   := $(BUILD)/test_nanoc.o

.PHONY: all test examples clean

all: $(BIN)

$(BUILD):
	@mkdir -p $(BUILD)

$(BUILD)/%.o: src/%.c | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(DEPFLAGS) -c -o $@ $<

$(BUILD)/nanoc.o: tools/nanoc.c | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(DEPFLAGS) -DNANOC_VERSION='"$(VERSION)"' -c -o $@ $<

$(BUILD)/test_nanoc.o: tests/test_nanoc.c | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(DEPFLAGS) -c -o $@ $<

$(BUILD)/vendor_%.o: ../phase1/src/%.c | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(DEPFLAGS) -Wno-missing-prototypes -c -o $@ $<

$(BIN): $(CORE_OBJ) $(VENDOR_OBJ) $(CLI_OBJ)
	$(CC) -o $@ $^ $(LDFLAGS) $(LDLIBS)

$(TESTBIN): $(CORE_OBJ) $(VENDOR_OBJ) $(READER_OBJ) $(TEST_OBJ)
	$(CC) -o $@ $^ $(LDFLAGS) $(LDLIBS)

test: $(TESTBIN) $(BIN)
	./$(TESTBIN)
	NANOC_BIN=$(abspath $(BIN)) ./tests/run_examples.sh

examples: $(BIN)
	@for f in examples/*.nano; do \
		$(BIN) "$$f" -o "$(BUILD)/$$(basename $$f .nano).nanow" --print-manifest; \
	done

clean:
	rm -rf $(BUILD)

-include $(CORE_OBJ:.o=.d) $(VENDOR_OBJ:.o=.d) $(READER_OBJ:.o=.d) \
         $(CLI_OBJ:.o=.d) $(TEST_OBJ:.o=.d)
