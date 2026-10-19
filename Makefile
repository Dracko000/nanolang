# Aggregate build for the NanoLang C components.
#
#   make            build everything the Python runner needs
#   make lib        just phase0/bin/libternary.so (required by run_smollm2.py)
#   make test       run the C loaders/demos plus all Python unit tests
#   make clean      remove build artifacts
#
# The Python runner itself needs no build step; it loads libternary.so through
# ctypes at start-up.

CC ?= gcc
PHASES := phase0 phase1 phase2 phase3

.PHONY: all lib test test-c test-py clean

all: $(addprefix build-,$(PHASES))

lib: build-phase0

$(addprefix build-,$(PHASES)):
	$(MAKE) -C $(patsubst build-%,%,$@) all

test-c: all
	$(MAKE) -C phase1 run
	$(MAKE) -C phase2 run
	$(MAKE) -C phase3 run
	python3 phase0/python/test_bitnet.py
	python3 phase3/python/test_semantic_dsl.py

test-py: lib
	python3 -m unittest tests.test_pipeline -v
	python3 -m unittest tests.test_tools -v

test: test-c test-py

clean:
	@for p in $(PHASES); do $(MAKE) -C $$p clean; done
