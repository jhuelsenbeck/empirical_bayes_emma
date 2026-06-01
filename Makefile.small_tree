# =============================================================================
#  Bayesian Inference of Phylogeny using Profile Likelihoods
#  Top-level Makefile
#
#  Builds on macOS (Apple Silicon arm64 or Intel x86_64) and other Unix systems.
#  Place this file in the directory that contains small_tree_src/ and shared_src/.
#
#  Quick start:
#     make                 # optimized native build (host architecture)
#     make -j              # same, parallel
#     make debug           # -O0 -g with assertions
#     make ARCH=arm64      # force Apple Silicon
#     make ARCH=x86_64     # force Intel
#     make universal       # fat binary (arm64 + x86_64), macOS only
#     make clean
#     make run ARGS="-i data.nex -o out"
#
#  Common overrides (all optional):
#     make CXX=g++-14            # pick a different compiler
#     make OPT="-O3 -march=native"
#     make EXTRA_CXXFLAGS=-DNDEBUG
# =============================================================================

# ---- What we are building ---------------------------------------------------
TARGET     := small_tree

# ---- Project layout ---------------------------------------------------------
SMALL_DIR  := small_tree_src
SHARED_DIR := shared_src
NCL_DIR    := $(SHARED_DIR)/ncl
BUILD_DIR  := build

# ---- Toolchain --------------------------------------------------------------
# On macOS, clang++ (and c++) resolve to the Apple toolchain. Override with
# `make CXX=...` to use Homebrew gcc, a specific clang, etc.
CXX        ?= clang++

# ---- User-tunable knobs -----------------------------------------------------
OPT            ?= -O2                  # optimization level for the default build
ARCH           ?= native               # native | arm64 | x86_64 | universal
EXTRA_CXXFLAGS ?=                      # appended to compile flags
EXTRA_LDFLAGS  ?=                      # appended to link flags
ARGS           ?=                      # arguments passed by `make run`

# Inline comments above leave trailing spaces in the values; strip them so the
# ifeq comparisons below see a clean token.
ARCH       := $(strip $(ARCH))
OPT        := $(strip $(OPT))

# ---- Fixed compiler settings ------------------------------------------------
STD        := -std=c++17
INCLUDES   := -I$(SMALL_DIR) -I$(SHARED_DIR) -I$(NCL_DIR)

# Keep the build readable: -Wall for our own code, silence the noisiest
# diagnostics coming out of the third-party NCL sources.
WARN       := -Wall \
              -Wno-sign-compare \
              -Wno-unused-variable \
              -Wno-unused-parameter \
              -Wno-deprecated-declarations

# ---- Architecture selection (macOS) -----------------------------------------
# `native` adds no -arch flag, so the compiler targets the host CPU. This is
# the right default and also keeps the Makefile usable on Linux. The explicit
# arm64 / x86_64 / universal options are meaningful only with Apple clang.
ifeq ($(ARCH),native)
  ARCHFLAGS :=
else ifeq ($(ARCH),arm64)
  ARCHFLAGS := -arch arm64
else ifeq ($(ARCH),x86_64)
  ARCHFLAGS := -arch x86_64
else ifeq ($(ARCH),universal)
  ARCHFLAGS := -arch arm64 -arch x86_64
else
  $(error Unknown ARCH '$(ARCH)'. Use one of: native, arm64, x86_64, universal)
endif

# Header-dependency tracking. clang cannot emit dependency files while building
# more than one -arch at once, so disable it for universal binaries.
ifeq ($(ARCH),universal)
  DEPFLAGS :=
else
  DEPFLAGS := -MMD -MP
endif

# ---- Assembled flags --------------------------------------------------------
# -pthread is required by Threads.cpp (std::thread / mutex / condition_variable);
# it is accepted by Apple clang and needed at link time on Linux.
CXXFLAGS := $(STD) $(OPT) $(WARN) $(ARCHFLAGS) $(INCLUDES) -pthread $(DEPFLAGS) $(EXTRA_CXXFLAGS)
LDFLAGS  := $(ARCHFLAGS) -pthread $(EXTRA_LDFLAGS)

# ---- Source discovery -------------------------------------------------------
SRCS := $(wildcard $(SMALL_DIR)/*.cpp) \
        $(wildcard $(SHARED_DIR)/*.cpp) \
        $(wildcard $(NCL_DIR)/*.cpp)

# Object/dependency files live under build/, mirroring the source tree.
OBJS := $(SRCS:%.cpp=$(BUILD_DIR)/%.o)
DEPS := $(OBJS:.o=.d)

# =============================================================================
#  Targets
# =============================================================================
.PHONY: all debug release universal run clean distclean info

all: $(TARGET)

# Convenience profiles ---------------------------------------------------------
debug:   OPT := -O0 -g
debug:   EXTRA_CXXFLAGS += -DDEBUG -fno-omit-frame-pointer
debug:   $(TARGET)

release: OPT := -O3
release: EXTRA_CXXFLAGS += -DNDEBUG
release: $(TARGET)

# Fat binary for distribution on both Apple Silicon and Intel Macs.
universal:
	$(MAKE) ARCH=universal

# Link step --------------------------------------------------------------------
$(TARGET): $(OBJS)
	@echo "  LINK   $@"
	$(CXX) $(OBJS) $(LDFLAGS) -o $@

# Compile step (auto-creates the mirrored build/ subdirectories) ---------------
$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	@echo "  CXX    $<"
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Run the program with optional ARGS ------------------------------------------
run: $(TARGET)
	./$(TARGET) $(ARGS)

# Housekeeping -----------------------------------------------------------------
clean:
	@echo "  CLEAN  build artifacts"
	@rm -rf $(BUILD_DIR) $(TARGET)

distclean: clean
	@rm -f $(TARGET).dSYM 2>/dev/null || true
	@rm -rf $(TARGET).dSYM

# Print the resolved configuration (handy for debugging the build itself) -----
info:
	@echo "CXX        = $(CXX)"
	@echo "ARCH       = $(ARCH)   ARCHFLAGS = $(ARCHFLAGS)"
	@echo "OPT        = $(OPT)"
	@echo "CXXFLAGS   = $(CXXFLAGS)"
	@echo "LDFLAGS    = $(LDFLAGS)"
	@echo "sources    = $(words $(SRCS)) files"

# ---- Auto-generated header dependencies -------------------------------------
-include $(DEPS)
