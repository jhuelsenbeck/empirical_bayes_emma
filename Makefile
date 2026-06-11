# =============================================================================
#  Bayesian Inference of Phylogeny using Profile Likelihoods
#  Top-level Makefile
#
#  Builds on macOS (Apple Silicon arm64 or Intel x86_64) and other Unix systems.
#  Place this file in the directory that contains small_tree_src/ and shared_src/.
#
#  Quick start:
#     make                         # optimized native build (host architecture)
#     make -j                      # same, parallel
#     make debug                   # -O0 -g with assertions
#     make release                 # -O3 -DNDEBUG
#     make ARCH=arm64              # force Apple Silicon build on macOS
#     make ARCH=x86_64             # force Intel build on macOS
#     make universal               # fat binary: arm64 + x86_64, macOS only
#     make clean
#     make run ARGS="-i data.nex -o out"
#
#  Eigen/Spectra notes:
#     Eigen and Spectra are header-only libraries. This Makefile searches the
#     common Homebrew locations automatically:
#        Apple Silicon Homebrew: /opt/homebrew/include
#        Intel Homebrew:         /usr/local/include
#
#     You can override paths explicitly, e.g.
#        make EIGEN_INC=/opt/homebrew/include/eigen3 SPECTRA_INC=/opt/homebrew/include
#
#     If Eigen was installed by Homebrew, the correct include directory is
#     usually /opt/homebrew/include/eigen3 or /usr/local/include/eigen3.
#     If Spectra was installed by Homebrew or copied manually, the include
#     directory should be the directory containing the Spectra/ folder.
#
#  Universal binary notes:
#     make universal uses Apple clang++ and -arch arm64 -arch x86_64.
#     Since Eigen and Spectra are header-only, no separate arm64/x86_64 library
#     linking is needed for them.
# =============================================================================

# ---- What we are building ---------------------------------------------------
TARGET     := small_tree

# ---- Project layout ---------------------------------------------------------
SMALL_DIR  := small_tree_src
SHARED_DIR := shared_src
NCL_DIR    := $(SHARED_DIR)/ncl
BUILD_DIR  := build

# ---- Toolchain --------------------------------------------------------------
CXX        ?= clang++
UNAME_S    := $(shell uname -s 2>/dev/null || echo unknown)

# ---- User-tunable knobs -----------------------------------------------------
OPT            ?= -O2                  # optimization level for the default build
ARCH           ?= native               # native | arm64 | x86_64 | universal
EXTRA_CXXFLAGS ?=                      # appended to compile flags
EXTRA_LDFLAGS  ?=                      # appended to link flags
ARGS           ?=                      # arguments passed by `make run`

# Optional user overrides for Eigen/Spectra include locations.
# EIGEN_INC should be the directory containing Eigen/Core.
# SPECTRA_INC should be the directory containing Spectra/GenEigsSolver.h.
EIGEN_INC      ?=
SPECTRA_INC    ?=

# Strip inline-comment whitespace from simple variables.
ARCH       := $(strip $(ARCH))
OPT        := $(strip $(OPT))
EIGEN_INC  := $(strip $(EIGEN_INC))
SPECTRA_INC:= $(strip $(SPECTRA_INC))

# ---- Fixed compiler settings ------------------------------------------------
STD        := -std=c++17
PROJECT_INCLUDES := -I$(SMALL_DIR) -I$(SHARED_DIR) -I$(NCL_DIR)

# ---- Eigen and Spectra include paths ----------------------------------------
# Common locations. Keeping both Apple Silicon and Intel Homebrew include roots
# here makes the same Makefile work on both kinds of Macs. These are harmless if
# the directories do not exist.
HOMEBREW_INC_DIRS := /opt/homebrew/include /usr/local/include
EIGEN_CANDIDATES  := /opt/homebrew/include/eigen3 /usr/local/include/eigen3 \
                     /opt/local/include/eigen3 /usr/include/eigen3
SPECTRA_CANDIDATES:= /opt/homebrew/include /usr/local/include \
                     /opt/local/include /usr/include

ifneq ($(EIGEN_INC),)
  EIGEN_FLAGS := -I$(EIGEN_INC)
else
  EIGEN_FLAGS := $(addprefix -I,$(EIGEN_CANDIDATES))
endif

ifneq ($(SPECTRA_INC),)
  SPECTRA_FLAGS := -I$(SPECTRA_INC)
else
  SPECTRA_FLAGS := $(addprefix -I,$(SPECTRA_CANDIDATES))
endif

# Also include Homebrew roots directly. This helps if Spectra is installed as
# /opt/homebrew/include/Spectra/... while Eigen is under include/eigen3.
BREW_FLAGS := $(addprefix -I,$(HOMEBREW_INC_DIRS))

INCLUDES := $(PROJECT_INCLUDES) $(EIGEN_FLAGS) $(SPECTRA_FLAGS) $(BREW_FLAGS)

# Keep the build readable: -Wall for our own code, silence the noisiest
# diagnostics coming out of the third-party NCL/Eigen/Spectra headers.
WARN       := -Wall \
              -Wno-sign-compare \
              -Wno-unused-variable \
              -Wno-unused-parameter \
              -Wno-deprecated-declarations \
              -Wno-unused-private-field

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
  ifneq ($(UNAME_S),Darwin)
    $(error ARCH=universal is supported only on macOS/Darwin)
  endif
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
# EIGEN_NO_DEBUG removes Eigen runtime checks in optimized builds when NDEBUG is
# also defined by `make release`.
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
.PHONY: all debug release universal run clean distclean info check-includes

all: $(TARGET)

# Convenience profiles ---------------------------------------------------------
debug:   OPT := -O0 -g
debug:   EXTRA_CXXFLAGS += -DDEBUG -fno-omit-frame-pointer
debug:   $(TARGET)

release: OPT := -O3
release: EXTRA_CXXFLAGS += -DNDEBUG -DEIGEN_NO_DEBUG
release: $(TARGET)

# Fat binary for distribution on both Apple Silicon and Intel Macs.
universal:
	$(MAKE) ARCH=universal release

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

# Quick compile-only checks for Eigen and Spectra include visibility -----------
check-includes:
	@echo '#include <Eigen/Core>' > /tmp/check_eigen_spectra.cpp
	@echo '#include <Spectra/GenEigsSolver.h>' >> /tmp/check_eigen_spectra.cpp
	@echo 'int main(){return 0;}' >> /tmp/check_eigen_spectra.cpp
	$(CXX) $(STD) $(ARCHFLAGS) $(INCLUDES) /tmp/check_eigen_spectra.cpp -c -o /tmp/check_eigen_spectra.o
	@rm -f /tmp/check_eigen_spectra.cpp /tmp/check_eigen_spectra.o
	@echo "Eigen and Spectra headers found."

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
	@echo "EIGEN_INC  = $(if $(EIGEN_INC),$(EIGEN_INC),auto candidates)"
	@echo "SPECTRA_INC= $(if $(SPECTRA_INC),$(SPECTRA_INC),auto candidates)"
	@echo "INCLUDES   = $(INCLUDES)"
	@echo "CXXFLAGS   = $(CXXFLAGS)"
	@echo "LDFLAGS    = $(LDFLAGS)"
	@echo "sources    = $(words $(SRCS)) files"

# ---- Auto-generated header dependencies -------------------------------------
-include $(DEPS)
