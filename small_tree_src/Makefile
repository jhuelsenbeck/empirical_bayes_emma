# Makefile — Bayesian Inference of Phylogeny using Profile Likelihoods
# Layout: project .cpp/.hpp in this directory; Nexus Class Library sources in ./ncl

CXX      := clang++          # g++ works too; on macOS clang++ is the default
TARGET   := epic

# ---------------------------------------------------------------------------
# Third-party header paths — edit for your machine.
#   EIGEN_INC   : directory containing the "Eigen/" header folder
#   SPECTRA_INC : directory containing the "Spectra/" header folder
# `brew --prefix` fills these in on a Homebrew setup; override on the command
# line if your headers live elsewhere, e.g.  make EIGEN_INC=/usr/include/eigen3
# ---------------------------------------------------------------------------
BREW_PREFIX ?= $(shell brew --prefix 2>/dev/null)
EIGEN_INC   ?= $(BREW_PREFIX)/include/eigen3
SPECTRA_INC ?= $(BREW_PREFIX)/include

# ---------------------------------------------------------------------------
# Flags
#   -std=c++20   : std::unordered_set::contains and other C++20 conveniences
#   -O3 -DNDEBUG : optimization on, Eigen internal asserts off (needed for speed)
#   -flto        : link-time optimization
#   -march=native: tune for this machine's CPU (drop it for a portable binary)
#   -pthread     : the ThreadPool
#   -Incl        : find the NCL headers (use -I. instead if your code includes
#                  them as "ncl/nxsblock.h" rather than "nxsblock.h")
#   -DNO_ACCELERATE : build the pure-CPU SpMV path (the Apple beta-SDK header
#                     clash we hit; remove once Accelerate builds cleanly for you)
# ---------------------------------------------------------------------------
OPT      := -O3 -DNDEBUG -flto -march=native

# Your code: full warnings.
CXXFLAGS := -std=c++20 -Wall $(OPT) -pthread -DNO_ACCELERATE \
            -I$(EIGEN_INC) -I$(SPECTRA_INC) -Incl

# Third-party NCL: same everything, but warnings off (do not edit vendored code
# to silence cosmetic warnings — suppress them here instead).
NCL_CXXFLAGS := $(filter-out -Wall,$(CXXFLAGS)) -w

LDFLAGS  := -flto -pthread

# ---------------------------------------------------------------------------
# Sources: every .cpp in this directory plus every .cpp under ncl/, discovered
# automatically so the list can't go stale.
#
# WARNING: MarkovChainAnalyzer_multiple_try.cpp is in this directory and is NOT
# part of the build — the wildcard would compile it and you'd get duplicate-symbol
# link errors. It is excluded below; keep any other scratch/backup .cpp out of
# these directories or exclude them the same way.
# ---------------------------------------------------------------------------
EXCLUDE  := MarkovChainAnalyzer_multiple_try.cpp
SOURCES  := $(filter-out $(EXCLUDE),$(wildcard *.cpp)) $(wildcard ncl/*.cpp)
OBJECTS  := $(SOURCES:.cpp=.o)
DEPS     := $(OBJECTS:.o=.d)

# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------
all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# NCL objects: compiled with warnings suppressed. This rule is more specific than
# the generic one below, so it wins for anything under ncl/.
ncl/%.o: ncl/%.cpp
	$(CXX) $(NCL_CXXFLAGS) -MMD -MP -c $< -o $@

# Your objects: compiled with full warnings.
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

-include $(DEPS)

clean:
	rm -f $(OBJECTS) $(DEPS) $(TARGET)

.PHONY: all clean