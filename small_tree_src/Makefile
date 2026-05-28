# Makefile for Bayesian Inference of Phylogeny using Profile Likelihoods
# Supports macOS Intel and Apple Silicon (universal binary)

# PROJECT SETTINGS
CXX = clang++
CXXFLAGS = -std=c++17 -Wall -O3 -g
LDFLAGS =
TARGET = phylo-bayes

# Create a list of source files (add new .cpp files as needed)
SOURCES = \
    main.cpp \
    Alignment.cpp \
    BitSetFactory.cpp \
    ExhaustiveSearch.cpp \
    Mcmc.cpp \
    RandomVariable.cpp \
    Threads.cpp \
    TreeCache.cpp \
    TreeLikelihoods.cpp \
    TreeNeighborGenerator.cpp \
    TreeNeighbors.cpp \
    TreeSpace.cpp \
    UserSettings.cpp \
    Tree.cpp \
    TreePartitions.cpp 

# Object files (translates .cpp to .o)
OBJECTS = $(SOURCES:.cpp=.o)

# Universal binary settings for macOS (Apple Silicon + Intel)
UNIVERSAL_FLAGS = -arch arm64 -arch x86_64

# Default build
all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(UNIVERSAL_FLAGS) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Pattern rule for compiling .cpp to .o
%.o: %.cpp
	$(CXX) $(UNIVERSAL_FLAGS) $(CXXFLAGS) -c $< -o $@

# Clean rule to remove generated files
clean:
	rm -f $(OBJECTS) $(TARGET)

.PHONY: all clean
