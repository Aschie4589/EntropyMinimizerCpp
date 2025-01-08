# Compiler and flags
CXX = /usr/bin/g++
CXXFLAGS = -std=c++17 -fdiagnostics-color=always
LDFLAGS = -L/opt/homebrew/opt/lapack/lib -L/opt/homebrew/opt/openblas/lib
INCLUDES = -I/opt/homebrew/opt/lapack/include -I/opt/homebrew/opt/openblas/include -I./include
LIBS = -lopenblas -lblas -llapack -llapacke

# Directories
SRC_DIR = src
OBJ_DIR = build
INCLUDE_DIR = include

# Source and object files
SOURCES := $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS := $(SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

# Target
TARGET = out  # Name of the final executable

# Default build rule
all: $(TARGET)

# Create object directory if it doesn't exist
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Link the target
$(TARGET): $(OBJ_DIR) $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $(OBJECTS) $(LDFLAGS) $(LIBS)

# Compile each .cpp file into .o in the obj directory
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean up
clean:
	rm -rf $(OBJ_DIR) $(TARGET)
