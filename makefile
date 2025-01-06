# Compiler and flags
CXX = /usr/bin/g++
CXXFLAGS = -std=c++17 -fdiagnostics-color=always
LDFLAGS = -L/opt/homebrew/opt/lapack/lib
INCLUDES = -I/opt/homebrew/opt/lapack/include
LIBS = -llapack -llapacke

# Source and target
SOURCES = main.cpp Minimizer.cpp
BUILD_DIR = build
OBJECTS = $(SOURCES:%.cpp=$(BUILD_DIR)/%.o)
TARGET = out  # Name of the final executable

# Default build rule
all: $(TARGET)

# Create build directory if it doesn't exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Link the target
$(TARGET): $(BUILD_DIR) $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $(OBJECTS) $(LDFLAGS) $(LIBS)

# Compile each .cpp file into .o in the build directory
$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean up
clean:
	rm -rf $(BUILD_DIR) $(TARGET)
