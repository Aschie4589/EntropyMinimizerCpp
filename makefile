# Directories
SRC_DIR = src
OBJ_DIR = build
INCLUDE_DIR = include

INCLUDES = -I./$(INCLUDE_DIR)
CXXFLAGS = -std=c++17


##### SETUP FLAGS FOR BUILD SYSTEM #####

# Example use: "make PLATFORM=apple LAPACK=accelerate"

# Default platform. Used to determine which libraries to link against and which compiler to use (g++ or icpcx)
PLATFORM ?= apple # Options: "apple", "linux"
# What linear algebra package to use
ifeq ($(origin LAPACK), undefined)
    LAPACK = openblas# Use this to specify the default linear algebra package to use. Options: "mkl", "openblas", "accelerate". NO TRAILING SPACES!
endif 

# Default architecture
ifeq ($(PLATFORM), apple)
    ARCH ?= arm64
else ifeq ($(PLATFORM), linux)
    ARCH ?= x86_64
endif

# Flags based on PLATFORM, to include the correct libraries etc.
ifeq ($(PLATFORM), apple)
    CXX = g++
    CXXFLAGS += -DTARGET_APPLE	 # This makes sure that "TARGET_APPLE" is defined in the code as a preprocessor macro
    CXXFLAGS += -fdiagnostics-color=always # Color diagnostic messages
    CXXFLAGS += -arch $(ARCH) # Compile for arm64
    # Other stuff to add for apple

else ifeq ($(PLATFORM), linux)
    CXX = icpx # Ehm is this right? Use intel compiler
    CXXFLAGS += -DTARGET_LINUX # This makes sure that "TARGET_LINUX" is defined in the code as a preprocessor macro
    CXXFLAGS += -qdiagnostics-color=always # Color diagnostic message
    CXXFLAGS += -arch $(ARCH)
    # Other stuff to add for linux

endif


# Add flags for linear algebra libraries
ifeq ($(LAPACK), accelerate)
    # Add the headers for Accelerate framework
    CXXFLAGS += -DLAPACK_ACCELERATE
    CXXFLAGS += -DACCELERATE_NEW_LAPACK
    INCLUDES += -I/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Headers
    LIBDIRFLAGS = -L/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework
    LIBS = -framework Accelerate
    LIBS += -O3 -ftree-vectorize -march=native

else ifeq ($(LAPACK), mkl)
    # Add headers for Intel MKL
    CXXFLAGS += -DLAPACK_MKL
    INCLUDES += -I/opt/intel/oneapi/mkl/latest/include   #Add Intel mkl include path
    LIBDIRFLAGS = -L/opt/intel/oneapi/mkl/latest/lib -L/opt/intel/oneapi/compiler/latest/mac/compiler/lib
    LIBS = -lmkl_rt -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -liomp5 -lpthread -lm -ldl -Wl,-rpath,/opt/intel/oneapi/mkl/latest/lib

else ifeq ($(LAPACK),openblas)
    # Add headers for OpenBLAS
    CXXFLAGS += -DLAPACK_OPENBLAS
    INCLUDES += -I/opt/homebrew/opt/lapack/include -I/opt/homebrew/opt/openblas/include
    LIBDIRFLAGS = -L/opt/homebrew/opt/lapack/lib -L/opt/homebrew/opt/openblas/lib
    LIBS = -lopenblas -lblas -llapack -llapacke
else
    $(error "Invalid LAPACK setting: $(LAPACK). Please choose accelerate, mkl, or openblas.")

endif

# Source and object files
SOURCES := $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS := $(SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

# Target
TARGET = moe  # Name of the final executable

# Default build rule
all: $(TARGET)

# Create object directory if it doesn't exist
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Link the target
$(TARGET): $(OBJ_DIR) $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $(OBJECTS) $(LIBDIRFLAGS) $(LIBS)

# Compile each .cpp file into .o in the obj directory
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean up
clean:
	rm -rf $(OBJ_DIR) $(TARGET)