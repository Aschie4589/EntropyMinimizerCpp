# Directories
SRC_DIR = src
OBJ_DIR = build
INCLUDE_DIR = include


##### COMMON COMPILER FLAGS #####
# Always use g++ as compiler. 
#For MacOS, this will just be a wrapper around clang, if XCode tools are installed. Otherwise, g++ is needed and can be installed via Homebrew.
CXX = g++
CXXFLAGS = -std=c++17 
CXXFLAGS += -march=native# This last flag is to enable all CPU-specific optimizations on the host machine. ONLY LOCAL COMPILE!
CXXFLAGS += -O3 -ftree-vectorize

##### SPECIFIC FLAGS #####
# Example use: "make PLATFORM=apple LAPACK=accelerate"

# Default platform. Used to determine which libraries to link against, and have specific flags. For now only diagnostics color is specific.
ifeq ($(origin PLATFORM), undefined)
    PLATFORM = apple# Use this to specify the default platform. Options: "apple", "linux". NO TRAILING SPACES!
endif 
# What linear algebra package to use
ifeq ($(origin LAPACK), undefined)
    LAPACK = openblas# Use this to specify the default linear algebra package to use. Options: "mkl", "openblas", "accelerate". NO TRAILING SPACES!
endif 


# Flags based on PLATFORM, to include the correct libraries etc.
ifeq ($(PLATFORM), apple)
    CXXFLAGS += -fdiagnostics-color=always # Color diagnostic messages
    # Other stuff to add for apple d
else ifeq ($(PLATFORM), linux)
    CXXFLAGS += -fdiagnostics-color=always # Color diagnostic message
    # Other stuff to add for linux d
endif


INCLUDES = -I./$(INCLUDE_DIR)

# Add flags for linear algebra libraries. Do so only if not cleaning
ifeq ($(MAKECMDGOALS), clean)
    # Do nothing
else ifeq ($(LAPACK), accelerate)
    # Find the accelerate framework location
    ACCELERATE_PATH=$(xcrun --show-sdk-path)/System/Library/Frameworks/Accelerate.framework
    $(info Debug: Accelerate framework path: $(ACCELERATE_PATH))

    # Add the headers for Accelerate framework. No need to specify flags for the library, as it is linked by default.
    CXXFLAGS += -DLAPACK_ACCELERATE
    CXXFLAGS += -DACCELERATE_NEW_LAPACK
    LIBS = -framework Accelerate

else ifeq ($(LAPACK), openblas)

    # Attempt to locate OpenBLAS and LAPACK
    OPENBLAS_LIB := $(shell find $(CONDA_PREFIX) /opt/homebrew /usr /usr/local  -name "libopenblas.*" 2>/dev/null | head -n 1 | xargs dirname)
    $(info Debug: OpenBLAS library path: $(OPENBLAS_LIB))
    LAPACK_LIB := $(shell find $(CONDA_PREFIX) /opt/homebrew /usr /usr/local -name "liblapack.*" 2>/dev/null | head -n 1 | xargs dirname)
    $(info Debug: LAPACK library path: $(LAPACK_LIB))
    OPENBLAS_INC := $(shell find $(CONDA_PREFIX) /opt/homebrew /usr /usr/local -name "cblas.h" 2>/dev/null | head -n 1 | xargs dirname)
    $(info Debug: OpenBLAS include path: $(OPENBLAS_INC))
    LAPACK_INC := $(shell find $(CONDA_PREFIX) /opt/homebrew /usr /usr/local -name "lapacke.h" 2>/dev/null | head -n 1 | xargs dirname)
    $(info Debug: LAPACK include path: $(LAPACK_INC))

    # Add flags if both libraries are found
    ifeq ($(OPENBLAS_LIB),)
        $(warning Warning: OpenBLAS library not found. Compilation may fail.)
    else
        CXXFLAGS += -DLAPACK_OPENBLAS
        INCLUDES += -I$(OPENBLAS_INC)
        LIBDIRFLAGS += -L$(OPENBLAS_LIB) 
        $(info Debug: OpenBLAS flags added: -I$(OPENBLAS_INC) -L$(OPENBLAS_LIB))
    endif

    ifeq ($(LAPACK_LIB),)
        $(warning Warning: LAPACK library not found. Compilation may fail.)
    else
        INCLUDES += -I$(LAPACK_INC)
        LIBDIRFLAGS += -L$(LAPACK_LIB)
        $(info Debug: LAPACK flags added: -I$(LAPACK_INC) -L$(LAPACK_LIB))
    endif
    # Add headers for OpenBLAS
    CXXFLAGS += -DLAPACK_OPENBLAS
    LIBS += -lopenblas -lpthread -lrt -ldl
    
else
    $(error "Invalid LAPACK setting: $(LAPACK). Please choose accelerate, mkl, aocllibm or openblas.")

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