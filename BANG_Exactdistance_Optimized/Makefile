# =============================================================================
# BANG Exact Distance - Makefile
# =============================================================================

# Compiler
NVCC = nvcc

# Architecture (sm_80 for A100/Quadro)
ARCH = -arch=sm_80

# Compiler flags
CFLAGS = -std=c++14 -Xcompiler -fopenmp -I../../utils -O3

# Source files
SOURCES = main.cu parANN.cu

# Output executable
TARGET = bang_exact

# Default target
all: $(TARGET)

$(TARGET): $(SOURCES) parANN.h fused_kernel.cu
	$(NVCC) $(ARCH) $(SOURCES) $(CFLAGS) -o $(TARGET)

# Clean build artifacts
clean:
	rm -f $(TARGET)

# Rebuild from scratch
rebuild: clean all

# Run SIFT100M test
run-sift:
	./run_sift100m.sh

# Run DEEP100M test
run-deep:
	./run_deep100m.sh

# Help
help:
	@echo "BANG Exact Distance - Build Targets"
	@echo "===================================="
	@echo "  make          - Build the executable"
	@echo "  make clean    - Remove executable"
	@echo "  make rebuild  - Clean and rebuild"
	@echo "  make run-sift - Run SIFT100M test"
	@echo "  make run-deep - Run DEEP100M test"
	@echo "  make help     - Show this help"
	@echo ""
	@echo "Before building, ensure:"
	@echo "  1. parANN.h has correct dataset define (SIFT100M or DEEP100M)"
	@echo "  2. L value is set appropriately"
	@echo "  3. fused_kernel.cu is in the same directory"

.PHONY: all clean rebuild run-sift run-deep help
