# ---------------- Project ----------------
EXAMPLE := transposePerf

# Add more sources if needed
SRCS := main.cc

# ---------------- ROCm / HIP setup ----------------
GPU_RUNTIME := HIP
# Support multiple architectures (space-separated), e.g.: "gfx942 gfx90a"
GPU_ARCHS ?= gfx942

# ROCm installation prefix (override with ROCM_PATH if set)
ROCM_INSTALL_DIR := $(if $(ROCM_PATH),$(ROCM_PATH),/opt/rocm)
HIPCXX ?= $(ROCM_INSTALL_DIR)/bin/hipcc

# ---------------- Build type ----------------
# Options: Release / RelWithDebInfo / Debug / Asan
BUILD ?= Release

# ---------------- Directories ----------------
BUILD_DIR := build
OBJ_DIR   := $(BUILD_DIR)/obj

# ---------------- Common flags ----------------
CXX_STD    := c++17
WARNFLAGS  := -Wall -Wextra -Wno-unused-parameter
# rpath helps runtime linkers locate ROCm shared libs without setting LD_LIBRARY_PATH
RPATHFLAGS := -Wl,-rpath,$(ROCM_INSTALL_DIR)/lib -Wl,-rpath,$(ROCM_INSTALL_DIR)/lib64

# hipcc/clang common flags; ensure temp files are saved
HIP_COMMON := -std=$(CXX_STD) $(WARNFLAGS)

# Expand multiple --offload-arch flags
OFFLOAD_ARCH_FLAGS := $(foreach arch,$(GPU_ARCHS),--offload-arch=$(arch))

# Optimization/Debug per build type
ifeq ($(BUILD),Release)
  OPTFLAGS := -O3 
else ifeq ($(BUILD),Debug)
  OPTFLAGS := -O0 -g3 
  HIP_COMMON += --save-temps
else
  $(error Unknown BUILD=$(BUILD). Use Release/RelWithDebInfo/Debug/Asan)
endif

# ---------------- Platform selection (HIP only) ----------------
ifeq ($(GPU_RUNTIME),HIP)
  COMPILER := $(HIPCXX)
  CPPDEFS  := -D__HIP_PLATFORM_AMD__
else
  $(error GPU_RUNTIME must be HIP (ROCm). CUDA is not supported here)
endif

# ---------------- Includes / Libs ----------------
INCLUDES :=
LIBDIRS  := -L$(ROCM_INSTALL_DIR)/lib -L$(ROCM_INSTALL_DIR)/lib64
LDLIBS   :=

# Aggregate flags
CXXFLAGS := $(HIP_COMMON) $(OFFLOAD_ARCH_FLAGS) $(OPTFLAGS) $(DBGFLAGS) 
CPPFLAGS := $(INCLUDES) $(CPPDEFS)
LDFLAGS  := $(LIBDIRS) $(RPATHFLAGS)

# ---------------- Objects / Deps ----------------
OBJS := $(patsubst %.cc,$(OBJ_DIR)/%.o,$(SRCS))
DEPS := $(OBJS:.o=.d)

# ---------------- Rules ----------------
.PHONY: all clean run profile print-config

all: $(EXAMPLE)

$(EXAMPLE): $(OBJS)
	@mkdir -p $(dir $@)
	$(COMPILER) $(CXXFLAGS) $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)

# Generic rule for .cc -> .o
$(OBJ_DIR)/%.o: %.cc
	@mkdir -p $(dir $@)
	$(COMPILER) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

# Quick run
run: $(EXAMPLE)
	./$(EXAMPLE)

# Profiling with rocprofiler-compute
profile: $(EXAMPLE)
	rocprofiler-compute profile -n $(EXAMPLE) --no-roof -- ./$(EXAMPLE)

# Cleanup
clean:
	$(RM) -r $(BUILD_DIR) $(EXAMPLE) *.link *.s *.ii *.bc *.o *.d *.hipfb *.out *.hipi *.txt

# Print current configuration
print-config:
	@echo "BUILD=$(BUILD)"
	@echo "GPU_ARCHS=$(GPU_ARCHS)"
	@echo "ROCM_INSTALL_DIR=$(ROCM_INSTALL_DIR)"
	@echo "COMPILER=$(COMPILER)"
	@echo "CXXFLAGS=$(CXXFLAGS)"
	@echo "LDFLAGS=$(LDFLAGS)"

# Auto-included dependency files
-include $(DEPS)
