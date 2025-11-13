# 使用的编译器：ROCm 下的 hipcc
HIPCC      ?= hipcc

# GPU 架构，根据你的卡改：
#   - MI300A: gfx942
#   - MI250:  gfx90a
#   - 其它卡：查看 rocminfo 输出
GPU_ARCH   ?= gfx942

# 通用编译选项
CXXFLAGS   ?= -O3 -std=c++17
CXXFLAGS   += --offload-arch=$(GPU_ARCH)

# 如果需要额外 include 目录，在这里加：
INCLUDES   ?=

# 源文件列表：
# 如果你的 kernel 源文件是 .hip，可以改成：
SRC        = main.hip

OBJ        = $(SRC:.cpp=.o)
# 对应 .hip 源文件的情况，也转成 .o
OBJ        := $(OBJ:.hip=.o)

TARGET     = test_topk_rocm

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJ)
	$(HIPCC) $(CXXFLAGS) $^ -o $@

# 规则：编译 .cpp -> .o
%.o: %.cpp
	$(HIPCC) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# 规则：编译 .hip -> .o（如果你有 .hip 源文件）
%.o: %.hip
	$(HIPCC) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

profile: $(TARGET)
	rocprof-compute profile -n $(TARGET) --no-roof -- ./$(TARGET)

clean:
	rm -f $(OBJ) $(TARGET)
