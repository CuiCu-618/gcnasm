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

# 两个源文件：baseline 和优化版
SRC_BASE   = main.hip
SRC_AIR    = main_air.hip

OBJ_BASE   = $(SRC_BASE:.hip=.o)
OBJ_AIR    = $(SRC_AIR:.hip=.o)

TARGET_BASE = test_topk_rocm_base
TARGET_AIR  = test_topk_rocm_air

.PHONY: all clean run_base run_air profile_base profile_air

# 默认：两个都编译
all: $(TARGET_BASE) $(TARGET_AIR)

# 链接生成 baseline 程序
$(TARGET_BASE): $(OBJ_BASE)
	$(HIPCC) $(CXXFLAGS) $^ -o $@

# 链接生成优化版程序
$(TARGET_AIR): $(OBJ_AIR)
	$(HIPCC) $(CXXFLAGS) $^ -o $@

# 规则：编译 .hip -> .o
%.o: %.hip
	$(HIPCC) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# 运行 baseline
run_base: $(TARGET_BASE)
	./$(TARGET_BASE)

# 运行优化版
run_air: $(TARGET_AIR)
	./$(TARGET_AIR)

# 默认 run 就跑 baseline，也可以改成 run_air
run: run_base

# profile baseline
profile_base: $(TARGET_BASE)
	rocprof-compute profile -n $(TARGET_BASE) --no-roof -- ./$(TARGET_BASE)

# profile 优化版
profile_air: $(TARGET_AIR)
	rocprof-compute profile -n $(TARGET_AIR) --no-roof -- ./$(TARGET_AIR)

clean:
	rm -f $(OBJ_BASE) $(OBJ_AIR) $(TARGET_BASE) $(TARGET_AIR)
