# 在命令行模式下查看CUDA程序性能

要在命令行模式下查看CUDA程序的性能，特别是每条指令的耗时，你可以使用以下几种方法：

## 1. 使用nvprof (NVIDIA Profiler)

nvprof是NVIDIA提供的命令行性能分析工具，可以详细分析CUDA内核和API调用。

```bash
nvprof ./your_cuda_program
```

### 常用选项：
- `--print-gpu-trace`：显示GPU活动的时间线
- `--metrics all`：显示所有可用指标
- `--events all`：记录所有事件
- `--analysis-metrics`：提供更详细的分析

例如，查看每条指令的耗时：
```bash
nvprof --print-gpu-trace --metrics inst_executed ./your_cuda_program
```

## 2. 使用Nsight Compute (ncu)

Nsight Compute是更现代的CUDA内核分析工具，提供指令级分析：

```bash
ncu --set full ./your_cuda_program
```

```bash
sudo apt install nvidia-modprobe
sudo usermod -a -G video $USER
```

reboot

### 常用选项：
- `--section SpeedOfLight`：显示计算和内存吞吐量
- `--section InstructionStats`：显示指令统计
- `--section SchedulerStats`：显示调度器统计
- `--kernel-regex pattern`：只分析匹配的内核

例如，分析特定内核的指令耗时：
```bash
ncu --kernel-name myKernel --section InstructionStats ./your_cuda_program
```

## 3. 使用Nsight Systems (nsys)

对于系统级分析，包括CPU和GPU活动：

```bash
nsys profile ./your_cuda_program
```

## 4. 在代码中添加时间测量

你可以在代码中手动添加时间测量：

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
// 要测量的CUDA代码
myKernel<<<grid, block>>>(...);
cudaEventRecord(stop);

cudaEventSynchronize(stop);
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);

printf("Kernel time: %f ms\n", milliseconds);
```

## 5. 查看PTX/SASS指令

要查看实际的指令级性能，你需要分析PTX或SASS代码：

```bash
cuobjdump -sass your_cuda_program
```

然后结合Nsight Compute的分析结果来理解每条指令的性能。

## 注意事项

1. 确保你的程序是使用调试符号编译的（`-G`或`-lineinfo`选项）
2. 分析会影响程序性能，因此结果仅供参考
3. 对于指令级分析，Nsight Compute通常比nvprof提供更详细的信息


# CUDA性能分析工具优缺点对比

以下是各种CUDA性能分析方法的优缺点比较，帮助你根据需求选择合适的工具：

## 1. nvprof (NVIDIA Profiler)

**优点**：
- 简单易用，基本性能数据一目了然
- 支持所有CUDA版本（虽然新版本已弃用，但仍广泛使用）
- 可以收集各种硬件计数器
- 提供API调用和内核执行的时间线

**缺点**：
- 在较新CUDA版本中已标记为弃用（推荐使用Nsight工具）
- 指令级分析不够详细
- 对Ampere架构及更新的GPU支持有限
- 用户界面相对简单

## 2. Nsight Compute (ncu)

**优点**：
- 提供最详细的指令级性能分析
- 支持最新的GPU架构特性
- 可以分析单个内核的SASS指令性能
- 提供丰富的性能指标和瓶颈分析
- 支持交互式分析模式

**缺点**：
- 学习曲线较陡峭
- 详细分析会显著增加程序运行时间
- 输出信息量大，需要专业知识解读
- 主要关注内核性能，系统级分析较弱

## 3. Nsight Systems (nsys)

**优点**：
- 提供完整的系统级视图（CPU+GPU+内存等）
- 时间线可视化非常强大
- 开销相对较小，适合分析整个应用
- 支持多进程和多GPU分析
- 可以识别CPU-GPU同步问题

**缺点**：
- 不提供指令级分析细节
- 对于纯内核优化帮助有限
- 生成的报告文件可能很大
- 需要GUI（nsight-sys）获得最佳体验

## 4. 代码内手动计时

**优点**：
- 精确控制测量范围
- 无需额外工具依赖
- 可以集成到自动化测试中
- 适合生产环境监控

**缺点**：
- 只提供粗粒度时间测量
- 无法获取硬件性能计数器
- 需要修改代码
- 测量本身可能引入开销

## 5. PTX/SASS指令分析 (cuobjdump)

**优点**：
- 查看实际执行的机器指令
- 理解编译器优化结果
- 帮助进行极端优化

**缺点**：
- 需要深厚的GPU架构知识
- 无法直接关联到源代码
- 不提供性能数据
- 分析过程繁琐

## 工具选择建议

| 分析需求 | 推荐工具 |
|---------|---------|
| 快速获取基本性能数据 | nvprof |
| 深入分析内核瓶颈 | Nsight Compute |
| 系统级性能分析 | Nsight Systems |
| 长期性能监控 | 代码内计时 |
| 极端优化/指令分析 | cuobjdump + Nsight Compute |

对于你的具体需求"查看每一条指令的耗时"，**Nsight Compute (ncu)** 是最合适的选择，它能提供最详细的指令级性能分析，包括指令吞吐量、停顿周期等信息。

```bash
sudo vim /etc/modprobe.d/nvidia-profiler.conf
# options nvidia NVreg_RestrictProfilingToAdminUsers=0
sudo update-initramfs -u
sudo reboot
```
