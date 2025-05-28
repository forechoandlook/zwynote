import torch
import torch.nn as nn
import time
from collections import defaultdict
import csv
import os
from transformers import TrainerCallback
from torch.profiler import profile, ProfilerActivity
import os
from pathlib import Path
import logging

profile_log = logging.getLogger('profile_log')

class TorchProfileCallback(TrainerCallback):
    """用于Torch Profiling的Callback类"""

    def __init__(self,
                 output_dir="./profile_traces",
                 activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 with_stack=True,
                 with_modules=True,
                 with_flops=True,
                 record_shapes=True,
                 profile_memory=True,
                 warmup_steps=2,           # 预热步数
                 schedule_steps=100,       # 每隔多少步记录一次
                 max_profile_records=5,    # 最多记录多少次profile
                 memory_stats=True):       # 是否记录内存统计
        """
        初始化Profiler Callback

        Args:
            output_dir: 性能分析文件的输出目录
            activities: 要分析的活动类型
            with_stack: 是否记录调用栈信息
            with_modules: 是否记录模块信息
            with_flops: 是否记录FLOPS信息
            record_shapes: 是否记录张量形状信息
            profile_memory: 是否分析内存使用
            warmup_steps: 预热步数，在这些步骤中不会记录性能数据
            schedule_steps: 每隔多少步记录一次性能数据
            max_profile_records: 最多记录多少次profile数据
            memory_stats: 是否记录内存统计信息
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.profiler_kwargs = {
            "activities": activities,
            "with_stack": with_stack,
            "with_modules": with_modules,
            "with_flops": with_flops,
            "record_shapes": record_shapes,
            "profile_memory": profile_memory
        }
        self.profiler = None
        self.step_num = 0
        self.warmup_steps = warmup_steps
        self.schedule_steps = schedule_steps
        self.max_profile_records = max_profile_records
        self.profile_count = 0
        self.memory_stats = memory_stats
        self.peak_memory = 0

    def _should_profile(self):
        """判断当前步骤是否应该记录性能数据"""
        if self.step_num <= self.warmup_steps:
            return False
        if self.profile_count >= self.max_profile_records:
            return False
        return (self.step_num - self.warmup_steps) % self.schedule_steps == 0

    def _record_memory_stats(self):
        """记录当前的内存使用情况"""
        if not self.memory_stats or not torch.cuda.is_available():
            return {}

        current_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
        max_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
        self.peak_memory = max(self.peak_memory, max_memory)

        return {
            "current_memory_mb": current_memory,
            "max_memory_mb": max_memory,
            "peak_memory_mb": self.peak_memory
        }

    def on_train_begin(self, args, state, control, **kwargs):
        """训练开始时启动profiler"""
        print(f"性能分析配置:\n"
              f"- 预热步数: {self.warmup_steps}\n"
              f"- 采样间隔: {self.schedule_steps}步\n"
              f"- 最大记录数: {self.max_profile_records}")
        self.profiler = profile(**self.profiler_kwargs)
        self.profiler.start()

    def on_step_end(self, args, state, control, **kwargs):
        """每个训练步骤结束时记录数据"""
        if self.profiler:
            self.profiler.step()
            self.step_num += 1

            if self._should_profile():
                self.profile_count += 1
                trace_file = self.output_dir / f"trace_step_{self.step_num}.json"
                self.profiler.export_chrome_trace(str(trace_file))

                # 获取内存统计
                memory_stats = self._record_memory_stats()

                # 打印性能统计信息
                print(f"\nStep {self.step_num} Profiling Stats (Record {self.profile_count}/{self.max_profile_records}):")
                print(self.profiler.key_averages().table(
                    sort_by="cuda_time_total",
                    row_limit=10
                ))

                if memory_stats:
                    print("\nMemory Statistics:")
                    for key, value in memory_stats.items():
                        print(f"- {key}: {value:.2f} MB")

    def on_train_end(self, args, state, control, **kwargs):
        """训练结束时停止profiler并保存最终结果"""
        if self.profiler:
            final_trace = self.output_dir / "trace_final.json"
            self.profiler.export_chrome_trace(str(final_trace))

            memory_stats = self._record_memory_stats()

            print("\nFinal Profiling Stats:")
            print(self.profiler.key_averages().table(
                sort_by="cuda_time_total",
                row_limit=20
            ))

            if memory_stats:
                print("\nFinal Memory Statistics:")
                for key, value in memory_stats.items():
                    print(f"- {key}: {value:.2f} MB")

            self.profiler = None

def make_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def register_hook_for_function(func, save_name="func"):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Function {save_name} took {elapsed_time:.4f} seconds to execute.")
        return result
    return wrapper

def register_hook_for_Function(cls:torch.autograd.Function):
    cls.forward = register_hook_for_function(cls.forward, save_name=cls.__name__)
    cls.backward = register_hook_for_function(cls.backward, save_name=cls.__name__)
    return cls


global_func_record = {}

class BasicModule(nn.Module):
    def __init__(self, name, func):
        super(BasicModule, self).__init__()
        self.name = name
        self.func = func
    
    @classmethod
    def from_function(cls, func):
        global global_func_record
        if func.__name__ in global_func_record:
            return global_func_record[func.__name__]
        else:
            global_func_record[func.__name__] = cls(func.__name__, func)
            return global_func_record[func.__name__]

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)

def replace_fn_into_module(fn):
    return BasicModule.from_function(fn.__name__,fn)


common_functions_list = [ "add", "sub", "mul", "div", "matmul", "pow", "exp", "log", "sqrt", "relu", "sigmoid", "tanh"]
# 常见函数转为BasicModule
def replace_common_functions(fn_lists: list[str] | None = common_functions_list):
    for fn_name in fn_lists:
        fn = getattr(torch, fn_name)
        if fn is not None:
            fn = replace_fn_into_module(fn)
            profile_log.info(f"Replace function {fn.__name__} into BasicModule")
        else:
            profile_log.warning(f"Function {fn_name} not found")


def replace_user_functions(fn_lists):
    for fn in fn_lists:
        fn = replace_fn_into_module(fn)
        profile_log.info(f"Replace user function {fn.__name__} into BasicModule")

class TimeProfiler:
    def __init__(self, model):
        self.model = model
        self.forward_hooks = []
        self.backward_hooks = []
        self.time_stats = defaultdict(lambda: {
            'forward': 0.0,
            'backward': 0.0,
            'count': 0,
            'self_forward': 0.0,
            'self_backward': 0.0,
            'parent': None,
            'depth': 0,
            'max_forward': 0.0,   # 新增：最大forward耗时
            'max_backward': 0.0,   # 新增：最大backward耗时
            'min_forward': float('inf'),   # 新增：最小forward耗时
            'min_backward': float('inf'),   # 新增：最小backward耗时
        })
        self.start_time = {}
        self.backward_start_time = {}
        self.module_names = {}

        for name, module in model.named_modules():
            self.module_names[module] = name
            depth = len(name.split('.'))
            self.time_stats[name]['depth'] = depth

            if '.' in name:
                parent_name = '.'.join(name.split('.')[:-1])
                self.time_stats[name]['parent'] = parent_name

    def _forward_pre_hook(self, module, *args, **kwargs):
        module_name = self.module_names.get(module, module.__class__.__name__)
        self.start_time[module_name] = time.perf_counter()

    def _forward_hook(self, module, *args, **kwargs):
        make_sync()
        module_name = self.module_names.get(module, module.__class__.__name__)
        if module_name in self.start_time:
            elapsed = time.perf_counter() - self.start_time[module_name]

            # 记录自身时间
            self.time_stats[module_name]['self_forward'] += elapsed
            self.time_stats[module_name]['forward'] += elapsed
            self.time_stats[module_name]['count'] += 1

            # 记录最大forward耗时
            if elapsed > self.time_stats[module_name]['max_forward']:
                self.time_stats[module_name]['max_forward'] = elapsed

            # 记录最小forward耗时
            if elapsed < self.time_stats[module_name]['min_forward']:
                self.time_stats[module_name]['min_forward'] = elapsed

            # 向上累加时间到父模块
            current = module_name
            while self.time_stats[current]['parent']:
                parent = self.time_stats[current]['parent']
                self.time_stats[parent]['forward'] += elapsed
                current = parent

            del self.start_time[module_name]
        else:
            print(f"Warning: No forward start time found for module {module.__class__.__name__}")

    def _backward_pre_hook(self, module, *args):
        module_name = self.module_names.get(module, module.__class__.__name__)
        self.backward_start_time[module_name] = time.perf_counter()

    def _backward_hook(self, module, *args, **kwargs):
        make_sync()
        module_name = self.module_names.get(module, module.__class__.__name__)
        if module_name in self.backward_start_time:
            elapsed = time.perf_counter() - self.backward_start_time[module_name]

            self.time_stats[module_name]['self_backward'] += elapsed
            self.time_stats[module_name]['backward'] += elapsed
            self.time_stats[module_name]['count'] += 1

            # 记录最大backward耗时
            if elapsed > self.time_stats[module_name]['max_backward']:
                self.time_stats[module_name]['max_backward'] = elapsed

            # 记录最小backward耗时
            if elapsed < self.time_stats[module_name]['min_backward']:
                self.time_stats[module_name]['min_backward'] = elapsed

            # 向上累加时间到父模块
            current = module_name
            while self.time_stats[current]['parent']:
                parent = self.time_stats[current]['parent']
                self.time_stats[parent]['backward'] += elapsed
                current = parent

            del self.backward_start_time[module_name]
        else:
            print(f"Warning: No backward start time found for module {module_name}")

    def register_hooks(self):
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # 只注册叶子模块
                forward_pre_hook = module.register_forward_pre_hook(self._forward_pre_hook)
                forward_hook = module.register_forward_hook(self._forward_hook)
                backward_hook = module.register_full_backward_hook(self._backward_hook)
                backward_pre_hook = module.register_full_backward_pre_hook(self._backward_pre_hook)

                self.forward_hooks.append(forward_pre_hook)
                self.forward_hooks.append(forward_hook)
                self.backward_hooks.append(backward_hook)
                self.backward_hooks.append(backward_pre_hook)
        # register hooks for function

    def remove_hooks(self):
        for hook in self.forward_hooks + self.backward_hooks:
            hook.remove()


    def print_tree(self, filename="profiling_tree.log"):
        """以树形结构打印性能分析结果"""
        try:
            with open(filename, 'w') as f:
                # 找到所有根节点（没有父节点的模块）
                root_modules = {name: stats for name, stats in self.time_stats.items()
                              if stats['parent'] is None}

                def print_module(name, stats, indent=0):
                    # 打印当前模块信息
                    module_str = "  " * indent + f"├─ {name}\n"
                    stats_str = "  " * (indent + 1) + (
                        f"Forward: {stats['forward']:.8f}s "
                        f"(self: {stats['self_forward']:.8f}s), "
                        f"Max Forward: {stats['max_forward']:.8f}s, "
                        f"Min Forward: {0 if stats['min_forward'] == float('inf') else stats['min_forward']:.8f}s, "
                        f"Backward: {stats['backward']:.8f}s "
                        f"(self: {stats['self_backward']:.8f}s), "
                        f"Max Backward: {stats['max_backward']:.8f}s, "
                        f"Min Backward: {0 if stats['min_backward'] == float('inf') else stats['min_backward']:.8f}s, "
                        f"Count: {stats['count']}\n"
                    )
                    f.write(module_str)
                    f.write(stats_str)

                    # 找到所有子模块
                    children = [(child_name, child_stats)
                              for child_name, child_stats in self.time_stats.items()
                              if child_stats['parent'] == name]

                    # 按照forward时间排序子模块
                    children.sort(key=lambda x: x[1]['forward'], reverse=True)

                    # 递归打印子模块
                    for child_name, child_stats in children:
                        print_module(child_name, child_stats, indent + 1)

                # 打印标题
                f.write("Performance Profiling Tree\n")
                f.write("========================\n\n")

                # 从每个根节点开始打印
                for name, stats in root_modules.items():
                    print_module(name, stats)

                # 打印总计
                total_forward = sum(s['self_forward'] for s in self.time_stats.values())
                total_backward = sum(s['self_backward'] for s in self.time_stats.values())
                total_count = sum(s['count'] for s in self.time_stats.values())

                f.write("\nTotal Statistics\n")
                f.write("===============\n")
                f.write(f"Total Forward Time: {total_forward:.4f}s\n")
                f.write(f"Total Backward Time: {total_backward:.4f}s\n")
                f.write(f"Total Count: {total_count}\n")

        except Exception as e:
            print(f"Error printing tree: {e}")
