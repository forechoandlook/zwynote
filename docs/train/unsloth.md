
我感觉得先看看性能，然后再想着以后优化。
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/mistral-7b-bnb-4bit", # Supports Llama, Mistral - replace this!
    max_seq_length = 2048, # Supports RoPE Scaling internally, so choose any!
    load_in_4bit = True,
)
```


| 训练框架 | 优点 | 缺点 |
| --- | --- | --- |
| Megatron-LM | - 支持 Transformer 模型的模型并行<br>- 适用于大规模模型训练 | - 可能需要较高的硬件资源和专家知识 |
| DeepSpeed | - 用于训练大型模型<br>- 高效的训练和推理能力<br>- 能够处理具有数十亿参数的模型<br>- 实现了极致的压缩 | - 需要专业的环境配置和较强的硬件支持 |
| FairScale | - 提供了 Fully Sharded Data Parallel (FSDP)<br>- 模块化设计<br>- 提供了最佳的性能和高效的并行化 | - 较 steep 的学习曲线 |
| ColossalAI | - 提供了一系列并行组件<br>- 支持多种并行策略 | - 配置和调试可能较为复杂 |
| Alpa | - 用于训练和服务大规模神经网络<br>- 实现了自动并行化<br>- 支持分布式集群上的高效训练 | - 主要针对有一定深度学习基础的研究人员和开发者 |
| OneFlow | - 接口与 PyTorch 类似，易于使用<br>- 支持 n 维并行/分布式执行<br>- 提供了静态图编译器 | - 相对较低的社区支持和生态系统 |
| Mesh-Tensorflow | - 用于分布式深度学习<br>- 适用于复杂的分布式训练任务 | - 主要针对有一定 TensorFlow 经验的研究人员和开发者 |
| ParallelFormers | - 基于 Megatron-LM 的库<br>- 与 Huggingface 库很好地集成在一起<br>- 允许用户通过一行代码并行化模型 | - 专注于与 Huggingface 库的集成 |


## 算子




## 工程实现


## 目前训练的性能