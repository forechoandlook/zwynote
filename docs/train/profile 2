## profile 

目前的profile方法很多，但是粒度的处理都不太行。一种好用的方法就是通过hook机制来做，但是hook目前又只能用于module，而非func，所以导致一些细节信息又缺失。

下面介绍一个方法用来记录更多信息的。

```python
class HOOKModule(nn.Module):

    def __init__(self, hook_func) -> None:
        super(HOOKModule, self).__init__()
        self.register_forward_hook(hook_func('forward'))
        self.register_backward_hook(hook_func('backward'))


class Add(HOOKModule):
    
    def forward(self, *args, **kwargs):
        return torch._C._VariableFunctions.add(*args, **kwargs)


def patch_add(*args, **kwargs):
    return Add()(*args, **kwargs)

setattr(torch.Tensor, 'add', patch_add)
```

在此你可以控制粒度了。 但是貌似不太行，因为这是用于 tensor.add 这种方式，总有一些函数是绕过的 

```python
class TorchOPTemplate(HOOKModule):

    def __init__(self, op_name, hook_func):
        self.op_name_ = op_name
        super().__init__(hook_func)

    def forward(self, *args, **kwargs):
        return getattr(torch._C._VariableFunctionsClass, str(self.op_name_))(*args, **kwargs)
```

forward_hook+backward_hook实现整网数据抓取

`唯一标识 = Global ID + Module类型 + 算子名 + 调用栈 + ……`

```python
class TraceTensor(object):

    def __init__(self, tensor, node=None, output_idx=-1):
        self.tensor = tensor
        self.from_node = [node]
        self.to_nodes = []
        self.output_idx = output_idx
        pass

```
```python
def pre_decode_hook(module, *args, **kwargs):
    input_tensors = decode_args(args, kwargs)
    # 根据input进行构图
    for tensor in input_tensors:
        module.graph_prior.append(tensor.from_node[-1])
        module.graph_prior_output_idx.append(tensor.output_index)

    # 获取全局唯一的标识，参考调用栈和Module封装关系
    module.global_path = xxx

    # 当前module为input tensors的to_nodes，如果当前module为上层module的第一个module，那么
    # input tensors的to_nodes属性不为空，这样可以遍历所有的封装关系，更新当前节点的parent。
    module.graph_parent.append(xxx)

    # 根据graph_prior更新prior节点的next节点
    for prior in module.graph_prior:
        prior.graph_next.append(module)

    # 根据graph_parent节点更新parent的children属性
    module.graph_parent[0].graph_children.append(module)

    # 更新input tensors的to_nodes属性，确定封装关系
    for tensor in input_tensors:
        tensor.to_nodes.append(module)
```


### 获取函数调用栈信息

`inspect` 

```python
import torch
import inspect  # 用于获取调用栈

# 保存原始函数
original_add = torch._C._VariableFunctions.add

# 定义带调用栈分析的包装函数
def wrapped_add(*args, **kwargs):
    # 获取当前调用栈
    stack = inspect.stack()
    
    # 打印调用栈信息（可自定义过滤逻辑）
    print(f"\n=== Intercepted torch.add() ===")
    for frame_info in stack[1:]:  # 跳过当前帧（wrapped_add本身）
        print(f"File: {frame_info.filename}, Line: {frame_info.lineno}, Function: {frame_info.function}")
    
    # 调用原始函数
    return original_add(*args, **kwargs)

# 替换原始实现（谨慎操作！）
torch._C._VariableFunctions.add = wrapped_add
```

这种做法 可以适用 monkey patch 


我们希望什么样子的 profile ？ ====> 我们希望能看到什么样子的信息记录 

1. 能从网络结构上看到耗时
2. 能看到数据的流向

我们关心的内容：显存和性能。

想了想 好像确实也没啥好办法
