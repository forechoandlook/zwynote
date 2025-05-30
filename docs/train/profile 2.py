import torch 
from torch import nn

hook_func = lambda x: print(x)

class HOOKModule(nn.Module):

    def __init__(self, hook_func) -> None:
        super(HOOKModule, self).__init__()
        self.register_forward_hook(hook_func('forward'))
        self.register_backward_hook(hook_func('backward'))


class TorchOPTemplate(HOOKModule):

    def __init__(self, op_name, hook_func):
        self.op_name_ = op_name
        super().__init__(hook_func)

    def forward(self, *args, **kwargs):
        return getattr(torch._C._VariableFunctionsClass, str(self.op_name_))(*args, **kwargs)