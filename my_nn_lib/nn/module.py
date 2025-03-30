from collections import OrderedDict
from typing import Iterator, Set, Tuple, Optional, Union
import weakref

from .parameter import Parameter
from ..core import Tensor

class Module:
    """
    所有神经网络模块的基类。

    你的模型也应该继承这个类。
    模块也可以包含其他模块，允许将它们嵌套在树结构中。
    你可以将子模块赋值为常规属性：

    import my_nn_lib.nn as nn

    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(10, 20)
            self.layer2 = nn.Linear(20, 5)

        def forward(self, x):
            x = self.layer1(x)
            x = nn.functional.relu(x) # 假设有 functional 模块
            x = self.layer2(x)
            return x

    模型将跟踪所有赋值为属性的 Parameter 和子 Module。
    """

    def __init__(self):
        """初始化 Module。"""
        # 使用弱引用集合来存储钩子，避免循环引用和内存泄漏
        self._forward_hooks = weakref.WeakKeyDictionary() # type: ignore[var-annotated]
        self._backward_hooks = weakref.WeakKeyDictionary() # type: ignore[var-annotated]
        
        self._parameters = OrderedDict() # type: OrderedDict[str, Optional[Parameter]]
        self._buffers = OrderedDict()    # type: OrderedDict[str, Optional[Tensor]] # 用于非参数状态，如 BatchNorm 的 running_mean
        self._modules = OrderedDict()    # type: OrderedDict[str, Optional['Module']]
        self._non_persistent_buffers_set = set() # type: Set[str] # 不应保存到 state_dict 的 buffer

        self._training = True # 默认处于训练模式

    def forward(self, *input):
        """定义每次调用时执行的计算。应该由所有子类重写。"""
        raise NotImplementedError(f'.forward() 方法未在 {self.__class__.__name__} 中实现')

    def register_parameter(self, name: str, param: Optional[Parameter]):
        """
        向模块添加一个参数。
        参数可以通过给定的名称作为属性访问。
        """
        if '_parameters' not in self.__dict__:
            raise AttributeError("不能在 Module.__init__() 调用 super().__init__() 之前注册参数")
        if not isinstance(name, str):
            raise TypeError(f"参数名称必须是字符串，但得到了 {type(name).__name__}")
        if '.' in name:
            raise KeyError("参数名称不能包含 '.'")
        if name == '':
            raise KeyError("参数名称不能为空字符串 ''")
        if hasattr(self, name) and name not in self._parameters:
            raise KeyError(f"属性 '{name}' 已存在")

        if param is None:
            self._parameters[name] = None
        elif not isinstance(param, Parameter):
            raise TypeError(f"不能将非 Parameter (类型为 {type(param).__name__}) 赋值为参数 '{name}'")
        else:
            self._parameters[name] = param

    def register_buffer(self, name: str, tensor: Optional[Tensor], persistent: bool = True):
        """
        向模块添加一个缓冲区。
        这通常用于注册不应被视为模型参数的状态。例如，BatchNorm 的 running_mean。
        缓冲区可以通过给定的名称作为属性访问。
        缓冲区默认是持久的，并会保存在模块的 state_dict 中。
        """
        if '_buffers' not in self.__dict__:
             raise AttributeError("不能在 Module.__init__() 调用 super().__init__() 之前注册缓冲区")
        # ... (省略类似 Parameter 的名称和类型检查) ...
        if tensor is not None and not isinstance(tensor, Tensor):
             raise TypeError(f"缓冲区必须是 Tensor 或 None，但得到了 {type(tensor).__name__}")

        self._buffers[name] = tensor
        if persistent:
            self._non_persistent_buffers_set.discard(name)
        else:
            self._non_persistent_buffers_set.add(name)

    def add_module(self, name: str, module: Optional['Module']):
        """
        向当前模块添加一个子模块。
        子模块可以通过给定的名称作为属性访问。
        """
        if not isinstance(module, Module) and module is not None:
            raise TypeError(f"{type(module).__name__} 不是 Module 的子类")
        # ... (省略类似 Parameter 的名称检查) ...
        self._modules[name] = module

    def __setattr__(self, name: str, value: Union[Parameter, 'Module', Tensor, any]):
        """
        重写 setattr 以自动注册 Parameter 和 Module。
        """
        if isinstance(value, Parameter):
            # 如果值是 Parameter，自动注册
            # 注意：移除旧参数（如果存在）以处理重新赋值的情况
            params = self.__dict__.get('_parameters')
            if params is not None and name in params:
                del params[name]
            self.register_parameter(name, value)
        elif isinstance(value, Module):
             # 如果值是 Module，自动注册
            modules = self.__dict__.get('_modules')
            if modules is not None and name in modules:
                 del modules[name]
            self.add_module(name, value)
        elif isinstance(value, Tensor) and '_buffers' in self.__dict__ and name in self._buffers:
             # 如果是对已注册缓冲区的 Tensor 赋值，更新缓冲区
             # 注意：这里假设缓冲区总是 Tensor，需要更严格的检查
             self.register_buffer(name, value, name not in self._non_persistent_buffers_set)
        
        # 调用原始的 setattr 来实际设置属性
        super().__setattr__(name, value)

    def __getattr__(self, name: str) -> Union[Tensor, Parameter, 'Module']:
        """
        使得可以通过属性访问注册的参数、缓冲区和模块。
        """
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]
        if '_modules' in self.__dict__:
            _modules = self.__dict__['_modules']
            if name in _modules:
                return _modules[name]
        raise AttributeError(f"'{self.__class__.__name__}' 对象没有属性 '{name}'")

    def __delattr__(self, name):
        """删除属性时，也从注册表中移除。"""
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._buffers:
            del self._buffers[name]
            self._non_persistent_buffers_set.discard(name)
        elif name in self._modules:
            del self._modules[name]
        else:
            super().__delattr__(name)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """
        返回模块参数的迭代器。
        通常包括所有子模块的参数。
        """
        for name, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Parameter]]:
        """返回模块参数的迭代器，同时产生参数的名称和参数本身。"""
        # 查找直接分配给此模块的参数
        memo = set() # 避免重复返回共享的参数
        for name, param in self._parameters.items():
            if param is not None and param not in memo:
                memo.add(param)
                yield prefix + ('.' if prefix else '') + name, param
        
        # 如果需要，递归查找子模块的参数
        if recurse:
            for name, module in self._modules.items():
                if module is not None:
                    submodule_prefix = prefix + ('.' if prefix else '') + name
                    for N, P in module.named_parameters(prefix=submodule_prefix, recurse=True):
                        if P not in memo: # 确保子模块的参数也没被重复添加
                             memo.add(P)
                             yield N, P

    # TODO: 实现 buffers(), named_buffers(), children(), named_children() 等辅助方法

    def train(self, mode: bool = True):
        """将模块设置为训练模式。"""
        self._training = mode
        for module in self._modules.values():
            if module is not None:
                module.train(mode)
        return self

    def eval(self):
        """将模块设置为评估模式。"""
        return self.train(False)

    def __call__(self, *input, **kwargs):
        """允许像函数一样调用模块实例。"""
        # TODO: 添加钩子 (hooks) 的逻辑
        result = self.forward(*input, **kwargs)
        return result

    def to(self, device: str):
        """将模块的参数和缓冲区移动到指定设备。"""
        # 目前只支持 'cpu'，为以后扩展 GPU 支持预留
        if device != 'cpu':
             raise NotImplementedError("目前只支持 'cpu' 设备")
        
        # 移动参数
        for param in self.parameters(recurse=False): # 只移动当前模块的
             if param is not None:
                  # Tensor 需要实现 to(device) 方法
                  # param.to(device) # 假设 Tensor.to 返回自身或新 Tensor
                  # 由于 Parameter 共享数据，可能需要特殊处理
                  # 暂时假设 Tensor.to 是原地操作或返回新 Tensor
                  # 如果返回新 Tensor，需要重新赋值给 Parameter
                  # param_on_device = param.to(device)
                  # if param_on_device is not param:
                  #    # 需要一种方式更新 Parameter 内部的 Tensor 数据指针
                  #    # 这再次指向了 Parameter 作为 Tensor 包装器可能更好
                  #    # 暂时跳过实际移动逻辑
                  pass

        # 移动缓冲区
        for buffer in self._buffers.values():
             if buffer is not None:
                  # buffer.to(device) # 同上
                  pass

        # 递归调用子模块
        for module in self._modules.values():
             if module is not None:
                  module.to(device)
        
        return self

    def __repr__(self):
        # TODO: 实现一个更好的 __repr__，显示模块结构
        return f"{self.__class__.__name__}()"