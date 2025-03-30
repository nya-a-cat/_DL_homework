from typing import Iterable, Dict, List, Optional, Union # 导入 Union

from ..core import Tensor
from ..core import Tensor
from ..nn import Parameter # 需要 Parameter 类
# 导入 torch 用于类型检查 (临时，应移除)
# import torch

class Optimizer:
    """优化器基类。"""

    def __init__(self, params: Union[Iterable[Parameter], Iterable[Dict[str, any]]], defaults: Dict[str, any]):
        """
        初始化 Optimizer。

        Args:
            params (Iterable[Parameter] or Iterable[Dict]):
                需要优化的参数的可迭代对象，或者定义了参数组的字典的可迭代对象。
            defaults (Dict): 包含优化选项默认值的字典 (例如学习率 'lr')。
        """
        self.defaults = defaults
        self.param_groups: List[Dict[str, any]] = [] # 存储参数组

        # 处理传入的 params
        # if isinstance(params, (Tensor, Parameter)): # 不允许传入单个 Tensor/Parameter
        # 改为检查是否可迭代
        if not isinstance(params, Iterable):
             raise TypeError("params argument should be an iterable of Parameters or dicts, but got " +
                             type(params).__name__)

        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        
        # 如果 params 不是字典列表，则将其视为单个参数组
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        # 构建内部的 param_groups 列表
        for param_group in param_groups:
            self.add_param_group(param_group)
            
        # 存储每个参数的状态 (例如 SGD 的 momentum)
        self.state: Dict[int, Dict] = {} # 使用 id(param) 作为 key

    def add_param_group(self, param_group: Dict[str, any]):
        """添加一个参数组到优化器。"""
        assert isinstance(param_group, dict), "参数组必须是字典"

        params = param_group['params']
        if isinstance(params, (Parameter, Tensor)): # 如果只传入一个参数
             param_group['params'] = [params]
        elif isinstance(params, set): # 如果传入集合，报错
             raise TypeError('optimizer parameters need to be organized in ordered collections, but '
                             'received a set')
        else:
             param_group['params'] = list(params) # 转换为列表

        # 检查参数类型
        for param in param_group['params']:
            if not isinstance(param, Parameter):
                raise TypeError(f"optimizer can only optimize Parameters, but one of the params is {type(param)}")
            if not param.is_leaf:
                 raise ValueError("can't optimize a non-leaf Tensor")

        # 合并默认值
        for name, default in self.defaults.items():
            if name not in param_group:
                param_group.setdefault(name, default)
        
        # TODO: 检查参数组中的超参数是否有效

        self.param_groups.append(param_group)


    def zero_grad(self, set_to_none: bool = False):
        """清除所有被优化参数的梯度。"""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        # 原地将梯度清零
                        # 需要 Tensor 支持原地乘法或 zeros_like
                        # p.grad.zero_() # 假设有 zero_() 方法
                        # 临时方案：
                        if p.grad.requires_grad: # 梯度本身不应需要梯度
                             p.grad.requires_grad = False
                        p.grad._data.fill(0) # 直接修改底层 NumPy 数据

    def step(self):
        """
        执行单步优化。
        应该由子类实现。
        """
        raise NotImplementedError

    # TODO: state_dict() 和 load_state_dict() 用于保存和加载优化器状态