import math
import math
import numpy as np
from typing import Optional # 导入 Optional

from .module import Module
from .parameter import Parameter
from ..core import Tensor, empty # 导入 empty

# TODO: 实现更好的初始化函数 (如 kaiming_uniform_)
# TODO: 实现更好的初始化函数 (如 kaiming_uniform_)
# 导入 zeros 和 randn 用于初始化 (如果需要)
# from ..core import zeros, randn

def kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    """临时占位符 - 使用简单的均匀分布初始化"""
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    # 简单的均匀分布，范围基于 fan_in (类似 PyTorch Linear 默认)
    bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0
    tensor.data[:] = np.random.uniform(-bound, bound, size=tensor.shape)

def _calculate_fan_in_and_fan_out(tensor):
    """计算 fan_in 和 fan_out (简化版)"""
    dimensions = tensor.ndim
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    
    num_input_fmaps = tensor.shape[1]
    num_output_fmaps = tensor.shape[0]
    receptive_field_size = 1
    if dimensions > 2:
        receptive_field_size = np.prod(tensor.shape[2:])
        
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size
    
    return fan_in, fan_out


class Linear(Module):
    """
    对输入数据应用线性变换：y = xA^T + b
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Parameter
    bias: Optional[Parameter]

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 创建权重 Parameter
        # 形状是 (out_features, in_features) 以便 forward 时使用 input @ weight.T
        self.weight = Parameter(empty((out_features, in_features))) 
        
        # 创建偏置 Parameter (如果需要)
        if bias:
            self.bias = Parameter(empty(out_features))
        else:
            # 注册为 None，这样 Module 就知道没有 bias 参数
            self.register_parameter('bias', None) 
            
        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # 初始化权重
        kaiming_uniform_(self.weight, a=math.sqrt(5)) # 模仿 PyTorch
        
        # 初始化偏置
        if self.bias is not None:
            fan_in, _ = _calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            self.bias.data[:] = np.random.uniform(-bound, bound, size=self.bias.shape)

    def forward(self, input: Tensor) -> Tensor:
        # 计算 output = input @ weight.T + bias
        
        # 矩阵乘法: input @ weight.T
        # input 形状: (N, *, H_in)， H_in = in_features
        # weight 形状: (out_features, in_features)
        # weight.T 形状: (in_features, out_features)
        # output 形状: (N, *, H_out)， H_out = out_features
        
        # TODO: 需要确保 matmul 和 add 支持多维输入 (N, *)
        # 目前我们的 Function 可能只考虑了 2D 输入
        
        # 临时假设输入是 2D (N, in_features)
        if input.ndim != 2 or input.shape[1] != self.in_features:
             # 简化处理，实际应支持更多维度
             raise ValueError(f"Linear 层期望 2D 输入 (N, in_features={self.in_features})，但得到 {input.shape}")

        output = input.matmul(self.weight.T) # 使用 .T 属性
        
        if self.bias is not None:
            # 加偏置，需要广播
            # output 形状 (N, out_features)
            # bias 形状 (out_features,) -> 会被广播到 (1, out_features) 然后到 (N, out_features)
            output = output + self.bias 
            
        return output

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'