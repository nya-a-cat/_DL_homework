from .optimizer import Optimizer
from ..core import Tensor # 需要 Tensor

# 导入 typing 用于类型提示 (虽然 Optimizer 基类已经用了)
from typing import List, Optional, Dict

class SGD(Optimizer):
    """
    实现随机梯度下降 (可选支持动量)。
    """

    def __init__(self, params, lr: float, momentum: float = 0, dampening: float = 0,
                 weight_decay: float = 0, nesterov: bool = False):
        """
        初始化 SGD 优化器。

        Args:
            params: 需要优化的模型参数 (Iterable or Dict)。
            lr (float): 学习率。
            momentum (float, optional): 动量因子 (默认为 0)。
            dampening (float, optional): 动量的抑制因子 (默认为 0)。
            weight_decay (float, optional): 权重衰减 (L2 惩罚) (默认为 0)。
            nesterov (bool, optional): 是否启用 Nesterov 动量 (默认为 False)。
        """
        if lr < 0.0:
            raise ValueError(f"无效的学习率: {lr}")
        if momentum < 0.0:
            raise ValueError(f"无效的动量值: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"无效的权重衰减值: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
             raise ValueError("Nesterov 动量需要 momentum > 0 且 dampening = 0")

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        super().__init__(params, defaults)

    def step(self):
        """执行单步优化。"""
        # TODO: 添加对闭包 (closure) 的支持 (可选)
        
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            dampening = group['dampening']
            weight_decay = group['weight_decay']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue # 如果参数没有梯度，则跳过

                grad = p.grad
                if grad.requires_grad:
                     # 梯度本身不应该需要梯度计算
                     # raise RuntimeError("梯度不应设置 requires_grad=True")
                     # 或者更安全地，创建一个不需要梯度的副本？
                     grad = Tensor(grad.data, requires_grad=False)


                # 应用权重衰减 (L2 惩罚)
                # grad = grad + weight_decay * p
                # 注意：p 是 Parameter，也是 Tensor。需要确保类型和 requires_grad 正确
                if weight_decay != 0:
                    # 确保 p.data 用于计算，避免创建计算图
                    # grad = grad + Tensor(weight_decay * p.data, requires_grad=False) # 效率低
                    # 原地操作：
                    grad.data += weight_decay * p.data # 直接修改梯度数据

                # 应用动量
                if momentum != 0:
                    param_state = self.state.setdefault(id(p), {}) # 获取或创建参数状态
                    if 'momentum_buffer' not in param_state:
                        # 初始化动量缓冲区
                        buf = param_state['momentum_buffer'] = grad.copy() # 复制梯度
                    else:
                        buf = param_state['momentum_buffer']
                        # 更新动量: buf = momentum * buf + (1 - dampening) * grad
                        # buf *= momentum
                        # buf += (1 - dampening) * grad # 需要实现 *= 和 +=
                        # 临时方案：
                        buf.data *= momentum
                        buf.data += (1 - dampening) * grad.data

                    if nesterov:
                        # Nesterov 动量: grad = grad + momentum * buf
                        # grad += momentum * buf # 需要实现 +=
                        grad.data += momentum * buf.data
                    else:
                        # 普通动量: grad = buf
                        grad = buf # grad 现在指向动量缓冲区

                # 执行 SGD 更新步骤: p = p - lr * grad
                # 需要确保 p 是 Parameter，grad 是 Tensor
                # 更新应该是原地操作，直接修改 p 的数据
                # p.data -= lr * grad.data
                
                # 使用 Tensor 的原地减法 (如果实现) 或直接操作 NumPy 数据
                # p -= lr * grad # 假设 Tensor 实现了 __isub__
                # 临时方案：直接修改底层 NumPy 数据
                p._data -= lr * grad.data # 修改为访问 _data
                # 注意：这种直接修改 _data 的方式会绕过 Autograd，
                # 但对于优化器更新叶子节点参数是允许的。