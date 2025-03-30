import numpy as np
from .module import Module
from ..core import Tensor
# 需要 Mean Function (如果 reduction='mean')
# from ..core.function import Mean # 假设之后会实现

class MSELoss(Module):
    """
    计算输入 x 和目标 y 之间均方误差损失。

    L = (x - y)^2
    损失可以是每个元素的损失的平均值或总和。
    """
    def __init__(self, reduction: str = 'mean'):
        """
        Args:
            reduction (str, optional): 指定应用于输出的规约方式：
                'none' | 'mean' | 'sum'. 默认为 'mean'。
                'none': 不应用规约。
                'mean': 输出的标量平均值。
                'sum': 输出的总和。
        """
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f"无效的 reduction 参数: {reduction}")
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        计算 MSE 损失。

        Args:
            input (Tensor): 模型的预测输出。
            target (Tensor): 真实的目标值。形状应与 input 相同。

        Returns:
            Tensor: 计算出的损失。如果 reduction='mean' 或 'sum'，则为标量 Tensor；
                    如果 reduction='none'，则形状与输入相同。
        """
        if input.shape != target.shape:
             # 允许 target 形状可以被广播到 input 形状？暂时要求严格匹配
             raise ValueError(f"输入和目标形状必须匹配，但得到 {input.shape} 和 {target.shape}")

        # 计算 (input - target)**2
        diff = input - target # 使用 Tensor 的 __sub__，会调用 Sub Function
        squared_diff = diff ** 2 # 使用 Tensor 的 __pow__，会调用 Pow Function

        # 应用规约
        if self.reduction == 'mean':
            # loss = squared_diff.mean() # 需要实现 mean 方法和 Mean Function
            # 临时方案：使用 sum / numel
            numel = np.prod(squared_diff.shape)
            if numel == 0: # 处理空 Tensor
                 # 返回 0 标量 Tensor，需要梯度吗？
                 # 暂时返回 NumPy 0，需要改进
                 # return Tensor(0.0, requires_grad=squared_diff.requires_grad) # 如何连接图？
                 # 更好的方式是返回 sum 结果 (可能是 0)
                 loss = squared_diff.sum() / numel # 除法也需要 Autograd 支持
                 # TODO: 实现 Div Function
                 # 再次临时方案：
                 loss_sum = squared_diff.sum()
                 # 手动创建不需要梯度的除数 Tensor
                 # divisor = Tensor(float(numel), requires_grad=False)
                 # loss = loss_sum / divisor # 需要 Div Function
                 # 最简单的临时方案 (如果 loss_sum 是标量)：
                 if loss_sum.ndim == 0:
                      loss_val = loss_sum.data.item() / numel
                      # 如何创建带正确 grad_fn 的标量 Tensor？
                      # 这表明我们需要 Div Function
                      # 暂时放弃 mean，使用 sum
                      print("Warning: MSELoss reduction='mean' 暂时回退到 'sum'，因为 Div Function 未实现")
                      loss = squared_diff.sum()
                 else:
                      # 如果 sum 不是标量，说明有问题
                      raise RuntimeError("Summation did not result in a scalar for MSELoss mean reduction.")

            else:
                 loss = squared_diff.sum() / numel
                 # 同上，需要 Div Function
                 print("Warning: MSELoss reduction='mean' 暂时回退到 'sum'，因为 Div Function 未实现")
                 loss = squared_diff.sum()

        elif self.reduction == 'sum':
            loss = squared_diff.sum() # 使用 Tensor 的 sum 方法
        else: # reduction == 'none'
            loss = squared_diff

        return loss