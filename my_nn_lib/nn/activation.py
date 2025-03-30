from .module import Module
from ..core import Tensor
from ..core.function import ReLU as ReLUFunction # 导入对应的 Function

class ReLU(Module):
    """
    应用修正线性单元函数 ReLU(x) = max(0, x)。
    """
    # inplace: bool # 暂时不支持 inplace

    # def __init__(self, inplace: bool = False):
    #     super().__init__()
    #     self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        # if self.inplace:
        #     # TODO: 实现 inplace ReLU (需要修改 Function 和 Tensor)
        #     raise NotImplementedError("Inplace ReLU not implemented yet")
        # else:
        # 通过 ReLU Function 计算
        # 检查输入是否需要梯度，如果需要，则通过 Function
        if input.requires_grad:
             return ReLUFunction(input).apply(input)
        else:
             # 如果输入不需要梯度，可以直接用 NumPy 计算
             # 但为了统一，还是通过 Function (它内部会处理 requires_grad=False 的情况)
             # 或者添加一个 functional 接口 F.relu()
             return ReLUFunction(input).apply(input) # Function.apply 会处理 requires_grad

    def extra_repr(self) -> str:
        # inplace_str = 'inplace=True' if self.inplace else ''
        # return inplace_str
        return "" # 暂时没有参数