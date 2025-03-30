import numpy as np
# from .tensor import Tensor # 移除直接导入以避免循环
import numpy as np
# 导入 typing 用于类型提示
import typing
if typing.TYPE_CHECKING:
    # 只在类型检查时导入 Tensor，避免运行时循环导入
    from .tensor import Tensor

class Function:
    """
    自动微分操作的基类。

    每个具体的运算（如加法、乘法）都应该继承这个类，
    并实现 forward 和 backward 静态方法。
    """
    parents: typing.Tuple['Tensor', ...] # Class level type hint for parents
    def __init__(self, *tensors: 'Tensor'):
        """
        初始化 Function，保存输入的 Tensor 以备反向传播使用。

        Args:
            *tensors: 输入到该运算的 Tensor。
        """
        # 保存输入的引用，用于构建计算图和反向传播
        self.parents = tensors
        # 保存需要在 backward 中使用的信息 (例如输入的形状、中间结果等)
        self.saved_tensors: typing.List['Tensor'] = [] # 类型提示

    def save_for_backward(self, *tensors: 'Tensor'):
        """保存 Tensor 以备 backward 使用。"""
        self.saved_tensors.extend(tensors)

    def apply(self, *args, **kwargs):
        """
        执行前向计算，并设置输出 Tensor 的 grad_fn。

        Args:
            *args: 传递给 forward 方法的参数。
            **kwargs: 传递给 forward 方法的关键字参数。

        Returns:
            计算结果的 Tensor (或 Tensor 元组)。
        """
        # 1. 调用子类的 forward 方法执行实际计算
        #    ctx (上下文) 就是 Function 实例自身，用于保存信息
        result_data = self.forward(self, *args, **kwargs)

        # 2. 确定输出是否需要梯度
        # 如果任何一个输入 Tensor 需要梯度，则输出也需要梯度 (确保检查 Tensor 类型)
        # 需要在运行时访问 Tensor 类以进行 isinstance 检查
        from .tensor import Tensor
        requires_grad = any(t.requires_grad for t in self.parents if isinstance(t, Tensor))

        # 3. 创建输出 Tensor
        if isinstance(result_data, tuple):
            # 如果 forward 返回多个结果
            # 需要导入 Tensor 类，但为了避免循环，我们在方法内部导入或使用 typing.cast
            from .tensor import Tensor # 在方法内部导入
            outputs = tuple(Tensor(data, requires_grad=requires_grad, _grad_fn=self if requires_grad else None, is_leaf=not requires_grad)
                            for data in result_data)
        else:
            # 如果 forward 返回单个结果
            from .tensor import Tensor # 在方法内部导入
            outputs = Tensor(result_data, requires_grad=requires_grad, _grad_fn=self if requires_grad else None, is_leaf=not requires_grad)

        return outputs

    @staticmethod
    def forward(ctx, *args, **kwargs):
        """
        执行前向计算。

        Args:
            ctx (Function): Function 实例，用于通过 ctx.save_for_backward() 保存信息。
            *args, **kwargs: 运算所需的输入 Tensor 或其他参数。

        Returns:
            计算结果 (NumPy 数组或数组元组)。
        """
        raise NotImplementedError("必须在子类中实现 forward 方法")

    @staticmethod
    def backward(ctx, grad_output: 'Tensor'):
        """
        执行反向传播，计算输入的梯度。

        Args:
            ctx (Function): Function 实例，包含通过 save_for_backward 保存的 Tensor。
            grad_output (Tensor): 输出 Tensor 的梯度。

        Returns:
            输入 Tensor 的梯度元组 (与 forward 输入的 Tensor 顺序一致)。
            如果某个输入不需要梯度，则对应位置返回 None。
        """
        raise NotImplementedError("必须在子类中实现 backward 方法")


# --- 辅助函数 ---

def _sum_to_shape(data: np.ndarray, target_shape: tuple) -> np.ndarray:
    """将 data (梯度) 沿着广播的维度求和，使其形状与 target_shape 匹配。"""
    if data.shape == target_shape:
        return data

    # 1. 计算需要求和的轴
    #    - data 比 target 多出的前导维度
    #    - data 和 target 维度相同，但 target 维度为 1 而 data 维度不为 1 的轴
    ndim_diff = data.ndim - len(target_shape)
    axes_to_sum = list(range(ndim_diff)) # 多出的前导维度

    for i in range(len(target_shape)):
        data_dim_idx = i + ndim_diff # data 中对应的维度索引
        if target_shape[i] == 1 and data.shape[data_dim_idx] != 1:
            axes_to_sum.append(data_dim_idx)
        elif target_shape[i] != 1 and data.shape[data_dim_idx] == 1:
             # data 维度为 1，target 维度不为 1 (data 被广播了)
             # sum 操作会自动处理这种情况，无需显式加入 axes_to_sum
             pass
        elif target_shape[i] != data.shape[data_dim_idx]:
             # 维度不匹配，且 target 维度不为 1，说明广播规则有问题
             # （理论上 NumPy 前向会报错，这里作为保险）
             raise ValueError(f"无法将形状 {data.shape} 的梯度缩减到 {target_shape}，维度 {i} 不匹配")

    # 2. 执行求和
    if axes_to_sum:
        summed_data = data.sum(axis=tuple(axes_to_sum), keepdims=True)
    else:
        summed_data = data # 无需执行 sum

    # 3. 调整形状以匹配 target_shape
    #    - 如果 summed_data 的维度多于 target_shape (因为 keepdims=True 和 ndim_diff)，移除多余的前导维度
    #    - 如果 summed_data 的维度少于 target_shape (不可能发生，因为 keepdims=True)
    #    - 如果维度相同但形状仍不匹配 (例如 (1, 3) vs (3,))，则 reshape
    
    if ndim_diff > 0:
         # 移除由于 ndim_diff 导致的多余前导维度 (这些维度在 sum 后大小为 1)
         summed_data = summed_data.reshape(data.shape[ndim_diff:])

    # 最终检查并 reshape (处理 (1, 3) -> (3,) 或 (3,) -> (1, 3) 的情况)
    if summed_data.shape != target_shape:
         try:
              # 尝试直接 reshape，如果元素数量匹配的话
              reshaped_data = summed_data.reshape(target_shape)
              if reshaped_data.shape == target_shape: # 再次确认
                   summed_data = reshaped_data
              else: # reshape 失败或形状仍不符
                   raise ValueError() # 跳转到 except
         except ValueError as e:
              raise ValueError(f"最终梯度形状调整失败: 从 {data.shape} "
                               f"求和后得到 {summed_data.shape}, "
                               f"无法调整为目标形状 {target_shape}") from e

    return summed_data


# --- 具体运算的 Function 实现 ---

class Add(Function):
    @staticmethod
    def forward(ctx, x: 'Tensor', y: 'Tensor') -> np.ndarray:
        """加法前向传播"""
        # 不需要保存任何东西用于 backward
        return x.data + y.data

    @staticmethod
    def backward(ctx, grad_output: 'Tensor'):
        """加法反向传播"""
        # 加法的梯度直接传递给两个输入
        # d(x+y)/dx = 1, d(x+y)/dy = 1
        # 链式法则：dL/dx = dL/d(x+y) * d(x+y)/dx = grad_output * 1
        #           dL/dy = dL/d(x+y) * d(x+y)/dy = grad_output * 1
        # 处理广播：将梯度缩减到输入的原始形状
        grad_x = _sum_to_shape(grad_output.data, ctx.parents[0].shape)
        grad_y = _sum_to_shape(grad_output.data, ctx.parents[1].shape)
        # 返回 Tensor
        # 需要导入 Tensor 类
        from .tensor import Tensor
        grad_x = Tensor(grad_x) if grad_x is not None else None
        grad_y = Tensor(grad_y) if grad_y is not None else None
        return grad_x, grad_y


class Sum(Function):
    @staticmethod
    def forward(ctx, x: 'Tensor', axis=None, keepdims=False) -> np.ndarray:
        """求和前向传播"""
        # 保存输入的形状和求和的轴/维度信息，用于 backward
        ctx.input_shape = x.shape
        ctx.axis = axis
        ctx.keepdims = keepdims
        return x.data.sum(axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(ctx, grad_output: 'Tensor'):
        """求和反向传播"""
        # Sum 操作的梯度是将输出梯度广播回输入的原始形状
        # d(sum(x))/dx_i = 1 for all i involved in the sum
        # 链式法则：dL/dx_i = dL/d(sum(x)) * d(sum(x))/dx_i = grad_output * 1
        # 我们需要将 grad_output (通常是标量或缩减了维度的 Tensor) 扩展回输入的形状
        
        input_shape = ctx.input_shape
        axis = ctx.axis
        keepdims = ctx.keepdims
        
        # 如果 grad_output 是标量，直接广播
        if np.isscalar(grad_output.data) or grad_output.ndim == 0:
            grad_input_data = np.full(input_shape, grad_output.data, dtype=grad_output.dtype)
        else:
            # 如果不是标量，说明 sum 时指定了 axis 并且 keepdims=True
            # 或者 sum 没有指定 axis 但输入本身就是标量 (这种情况 grad_output 也是标量)
            # 我们需要将 grad_output 扩展/广播回 input_shape
            # np.expand_dims 可以用来恢复被 sum 掉的维度 (如果 keepdims=False)
            # np.broadcast_to 可以将梯度广播到正确的形状

            # 1. 如果 keepdims=False，恢复被求和的维度
            if not keepdims and axis is not None:
                if isinstance(axis, int):
                    axis = (axis,)
                # 确保 axis 是正数
                actual_axis = tuple(a % len(input_shape) for a in axis)
                shape_with_kept_dims = list(input_shape)
                grad_shape = list(grad_output.shape)
                
                # 插入被移除的维度
                current_grad_dim = 0
                expanded_shape = []
                original_indices_in_grad = []
                for i in range(len(input_shape)):
                     if i in actual_axis:
                          expanded_shape.append(1) # 恢复维度大小为 1
                     else:
                          expanded_shape.append(grad_shape[current_grad_dim])
                          original_indices_in_grad.append(i)
                          current_grad_dim += 1
                
                # 如果 grad_output 维度与预期不符 (例如 sum 了所有维度)
                if current_grad_dim != len(grad_shape):
                     # 可能是 sum(axis=None) 但 keepdims=False
                     # 这种情况下 grad_output 是标量，前面已处理
                     # 如果不是标量，则逻辑有误
                     if grad_output.ndim != 0:
                          raise RuntimeError(f"Sum backward shape mismatch: grad_output {grad_output.shape}, expected reduced shape from {input_shape} along {axis}")

                # Reshape grad_output to have the kept dims
                grad_output_expanded = grad_output.data.reshape(expanded_shape)
            else: # keepdims=True or axis=None (grad is scalar)
                grad_output_expanded = grad_output.data

            # 2. 将梯度广播到原始输入形状
            grad_input_data = np.broadcast_to(grad_output_expanded, input_shape)

        # 需要导入 Tensor 类
        from .tensor import Tensor
        return Tensor(grad_input_data) # Sum 只有一个输入 Tensor


class Mul(Function):
    @staticmethod
    def forward(ctx, x: 'Tensor', y: 'Tensor') -> np.ndarray:
        """乘法前向传播"""
        # 需要保存输入 x 和 y 用于计算梯度
        ctx.save_for_backward(x, y)
        return x.data * y.data

    @staticmethod
    def backward(ctx, grad_output: 'Tensor'):
        """乘法反向传播"""
        # d(x*y)/dx = y, d(x*y)/dy = x
        # 链式法则：dL/dx = dL/d(x*y) * d(x*y)/dx = grad_output * y
        #           dL/dy = dL/d(x*y) * d(x*y)/dy = grad_output * x
        x, y = ctx.saved_tensors
        grad_x_unsummed = grad_output * y
        grad_y_unsummed = grad_output * x
        # 处理广播：将梯度缩减到输入的原始形状
        grad_x = _sum_to_shape(grad_x_unsummed.data, x.shape)
        grad_y = _sum_to_shape(grad_y_unsummed.data, y.shape)
        # 返回 Tensor
        # Ensure Tensor is imported locally if not already
        from .tensor import Tensor
        grad_x = Tensor(grad_x) if grad_x is not None else None
        grad_y = Tensor(grad_y) if grad_y is not None else None
        return grad_x, grad_y


class Transpose(Function):
    @staticmethod
    def forward(ctx, x: 'Tensor', axes=None) -> np.ndarray:
        """转置前向传播"""
        # 保存轴信息用于 backward
        ctx.axes = axes
        return x.data.transpose(axes)

    @staticmethod
    def backward(ctx, grad_output: 'Tensor'):
        """转置反向传播"""
        # 梯度的梯度是再次转置回去
        # 如果原始 axes 是 (1, 0)，那么 backward 的 axes 也是 (1, 0)
        # 如果原始 axes 是 None (反转所有维度)，backward 也反转所有维度
        
        # 需要一种方法来获取反向转置所需的轴
        # 如果 axes = (a, b, c), 那么反向需要找到 p 使得 p[a]=0, p[b]=1, p[c]=2
        # 如果 axes = None, 反向也是 None
        
        axes = ctx.axes
        if axes is None:
             # 反转所有维度
             grad_input_data = grad_output.data.transpose() # NumPy transpose(None) 行为
        else:
             # 计算逆置换
             # 例如 axes = (0, 2, 1) for a 3D tensor
             # backward_axes should be (0, 2, 1) as well? Let's test.
             # If x' = x.transpose(axes), then dx = dx'.transpose(axes) ? Yes.
             grad_input_data = grad_output.data.transpose(axes)

        # 需要导入 Tensor 类
        from .tensor import Tensor
        return Tensor(grad_input_data) # Transpose 只有一个输入


class MatMul(Function):
    @staticmethod
    def forward(ctx, x: 'Tensor', y: 'Tensor') -> np.ndarray:
        """矩阵乘法前向传播"""
        # 需要保存输入 x 和 y 用于计算梯度
        ctx.save_for_backward(x, y)
        return np.matmul(x.data, y.data)

    @staticmethod
    def backward(ctx, grad_output: 'Tensor'):
        """矩阵乘法反向传播"""
        # Z = X @ Y
        # dL/dX = dL/dZ @ Y.T
        # dL/dY = X.T @ dL/dZ
        x, y = ctx.saved_tensors
        
        # 计算 dL/dX
        # 需要实现 transpose 或在 matmul 中处理
        # 暂时假设有 transpose 方法
        # TODO: 实现 Tensor.transpose() 的 autograd 支持
        # grad_x = grad_output @ y.transpose() # 理想情况
        # 临时方案：直接使用 NumPy 计算，然后包装
        grad_x_data = np.matmul(grad_output.data, y.data.swapaxes(-1, -2)) # 使用 swapaxes 实现转置
        
        # 计算 dL/dY
        # grad_y = x.transpose() @ grad_output # 理想情况
        grad_y_data = np.matmul(x.data.swapaxes(-1, -2), grad_output.data)

        # 处理广播 (如果 matmul 涉及广播，例如 batch matmul)
        # matmul 的广播比较复杂，梯度缩减也复杂
        # 暂时假设没有 batch 维度或广播，或者广播由 NumPy 处理，梯度直接匹配
        # TODO: 为 matmul 实现更完善的广播梯度处理
        grad_x = _sum_to_shape(grad_x_data, x.shape)
        grad_y = _sum_to_shape(grad_y_data, y.shape)

        # 需要导入 Tensor 类
        from .tensor import Tensor
        grad_x = Tensor(grad_x) if grad_x is not None else None
        grad_y = Tensor(grad_y) if grad_y is not None else None
        
        return grad_x, grad_y


class ReLU(Function):
    @staticmethod
    def forward(ctx, x: 'Tensor') -> np.ndarray:
        """ReLU 前向传播"""
        # 保存输入 x 用于计算梯度 (只需要知道哪些元素 > 0)
        ctx.save_for_backward(x)
        return np.maximum(x.data, 0)

    @staticmethod
    def backward(ctx, grad_output: 'Tensor'):
        """ReLU 反向传播"""
        # d(relu(x))/dx = 1 if x > 0 else 0
        # 链式法则：dL/dx = dL/d(relu(x)) * d(relu(x))/dx = grad_output * (1 if x > 0 else 0)
        x, = ctx.saved_tensors
        mask = x.data > 0
        grad_input_data = grad_output.data * mask
        
        # 需要导入 Tensor 类
        from .tensor import Tensor
        return Tensor(grad_input_data) # ReLU 只有一个输入


class Sub(Function):
    @staticmethod
    def forward(ctx, x: 'Tensor', y: 'Tensor') -> np.ndarray:
        """减法前向传播"""
        return x.data - y.data

    @staticmethod
    def backward(ctx, grad_output: 'Tensor'):
        """减法反向传播"""
        # d(x-y)/dx = 1
        # d(x-y)/dy = -1
        # dL/dx = dL/d(x-y) * 1 = grad_output
        # dL/dy = dL/d(x-y) * (-1) = -grad_output
        grad_x = _sum_to_shape(grad_output.data, ctx.parents[0].shape)
        grad_y = _sum_to_shape(-grad_output.data, ctx.parents[1].shape) # 注意负号
        
        from .tensor import Tensor
        grad_x = Tensor(grad_x) if grad_x is not None else None
        grad_y = Tensor(grad_y) if grad_y is not None else None
        return grad_x, grad_y


class Pow(Function):
    @staticmethod
    def forward(ctx, x: 'Tensor', exponent: float) -> np.ndarray:
        """幂运算前向传播 (只支持标量指数)"""
        if not isinstance(exponent, (int, float)):
             raise TypeError("Pow exponent must be a scalar number")
        ctx.save_for_backward(x)
        ctx.exponent = exponent
        return x.data ** exponent

    @staticmethod
    def backward(ctx, grad_output: 'Tensor'):
        """幂运算反向传播"""
        # d(x^n)/dx = n * x^(n-1)
        # dL/dx = dL/d(x^n) * d(x^n)/dx = grad_output * n * x^(n-1)
        x, = ctx.saved_tensors
        exponent = ctx.exponent
        
        # 避免 x=0 且 exponent-1 < 0 的情况 (例如 sqrt 的梯度)
        # grad_input_data = grad_output.data * exponent * (x.data ** (exponent - 1))
        # 更安全的计算方式：
        pow_val = x.data ** (exponent - 1)
        # 处理可能出现的 inf/nan (例如 0**(-0.5))
        # 如果 x.data 是 0 且 exponent-1 < 0，pow_val 会是 inf
        # 梯度应该是 0 还是 inf？取决于具体函数。对于 sqrt(x)，x=0 处梯度是 inf。
        # 暂时不处理特殊情况，依赖 NumPy 的行为
        grad_input_data = grad_output.data * exponent * pow_val
        
        # 处理广播 (幂运算通常是逐元素的，所以梯度形状应与输入一致)
        grad_input = _sum_to_shape(grad_input_data, x.shape)

        from .tensor import Tensor
        return Tensor(grad_input) # Pow 只有一个 Tensor 输入

