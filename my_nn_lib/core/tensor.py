import numpy as np
# 导入 Function 和具体的运算类
from .function import Function, Add, Mul, Sum, MatMul, Transpose, Sub, Pow # 添加 Sub, Pow

class Tensor:
    """
    一个简单的张量类，目前使用 NumPy 作为后端。
    """
    def __init__(self, data, dtype=None, requires_grad=False, _grad_fn=None, is_leaf=True):
        """
        初始化 Tensor。

        Args:
            data: 可以是 list, tuple, numpy.ndarray 或其他 Tensor。
            dtype: 指定的数据类型 (例如 np.float32)。如果为 None，则尝试从 data 推断。
            requires_grad (bool): 是否需要计算该张量的梯度。
            _grad_fn (Function, optional): 创建该张量的 Function 对象 (用于反向传播)。
            is_leaf (bool): 指示该张量是否为叶子节点。
        """
        if isinstance(data, np.ndarray):
            # 如果已经是 ndarray，根据 dtype 创建副本或直接使用
            if dtype is not None and data.dtype != dtype:
                self._data = data.astype(dtype)
            else:
                self._data = data # 避免不必要的复制
        elif isinstance(data, (list, tuple)):
            # 从 list 或 tuple 创建 ndarray
            self._data = np.array(data, dtype=dtype)
        elif isinstance(data, Tensor):
             # 从另一个 Tensor 创建 (共享数据还是复制？暂时复制以避免副作用)
             # TODO: 考虑是否需要更复杂的共享/复制策略
            if dtype is not None and data.dtype != dtype:
                self._data = data.data.astype(dtype).copy()
            else:
                self._data = data.data.copy()
                # 如果从另一个需要梯度的 Tensor 创建，则新 Tensor 也需要梯度
                # 但通常创建函数会处理 requires_grad
                # requires_grad = requires_grad or data.requires_grad # 暂时注释，让创建函数决定
        else:
            # 检查是否为 Python 或 NumPy 的数字标量类型
            if isinstance(data, (int, float, np.number)):
                # 使用 np.asarray 将标量转换为 0-d 数组
                self._data = np.asarray(data, dtype=dtype)
            else:
                # 对于所有其他不支持的类型，直接抛出 TypeError
                raise TypeError(f"不支持的数据类型来创建 Tensor: {type(data)}")

        # 确保 self._data 是一个 ndarray
        if not isinstance(self._data, np.ndarray):
             # 这理论上不应该发生，但作为保险
             self._data = np.array(self._data, dtype=dtype)

        # 设置 device 属性 (M1 阶段默认为 'cpu')
        self._device = 'cpu' # 稍后会扩展

        # Autograd 相关属性
        self.requires_grad = requires_grad and np.issubdtype(self.dtype, np.floating) # 梯度只对浮点数有意义
        self.grad = None # 存储梯度，也是一个 Tensor
        self._grad_fn = _grad_fn # 指向创建此 Tensor 的 Function
        self._is_leaf = is_leaf or not self.requires_grad # 用户创建的或不需要梯度的都是叶子节点

        if self.requires_grad and not self._is_leaf and self._grad_fn is None:
             # requires_grad=True 的非叶子节点必须有关联的 _grad_fn
             # (除非是 inplace 操作修改了叶子节点，暂时不考虑 inplace)
             raise RuntimeError("requires_grad=True 的非叶子节点必须有关联的 grad_fn")

        if self.requires_grad and self._is_leaf:
             # 叶子节点不应该有 _grad_fn
             if self._grad_fn is not None:
                 raise RuntimeError("叶子节点不应有关联的 grad_fn")

    @property
    def data(self) -> np.ndarray:
        """返回底层的 NumPy 数组。"""
        return self._data

    @property
    def shape(self) -> tuple:
        """返回张量的形状。"""
        return self._data.shape

    @property
    def dtype(self):
        """返回张量的数据类型。"""
        return self._data.dtype

    @property
    def device(self) -> str:
        """返回张量所在的设备。"""
        return self._device

    @classmethod
    def _make_subclass(cls, subclass, data, requires_grad):
        """
        工厂方法，用于创建 Tensor 的子类实例，共享底层数据。
        这主要用于 Parameter 类。
        """
        # 创建子类实例，跳过复杂的 __init__ 逻辑
        instance = object.__new__(subclass)
        
        # 直接设置必要的属性
        instance._data = data # 共享传入的 NumPy 数组
        instance._device = 'cpu' # 假设目前只有 CPU
        instance.requires_grad = requires_grad and np.issubdtype(data.dtype, np.floating)
        instance.grad = None
        # 子类（如 Parameter）通常是叶子节点
        instance._is_leaf = True
        instance._grad_fn = None

        # 执行必要的检查 (从 __init__ 复制过来)
        if instance.requires_grad and not instance._is_leaf and instance._grad_fn is None:
             raise RuntimeError("requires_grad=True 的非叶子节点必须有关联的 grad_fn")
        if instance.requires_grad and instance._is_leaf:
             if instance._grad_fn is not None:
                 raise RuntimeError("叶子节点不应有关联的 grad_fn")

        return instance

    @property
    def is_leaf(self) -> bool:
        """判断是否为叶子节点。"""
        return self._is_leaf

    @property
    def ndim(self) -> int:
        """返回张量的维度数。"""
        return self._data.ndim

    def __len__(self) -> int:
        """返回第一个维度的大小。"""
        if self.ndim == 0:
            raise TypeError("0-d tensor doesn't have len()")
        return self.shape[0]

    def __repr__(self) -> str:
        """返回 Tensor 的详细字符串表示形式。"""
        grad_fn_str = f", grad_fn=<{self._grad_fn.__class__.__name__}>" if self._grad_fn else ""
        req_grad_str = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor({self.data}, dtype={self.dtype}, device='{self.device}'{req_grad_str}{grad_fn_str})"

    def __str__(self) -> str:
        """返回 Tensor 的更简洁字符串表示形式。"""
        # 优化打印，避免过长输出
        return np.array_str(self.data, precision=4, suppress_small=True)

    # --- 形状操作 ---

    def reshape(self, *shape) -> 'Tensor':
        """
        改变 Tensor 的形状。

        Args:
            *shape: 新的形状元组或整数序列。

        Returns:
            一个新的 Tensor，具有指定的形状。数据可能共享或复制，取决于 NumPy 的实现。
        """
        # 如果 shape 是一个元组/列表，解包它
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return Tensor(self.data.reshape(shape))

    def transpose(self, *axes) -> 'Tensor':
        """
        转置 Tensor 的维度。

        Args:
            *axes: 指定维度顺序的元组或整数序列。如果省略，则反转所有维度。

        Returns:
            一个新的转置后的 Tensor。
        """
        # 处理 axes 输入，使其成为 None 或元组
        actual_axes = axes if axes else None
        if len(axes) == 1 and isinstance(axes[0], (list, tuple)):
             actual_axes = tuple(axes[0])
        elif len(axes) > 1:
             actual_axes = tuple(axes)
        # else: actual_axes is None or a single int? NumPy handles single int? Let's assume tuple or None.

        if self.requires_grad:
             return Transpose(self).apply(self, axes=actual_axes)
        else:
             return Tensor(self.data.transpose(actual_axes), requires_grad=False)

    # 添加 T 属性作为 transpose() 的快捷方式 (反转最后两个维度)
    @property
    def T(self) -> 'Tensor':
        """
        返回最后两个维度转置后的 Tensor。
        等效于 self.transpose(-1, -2) 如果 ndim >= 2，否则是 self。
        """
        if self.ndim < 2:
            return self
        else:
            axes = list(range(self.ndim))
            axes[-1], axes[-2] = axes[-2], axes[-1]
            return self.transpose(tuple(axes))

    # --- 数学运算 ---

    def _ensure_tensor(self, other) -> 'Tensor':
        """确保另一个操作数是 Tensor 或可以转换为 Tensor。"""
        if not isinstance(other, Tensor):
            # 尝试将标量或其他类型转换为与 self 兼容的 Tensor
            try:
                # 使用 self 的 dtype 以保持一致性
                return Tensor(other, dtype=self.dtype)
            except TypeError:
                return NotImplemented # 表示无法处理此类型
        return other
    
    # Note: The arithmetic operations (add, sub, mul, div, matmul) themselves
    # seem okay, relying on _ensure_tensor and the fixed __init__.
    # We just need to ensure the operator overrides correctly handle NotImplemented.

    def add(self, other) -> 'Tensor':
        """逐元素加法。"""
        other = self._ensure_tensor(other)
        if other is NotImplemented:
            return NotImplemented
        
        # 如果 self 或 other 需要梯度，则通过 Add Function 计算
        if self.requires_grad or other.requires_grad:
            return Add(self, other).apply(self, other)
        else:
            # 否则直接计算，不记录梯度
            return Tensor(self.data + other.data, requires_grad=False)

    def sub(self, other) -> 'Tensor':
        """逐元素减法。"""
        other = self._ensure_tensor(other)
        if other is NotImplemented:
            return NotImplemented
        
        if self.requires_grad or other.requires_grad:
             return Sub(self, other).apply(self, other)
        else:
             return Tensor(self.data - other.data, requires_grad=False)

    def mul(self, other) -> 'Tensor':
        """逐元素乘法。"""
        other = self._ensure_tensor(other)
        if other is NotImplemented:
            return NotImplemented

        # 如果 self 或 other 需要梯度，则通过 Mul Function 计算
        if self.requires_grad or other.requires_grad:
            return Mul(self, other).apply(self, other)
        else:
             # 否则直接计算，不记录梯度
            return Tensor(self.data * other.data, requires_grad=False)

    def div(self, other) -> 'Tensor':
        """逐元素除法。"""
        other = self._ensure_tensor(other)
        if other is NotImplemented:
            return NotImplemented
        # 考虑除零错误？NumPy 默认会产生 inf 或 nan
        return Tensor(self.data / other.data)

    def matmul(self, other) -> 'Tensor':
        """矩阵乘法。"""
        other = self._ensure_tensor(other)
        if other is NotImplemented:
            return NotImplemented
        if self.ndim < 1 or other.ndim < 1:
             raise ValueError(f"matmul requires at least 1-dimensional tensors, but got {self.ndim}D and {other.ndim}D")
        
        # 使用 MatMul Function
        if self.requires_grad or other.requires_grad:
             return MatMul(self, other).apply(self, other)
        else:
             return Tensor(np.matmul(self.data, other.data), requires_grad=False)

    def pow(self, exponent: float) -> 'Tensor':
        """逐元素幂运算。"""
        if not isinstance(exponent, (int, float)):
             raise TypeError("Pow exponent must be a scalar number")
             
        if self.requires_grad:
             # 注意：Pow Function 只接受 Tensor 作为第一个参数
             return Pow(self).apply(self, exponent=exponent)
        else:
             return Tensor(self.data ** exponent, requires_grad=False)

    def sum(self, axis=None, keepdims=False) -> 'Tensor':
        """计算张量元素的和。"""
        if self.requires_grad:
            # 通过 Sum Function 计算以支持 autograd
            return Sum(self).apply(self, axis=axis, keepdims=keepdims)
        else:
            # 直接计算
            return Tensor(self.data.sum(axis=axis, keepdims=keepdims), requires_grad=False)

    # --- 运算符重载 ---

    def __add__(self, other):
        return self.add(other)
        # Python 会自动处理 NotImplemented:
        # If self.add(other) returns NotImplemented, Python tries other.__radd__(self)

    def __radd__(self, other): # 处理 other + self 的情况
        other = self._ensure_tensor(other)
        if other is NotImplemented:
            return NotImplemented
        return other.add(self)

    def __sub__(self, other):
        return self.sub(other)
        # Python 会自动处理 NotImplemented

    def __rsub__(self, other): # 处理 other - self 的情况
        other = self._ensure_tensor(other)
        if other is NotImplemented:
            return NotImplemented
        return other.sub(self)

    def __mul__(self, other):
        return self.mul(other)
        # Python 会自动处理 NotImplemented

    def __rmul__(self, other): # 处理 other * self 的情况
        other = self._ensure_tensor(other)
        if other is NotImplemented:
            return NotImplemented
        return other.mul(self)

    def __truediv__(self, other):
        return self.div(other)
        # Python 会自动处理 NotImplemented

    def __rtruediv__(self, other): # 处理 other / self 的情况
        other = self._ensure_tensor(other)
        if other is NotImplemented:
            return NotImplemented
        return other.div(self)

    def __matmul__(self, other):
        return self.matmul(other)
        # Python 会自动处理 NotImplemented

    def __rmatmul__(self, other): # 处理 other @ self 的情况
        other = self._ensure_tensor(other)
        if other is NotImplemented:
            return NotImplemented
        return other.matmul(self)

    def __pow__(self, exponent):
        # 只处理右侧是标量指数的情况
        if isinstance(exponent, (int, float)):
             return self.pow(exponent)
        else:
             # 如果指数是 Tensor，需要实现不同的 Function
             return NotImplemented
    
    # Note: __rpow__ (other ** self) is more complex as the base changes.
    # We don't implement it for now.

    def __iadd__(self, other):
        """原地加法 (+=)。"""
        other = self._ensure_tensor(other)
        if other is NotImplemented:
            return NotImplemented
        
        # 执行原地加法
        # 注意：这会修改 Tensor 的底层数据！
        # 如果这个 Tensor 被其他地方共享，可能会产生副作用。
        # 对于梯度累积，这通常是期望的行为。
        if self.dtype != other.dtype:
             # 或者尝试类型提升？暂时不允许不同类型原地相加
             raise TypeError(f"原地加法不支持不同数据类型: {self.dtype} 和 {other.dtype}")
        
        try:
            # 直接操作底层的 _data
            self._data += other.data # NumPy 的 += 操作
        except ValueError as e:
             # 捕获可能的广播错误
             raise ValueError(f"原地加法广播错误: {self.shape} += {other.shape}") from e

        # 原地操作不应改变 requires_grad 或 grad_fn
        # 返回 self 以支持链式操作 (虽然 += 通常不这么用)
        return self

    def copy(self) -> 'Tensor':
        """
        创建 Tensor 的一个副本。
        新 Tensor 具有相同的数据、dtype 和 requires_grad，
        但不包含梯度信息 (grad 和 _grad_fn 设为 None)。
        """
        # 创建底层 NumPy 数组的副本
        new_data = self._data.copy()
        # 创建新的 Tensor 实例
        # requires_grad 继承自原 Tensor，但 _grad_fn 和 grad 总是 None
        # is_leaf 取决于 requires_grad (如果 requires_grad=False，则为 True)
        new_tensor = Tensor(new_data, dtype=self.dtype, requires_grad=self.requires_grad,
                            _grad_fn=None, is_leaf=not self.requires_grad)
        # 确保 device 也被复制 (虽然目前只有 'cpu')
        new_tensor._device = self._device
        return new_tensor

    # --- Autograd 方法 ---

    def backward(self, gradient=None):
        """
        计算当前张量相对于图中叶子节点的梯度。

        Args:
            gradient (Tensor, optional): 对于非标量张量，需要提供初始梯度。
                                        对于标量张量，默认为 1.0。
        """
        if not self.requires_grad:
            raise RuntimeError("不能在 requires_grad=False 的张量上调用 backward()")

        if self.ndim != 0 and gradient is None:
            # 对于非标量输出，需要显式提供梯度
            raise RuntimeError("对于非标量张量，gradient 参数必须指定")
        elif self.ndim == 0 and gradient is None:
            # 标量输出的默认梯度是 1.0
            gradient = Tensor(1.0, dtype=self.dtype)
        elif gradient is not None:
            # 确保传入的 gradient 是 Tensor
            if not isinstance(gradient, Tensor):
                try:
                    gradient = Tensor(gradient, dtype=self.dtype)
                except TypeError:
                    raise TypeError(f"gradient 必须是 Tensor 或可以转换为 Tensor，但收到了 {type(gradient)}")
            if gradient.shape != self.shape:
                raise ValueError(f"gradient 形状必须匹配张量形状，期望 {self.shape}，得到 {gradient.shape}")

        # 如果调用 backward 的节点本身没有 grad_fn (即它是叶子节点或非浮点类型)，则无需操作
        if self._grad_fn is None:
            # 如果是叶子节点，梯度就是传入的 gradient (需要累加)
            if self.is_leaf:
                 if self.grad is None:
                      self.grad = gradient.copy()
                 else:
                      self.grad += gradient
            # else: 非叶子但无 grad_fn 的情况已在 __init__ 中阻止
            return

        # --- 实现拓扑排序和反向传播 ---

        # 1. 构建反向计算图并进行拓扑排序
        topo_order = []
        visited = set()
        def build_topo(node):
            # 只访问需要梯度的 Tensor 节点
            if isinstance(node, Tensor) and node.requires_grad and node not in visited:
                visited.add(node)
                if node._grad_fn: # 如果有关联的 Function，则递归访问其父节点
                    for parent in node._grad_fn.parents:
                        build_topo(parent)
                    topo_order.append(node) # 将当前节点添加到排序列表（后序遍历）

        build_topo(self)

        # 2. 初始化梯度字典，存储每个节点的累计梯度
        #    使用 id(Tensor) 作为 key
        node_grads = {id(t): None for t in visited} # 初始化所有访问过的节点梯度为 None
        node_grads[id(self)] = gradient # 设置起始节点的梯度

        # 3. 按照拓扑排序的逆序进行反向传播
        for node in reversed(topo_order):
            grad_fn = node._grad_fn
            # 叶子节点没有 grad_fn，在 build_topo 时不会加入 topo_order
            if grad_fn is None: continue

            # 获取当前节点的累计梯度
            current_grad = node_grads[id(node)]
            if current_grad is None:
                # 如果一个需要梯度的非叶子节点没有收到梯度，跳过
                # (可能后续路径不需要梯度，或者计算图有特殊结构)
                continue

            # 调用 Function 的 backward 方法计算输入的梯度
            input_grads = grad_fn.backward(grad_fn, current_grad) # grad_fn 实例作为 ctx
            if not isinstance(input_grads, tuple): input_grads = (input_grads,)

            # 将计算得到的梯度累加到父节点
            if len(grad_fn.parents) != len(input_grads):
                 raise RuntimeError(f"Function {grad_fn.__class__.__name__}.backward() 返回的梯度数量 "
                                    f"({len(input_grads)}) 与输入数量 ({len(grad_fn.parents)}) 不匹配")

            for parent_node, grad in zip(grad_fn.parents, input_grads):
                # 只处理需要梯度的 Tensor 父节点
                if isinstance(parent_node, Tensor) and parent_node.requires_grad:
                    if grad is not None: # Function 可能对某些输入返回 None 梯度
                        parent_id = id(parent_node)
                        # 确保 parent_node 在 visited 中 (理论上 build_topo 应该保证了)
                        if parent_id not in node_grads:
                             # 如果父节点未被访问但需要梯度，这是个问题
                             # 可能发生在父节点是叶子节点的情况，需要加入 visited
                             if parent_node.is_leaf and parent_node not in visited:
                                  visited.add(parent_node)
                                  node_grads[parent_id] = None # 初始化梯度
                             else:
                                  # 如果不是叶子节点但未访问，说明 build_topo 有问题
                                  raise RuntimeError(f"发现未访问的父节点 {parent_node}，计算图构建可能出错")

                        # 累加梯度到 node_grads 字典
                        if node_grads[parent_id] is None:
                            node_grads[parent_id] = grad.copy() # 第一次收到梯度，复制
                        else:
                            node_grads[parent_id] += grad # 使用原地加法累加

        # 4. 将最终计算出的梯度赋给叶子节点的 .grad 属性
        for node in visited:
             if node.is_leaf:
                  node_id = id(node)
                  if node_grads[node_id] is not None:
                       # 累加到叶子节点的 .grad (处理多次 backward 调用)
                       if node.grad is None:
                            node.grad = node_grads[node_id] # 第一次直接赋值 (已经是 copy)
                       else:
                            node.grad += node_grads[node_id]


# --- 一些辅助创建函数 ---

def tensor(data, dtype=None, requires_grad=False) -> Tensor:
    """工厂函数，用于创建 Tensor。"""
    return Tensor(data, dtype=dtype, requires_grad=requires_grad, is_leaf=True)

def zeros(shape, dtype=np.float32, requires_grad=False) -> Tensor:
    """创建全零 Tensor。"""
    return Tensor(np.zeros(shape, dtype=dtype), requires_grad=requires_grad, is_leaf=True)

def ones(shape, dtype=np.float32, requires_grad=False) -> Tensor:
    """创建全一 Tensor。"""
    return Tensor(np.ones(shape, dtype=dtype), requires_grad=requires_grad, is_leaf=True)

def randn(shape, dtype=np.float32, requires_grad=False) -> Tensor:
    """创建符合标准正态分布的 Tensor。"""
    return Tensor(np.random.randn(*shape).astype(dtype), requires_grad=requires_grad, is_leaf=True)

def empty(shape, dtype=np.float32, requires_grad=False) -> Tensor:
    """创建未初始化的 Tensor。"""
    # 注意：未初始化的 Tensor 用于 autograd 可能不安全
    return Tensor(np.empty(shape, dtype=dtype), requires_grad=requires_grad, is_leaf=True)

def from_numpy(ndarray: np.ndarray, requires_grad=False) -> Tensor:
    """从 NumPy 数组创建 Tensor (共享内存)。"""
    # 注意：这里直接使用 ndarray，修改 Tensor 会影响原 ndarray
    # 如果需要隔离，应该使用 .copy()
    return Tensor(ndarray, requires_grad=requires_grad, is_leaf=True)