from ..core import Tensor

class Parameter(Tensor):
    """
    一个特殊的 Tensor，表示模型的可训练参数。

    参数默认 requires_grad=True。
    当它们被赋值给 Module 的属性时，会被自动注册。
    """
    def __new__(cls, data, requires_grad=True):
        # 使用 __new__ 是因为 Tensor 的 __init__ 已经比较复杂
        # 我们希望确保 Parameter 总是 requires_grad=True (除非显式指定为 False)
        # 并且它总是叶子节点
        
        # 调用 Tensor 的构造函数
        # 注意：我们不能直接调用 Tensor.__init__，因为 Tensor 可能不是这样设计的
        # 最安全的方式是创建一个普通的 Tensor，然后修改它的属性，或者确保 Tensor.__init__ 能处理
        
        # 方案1：创建 Tensor 并修改 (如果 Tensor 允许) - 不推荐，可能破坏 Tensor 内部状态
        # tensor = Tensor(data, requires_grad=requires_grad, is_leaf=True)
        # tensor.__class__ = cls # 强制改变类？危险！
        
        # 方案2：确保 Tensor.__init__ 可以被子类安全调用
        # 我们需要确保 Tensor 的 __init__ 不会覆盖 requires_grad
        # 目前 Tensor 的 __init__ 会根据 is_leaf 和 requires_grad 进行检查，应该还好
        
        # 直接调用 Tensor 的构造逻辑 (通过 super() 或直接调用 Tensor)
        # 这里我们假设 Tensor 的构造函数能正确处理
        # 我们需要确保 is_leaf 总是 True
        
        # 使用 Tensor 工厂函数创建基础 Tensor
        # 注意：工厂函数默认 is_leaf=True
        base_tensor = Tensor(data, dtype=None, requires_grad=requires_grad) 
        
        # 将其转换为 Parameter 对象 (共享数据)
        # 这是一种常用的技巧，类似于 PyTorch 的实现
        param = Tensor._make_subclass(cls, base_tensor.data, requires_grad)
        return param

    def __init__(self, data, requires_grad=True):
        # __init__ 主要用于确保 requires_grad 状态
        # __new__ 已经完成了对象的创建和大部分初始化
        # 我们可能不需要在这里做太多事情，但保留以备将来扩展
        # 确保 requires_grad 被设置 (虽然 __new__ 应该已经做了)
        # 注意：不能在这里再次调用 super().__init__，因为 __new__ 已经做了类似 Tensor 初始化的工作
        self.requires_grad = requires_grad
        # Parameter 总是叶子节点
        self._is_leaf = True 
        self._grad_fn = None # 叶子节点没有 grad_fn

    def __repr__(self):
        # 提供一个更清晰的表示
        return f"Parameter containing:\n{super().__repr__()}"

# 为了让 _make_subclass 工作，我们需要在 Tensor 中添加它
# 或者找到另一种方式来创建子类实例并共享数据

# 临时方案：在 Tensor 中添加 _make_subclass (需要修改 Tensor 类)
# 或者，Parameter 直接持有 Tensor (组合优于继承？)

# 重新考虑：Parameter 作为 Tensor 的简单封装可能更好？
# class Parameter:
#     def __init__(self, data, requires_grad=True):
#         self.tensor = Tensor(data, requires_grad=requires_grad, is_leaf=True)
#     # ... 代理 Tensor 的属性和方法 ...

# 暂时坚持继承方案，但需要在 Tensor 中添加辅助方法
# 我们将在下一步修改 Tensor