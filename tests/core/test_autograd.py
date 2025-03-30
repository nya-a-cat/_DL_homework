import pytest
import numpy as np
from numpy.testing import assert_allclose

# 假设可以导入
try:
    from my_nn_lib.core import Tensor, tensor
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from my_nn_lib.core import Tensor, tensor

# --- 辅助函数：数值梯度检查 ---

def numerical_gradient(f, target_tensor: Tensor, *other_tensors: Tensor, eps=1e-6):
    """
    计算函数 f 相对于 target_tensor 的数值梯度。
    f 应该接受 target_tensor 和 other_tensors 作为输入，并返回一个标量值。
    """
    target_data = target_tensor.data.copy() # 使用副本以避免修改原始 Tensor
    grad = np.zeros_like(target_data)
    it = np.nditer(target_data, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        original_value = target_data[idx]

        # 计算 f(target + eps)
        target_data[idx] = original_value + eps
        # 创建一个新的 Tensor 包含扰动后的数据，但保持 requires_grad 等状态不变
        # 注意：这里创建的是叶子节点，计算图不会被构建，这是数值梯度的预期行为
        perturbed_tensor_plus = Tensor(target_data, dtype=target_tensor.dtype)
        fx_plus_eps = f(perturbed_tensor_plus, *other_tensors) # 传递扰动后的和其他未扰动的

        # 计算 f(target - eps)
        target_data[idx] = original_value - eps
        perturbed_tensor_minus = Tensor(target_data, dtype=target_tensor.dtype)
        fx_minus_eps = f(perturbed_tensor_minus, *other_tensors)

        # 计算中心差分梯度
        grad[idx] = (fx_plus_eps - fx_minus_eps) / (2 * eps)

        # 恢复原始值以便下次迭代
        target_data[idx] = original_value
        it.iternext()

    return grad

# --- 测试 Autograd ---

def test_add_backward():
    x = tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = tensor([4.0, 5.0, 6.0], requires_grad=True)
    z = x + y
    
    # 假设最终损失是 z 中所有元素的和
    loss = z.sum() # 需要实现 Sum 操作的 Function
    # 暂时手动模拟 loss.backward()
    # loss = z1 + z2 + z3
    # dloss/dz1 = 1, dloss/dz2 = 1, dloss/dz3 = 1
    # 所以 z 的梯度是 [1, 1, 1]
    z_grad = tensor([1.0, 1.0, 1.0])
    
    # 手动调用 Add 的 backward (通常由 backward() 自动完成)
    add_func = z._grad_fn 
    assert add_func is not None
    dx, dy = add_func.backward(add_func, z_grad) # ctx is the function instance

    assert_allclose(dx.data, np.array([1.0, 1.0, 1.0]))
    assert_allclose(dy.data, np.array([1.0, 1.0, 1.0]))

    # --- 现在测试完整的 backward 调用 ---
    # 需要实现 Sum Function 才能自动计算 loss.backward()
    # 暂时跳过自动 backward 测试，直到 Sum 实现 # 已实现 Sum，移除 skip

# @pytest.mark.skip(reason="需要实现 Sum Function 才能运行完整的 backward") # 取消 skip
def test_add_backward_auto():
    x = tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = tensor([4.0, 5.0, 6.0], requires_grad=True)
    z = x + y
    loss = z.sum() # 假设 Sum 已实现
    loss.backward()

    assert x.grad is not None
    assert y.grad is not None
    assert_allclose(x.grad.data, np.array([1.0, 1.0, 1.0]))
    assert_allclose(y.grad.data, np.array([1.0, 1.0, 1.0]))


def test_mul_backward():
    x_data = np.array([1.0, 2.0, 3.0])
    y_data = np.array([4.0, 5.0, 6.0])
    x = tensor(x_data, requires_grad=True)
    y = tensor(y_data, requires_grad=True)
    z = x * y # z = [4, 10, 18]

    # 假设最终损失是 z 中所有元素的和
    # loss = z1 + z2 + z3
    # z 的梯度是 [1, 1, 1]
    z_grad = tensor([1.0, 1.0, 1.0])

    # 手动调用 Mul 的 backward
    mul_func = z._grad_fn
    assert mul_func is not None
    dx, dy = mul_func.backward(mul_func, z_grad)

    # dL/dx = dL/dz * dz/dx = z_grad * y
    # dL/dy = dL/dz * dz/dy = z_grad * x
    assert_allclose(dx.data, z_grad.data * y_data) # [1*4, 1*5, 1*6] = [4, 5, 6]
    assert_allclose(dy.data, z_grad.data * x_data) # [1*1, 1*2, 1*3] = [1, 2, 3]

# @pytest.mark.skip(reason="需要实现 Sum Function 才能运行完整的 backward") # 取消 skip
def test_mul_backward_auto():
    x = tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = tensor([4.0, 5.0, 6.0], requires_grad=True)
    z = x * y
    loss = z.sum() # 假设 Sum 已实现
    loss.backward()

    assert x.grad is not None
    assert y.grad is not None
    assert_allclose(x.grad.data, y.data) # [4, 5, 6]
    assert_allclose(y.grad.data, x.data) # [1, 2, 3]


def test_chain_rule():
    a = tensor(2.0, requires_grad=True)
    b = tensor(3.0, requires_grad=True)
    c = a * b # c = 6.0, grad_fn=Mul
    d = tensor(4.0, requires_grad=True)
    e = c + d # e = 10.0, grad_fn=Add

    # 手动反向传播 e 对 a, b, d 的梯度 (假设 de/de = 1)
    e_grad = tensor(1.0)

    # e = c + d => dc = de * 1 = 1.0, dd = de * 1 = 1.0
    add_func = e._grad_fn
    dc, dd = add_func.backward(add_func, e_grad)

    # c = a * b => da = dc * b = 1.0 * 3.0 = 3.0
    #             db = dc * a = 1.0 * 2.0 = 2.0
    mul_func = c._grad_fn
    da, db = mul_func.backward(mul_func, dc)

    assert_allclose(da.data, 3.0)
    assert_allclose(db.data, 2.0)
    assert_allclose(dd.data, 1.0)

# @pytest.mark.skip(reason="需要实现完整的 backward 逻辑才能运行") # 取消 skip
def test_chain_rule_auto():
    a = tensor(2.0, requires_grad=True)
    b = tensor(3.0, requires_grad=True)
    c = a * b
    d = tensor(4.0, requires_grad=True)
    e = c + d
    e.backward() # de/de = 1

    assert a.grad is not None
    assert b.grad is not None
    assert d.grad is not None
    assert_allclose(a.grad.data, 3.0) # de/da = de/dc * dc/da = 1 * b = 3.0
    assert_allclose(b.grad.data, 2.0) # de/db = de/dc * dc/db = 1 * a = 2.0
    assert_allclose(d.grad.data, 1.0) # de/dd = 1.0


def test_gradient_accumulation():
    x = tensor([1.0, 2.0], requires_grad=True)
    y1 = x * 2.0 # y1 = [2, 4]
    y2 = x * 3.0 # y2 = [3, 6]
    
    # 模拟两次 backward
    # 第一次: loss1 = y1.sum() => dloss1/dx = [2, 2]
    y1_grad = tensor([1.0, 1.0])
    mul_func1 = y1._grad_fn
    dx1, _ = mul_func1.backward(mul_func1, y1_grad) # dx1 = [1*2, 1*2] = [2, 2]

    # 第二次: loss2 = y2.sum() => dloss2/dx = [3, 3]
    y2_grad = tensor([1.0, 1.0])
    mul_func2 = y2._grad_fn
    dx2, _ = mul_func2.backward(mul_func2, y2_grad) # dx2 = [1*3, 1*3] = [3, 3]

    # 手动累加梯度
    if x.grad is None:
        x.grad = dx1
    else:
        x.grad += dx1 # 使用 +=

    if x.grad is None: # 理论上不会发生
        x.grad = dx2
    else:
        x.grad += dx2 # 使用 +=

    assert x.grad is not None
    assert_allclose(x.grad.data, np.array([5.0, 5.0])) # [2, 2] + [3, 3]

# @pytest.mark.skip(reason="需要实现完整的 backward 逻辑才能运行") # 取消 skip
def test_gradient_accumulation_auto():
    x = tensor([1.0, 2.0], requires_grad=True)
    y1 = x * 2.0
    y2 = x * 3.0
    
    loss1 = y1.sum() # 假设 Sum 已实现
    loss1.backward() # x.grad 变为 [2, 2]

    loss2 = y2.sum() # 假设 Sum 已实现
    loss2.backward() # x.grad 变为 [2, 2] + [3, 3] = [5, 5]

    assert x.grad is not None
    assert_allclose(x.grad.data, np.array([5.0, 5.0]))


# --- 梯度检查 ---

# @pytest.mark.skip(reason="需要实现 Sum 和完整的 backward 逻辑") # 取消 skip
def test_gradcheck_add():
    x = tensor(np.random.rand(2, 3), requires_grad=True)
    y = tensor(np.random.rand(2, 3), requires_grad=True)
    
    def func(inp):
        # inp 是 x 或 y
        # 需要根据 inp 是哪个来重新计算 z
        if np.array_equal(inp.data, x.data):
             z = inp + y
        else:
             z = x + inp
        # 返回标量 Python number
        loss = z.sum()
        return loss.data.item()

    # 计算 x 的数值梯度
    num_grad_x = numerical_gradient(func, x)
    # 计算 y 的数值梯度
    num_grad_y = numerical_gradient(func, y)

    # 计算解析梯度
    z = x + y
    loss = z.sum()
    loss.backward()

    assert x.grad is not None
    assert y.grad is not None
    assert_allclose(x.grad.data, num_grad_x, rtol=1e-4, atol=1e-5)
    assert_allclose(y.grad.data, num_grad_y, rtol=1e-4, atol=1e-5)


# @pytest.mark.skip(reason="需要实现 Sum 和完整的 backward 逻辑") # 取消 skip
def test_gradcheck_mul():
    # 使用 float64 以提高梯度检查的精度
    x_data = (np.random.rand(2, 3) + 0.1).astype(np.float64)
    y_data = (np.random.rand(2, 3) + 0.1).astype(np.float64)
    x = tensor(x_data, requires_grad=True)
    y = tensor(y_data, requires_grad=True)

    # 定义函数 f(input_x, input_y)，返回标量损失
    def func_for_mul(input_x: Tensor, input_y: Tensor):
        # 确保输入是 float64 (numerical_gradient 会创建新的 Tensor)
        if input_x.dtype != np.float64: input_x = Tensor(input_x.data, dtype=np.float64)
        if input_y.dtype != np.float64: input_y = Tensor(input_y.data, dtype=np.float64)
        
        # 使用传入的 input_x 和 input_y 进行计算
        z = input_x * input_y
        # 返回标量值
        loss = z.sum()
        return loss.data.item() # 返回标量 Python number

    # 计算 x 的数值梯度，此时 y 是 other_tensors
    num_grad_x = numerical_gradient(func_for_mul, x, y)
    # 计算 y 的数值梯度，此时 x 是 other_tensors
    num_grad_y = numerical_gradient(func_for_mul, y, x)

    # 重置解析梯度
    x.grad = None
    y.grad = None
    
    z = x * y
    loss = z.sum()
    loss.backward()

    assert x.grad is not None
    assert y.grad is not None
    print("\nGradcheck Mul:")
    print("  Analytical grad x:", x.grad.data)
    print("  Numerical grad x: ", num_grad_x)
    print("  Analytical grad y:", y.grad.data)
    print("  Numerical grad y: ", num_grad_y)
    # 使用更严格的容差进行比较 (适用于 float64)
    assert_allclose(x.grad.data, num_grad_x, rtol=1e-5, atol=1e-6)
    assert_allclose(y.grad.data, num_grad_y, rtol=1e-5, atol=1e-6)

# TODO: 添加更多测试，特别是涉及广播的梯度检查
