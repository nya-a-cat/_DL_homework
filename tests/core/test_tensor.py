import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal

# 假设 tensor.py 在父目录的 my_nn_lib/core 下
# 为了让测试能找到模块，可能需要调整 Python 路径或使用更复杂的项目结构
# 简单起见，我们暂时假设可以直接导入
try:
    from my_nn_lib.core.tensor import Tensor, tensor, zeros, ones, randn, empty, from_numpy
except ImportError:
    # 如果直接运行 pytest tests/core/test_tensor.py 可能找不到
    # 尝试添加上层目录到 sys.path (不是最佳实践，但用于快速测试)
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from my_nn_lib.core.tensor import Tensor, tensor, zeros, ones, randn, empty, from_numpy


# --- 测试创建 ---

def test_tensor_creation_from_list():
    t = Tensor([[1, 2], [3, 4]])
    assert isinstance(t.data, np.ndarray)
    assert_equal(t.data, np.array([[1, 2], [3, 4]]))
    assert t.shape == (2, 2)
    assert t.dtype == np.array([[1, 2], [3, 4]]).dtype # 默认 dtype

def test_tensor_creation_from_numpy():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    t = Tensor(a)
    assert t.data is a # 默认应该共享内存
    assert t.shape == (3,)
    assert t.dtype == np.float32

def test_tensor_creation_from_numpy_with_dtype():
    a = np.array([1, 2, 3])
    t = Tensor(a, dtype=np.float32)
    assert t.data is not a # dtype 不同，应该创建副本
    assert t.shape == (3,)
    assert t.dtype == np.float32
    assert_allclose(t.data, np.array([1., 2., 3.], dtype=np.float32))

def test_tensor_creation_from_tensor():
    t1 = Tensor([1, 2, 3])
    t2 = Tensor(t1)
    assert t1.data is not t2.data # 默认创建副本
    assert_equal(t1.data, t2.data)
    assert t1.shape == t2.shape
    assert t1.dtype == t2.dtype

def test_tensor_creation_from_tensor_with_dtype():
    t1 = Tensor([1, 2, 3], dtype=np.int32)
    t2 = Tensor(t1, dtype=np.float64)
    assert t1.data is not t2.data
    assert t1.shape == t2.shape
    assert t1.dtype == np.int32
    assert t2.dtype == np.float64
    assert_allclose(t2.data, np.array([1., 2., 3.], dtype=np.float64))

def test_tensor_creation_invalid_type():
    with pytest.raises(TypeError):
        Tensor("invalid string")

# --- 测试辅助创建函数 ---

def test_zeros():
    t = zeros((2, 3), dtype=np.int8)
    assert t.shape == (2, 3)
    assert t.dtype == np.int8
    assert_equal(t.data, np.zeros((2, 3), dtype=np.int8))

def test_ones():
    t = ones((4,), dtype=np.float64)
    assert t.shape == (4,)
    assert t.dtype == np.float64
    assert_equal(t.data, np.ones((4,), dtype=np.float64))

def test_randn():
    # 仅测试形状和类型，值是随机的
    t = randn((5, 2), dtype=np.float32)
    assert t.shape == (5, 2)
    assert t.dtype == np.float32

def test_empty():
     # 仅测试形状和类型，值未定义
    t = empty((1, 1, 1), dtype=np.bool_)
    assert t.shape == (1, 1, 1)
    assert t.dtype == np.bool_

def test_from_numpy():
    a = np.arange(6).reshape((2, 3))
    t = from_numpy(a)
    assert t.data is a # 应该共享内存
    assert t.shape == (2, 3)
    assert t.dtype == a.dtype

def test_tensor_factory():
    t = tensor([1, 2])
    assert isinstance(t, Tensor)
    assert_equal(t.data, np.array([1, 2]))

# --- 测试属性 ---

def test_properties():
    data = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
    t = Tensor(data)
    assert t.shape == (2, 2)
    assert t.dtype == np.float32
    assert t.ndim == 2
    assert t.device == 'cpu'
    assert len(t) == 2

def test_len_0d():
    t = Tensor(5)
    assert t.ndim == 0
    with pytest.raises(TypeError):
        len(t)

# --- 测试形状操作 ---

def test_reshape():
    t1 = Tensor(np.arange(6))
    t2 = t1.reshape(2, 3)
    assert t2.shape == (2, 3)
    assert_equal(t2.data, np.arange(6).reshape((2, 3)))
    # NumPy reshape 可能返回视图或副本，这里不强求 is/is not

    t3 = t1.reshape((3, 2))
    assert t3.shape == (3, 2)

    t4 = t1.reshape(-1, 1)
    assert t4.shape == (6, 1)

def test_transpose():
    t1 = Tensor(np.arange(6).reshape((2, 3)))
    t2 = t1.transpose() # 反转所有维度
    assert t2.shape == (3, 2)
    assert_equal(t2.data, np.arange(6).reshape((2, 3)).T)

    t3 = t1.transpose(1, 0) # 指定维度顺序
    assert t3.shape == (3, 2)
    assert_equal(t3.data, np.arange(6).reshape((2, 3)).transpose(1, 0))

    t4 = Tensor(np.arange(24).reshape((2, 3, 4)))
    t5 = t4.transpose(0, 2, 1)
    assert t5.shape == (2, 4, 3)
    assert_equal(t5.data, np.arange(24).reshape((2, 3, 4)).transpose(0, 2, 1))


# --- 测试数学运算 ---

@pytest.fixture
def tensor_a():
    return Tensor([[1., 2.], [3., 4.]], dtype=np.float32)

@pytest.fixture
def tensor_b():
    return Tensor([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32)

@pytest.fixture
def scalar():
    return 2.0

def test_add_tensor(tensor_a, tensor_b):
    result = tensor_a + tensor_b
    expected = Tensor([[1.5, 3.5], [5.5, 7.5]], dtype=np.float32)
    assert isinstance(result, Tensor)
    assert result.dtype == expected.dtype
    assert_allclose(result.data, expected.data)

def test_add_scalar_right(tensor_a, scalar):
    result = tensor_a + scalar
    expected = Tensor([[3., 4.], [5., 6.]], dtype=np.float32)
    assert isinstance(result, Tensor)
    assert result.dtype == expected.dtype
    assert_allclose(result.data, expected.data)

def test_add_scalar_left(tensor_a, scalar):
    result = scalar + tensor_a
    expected = Tensor([[3., 4.], [5., 6.]], dtype=np.float32)
    assert isinstance(result, Tensor)
    assert result.dtype == expected.dtype
    assert_allclose(result.data, expected.data)

def test_sub_tensor(tensor_a, tensor_b):
    result = tensor_a - tensor_b
    expected = Tensor([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32)
    assert isinstance(result, Tensor)
    assert result.dtype == expected.dtype
    assert_allclose(result.data, expected.data)

def test_sub_scalar_right(tensor_a, scalar):
    result = tensor_a - scalar
    expected = Tensor([[-1., 0.], [1., 2.]], dtype=np.float32)
    assert isinstance(result, Tensor)
    assert result.dtype == expected.dtype
    assert_allclose(result.data, expected.data)

def test_sub_scalar_left(tensor_a, scalar):
    result = scalar - tensor_a
    expected = Tensor([[1., 0.], [-1., -2.]], dtype=np.float32)
    assert isinstance(result, Tensor)
    assert result.dtype == expected.dtype
    assert_allclose(result.data, expected.data)

def test_mul_tensor(tensor_a, tensor_b):
    result = tensor_a * tensor_b
    expected = Tensor([[0.5, 3.], [7.5, 14.]], dtype=np.float32)
    assert isinstance(result, Tensor)
    assert result.dtype == expected.dtype
    assert_allclose(result.data, expected.data)

def test_mul_scalar_right(tensor_a, scalar):
    result = tensor_a * scalar
    expected = Tensor([[2., 4.], [6., 8.]], dtype=np.float32)
    assert isinstance(result, Tensor)
    assert result.dtype == expected.dtype
    assert_allclose(result.data, expected.data)

def test_mul_scalar_left(tensor_a, scalar):
    result = scalar * tensor_a
    expected = Tensor([[2., 4.], [6., 8.]], dtype=np.float32)
    assert isinstance(result, Tensor)
    assert result.dtype == expected.dtype
    assert_allclose(result.data, expected.data)

def test_div_tensor(tensor_a, tensor_b):
    result = tensor_a / tensor_b
    expected_data = np.array([[1., 2.], [3., 4.]]) / np.array([[0.5, 1.5], [2.5, 3.5]])
    expected = Tensor(expected_data, dtype=np.float32)
    assert isinstance(result, Tensor)
    assert result.dtype == expected.dtype
    assert_allclose(result.data, expected.data)

def test_div_scalar_right(tensor_a, scalar):
    result = tensor_a / scalar
    expected = Tensor([[0.5, 1.], [1.5, 2.]], dtype=np.float32)
    assert isinstance(result, Tensor)
    assert result.dtype == expected.dtype
    assert_allclose(result.data, expected.data)

def test_div_scalar_left(tensor_a, scalar):
    result = scalar / tensor_a
    expected_data = 2.0 / np.array([[1., 2.], [3., 4.]])
    expected = Tensor(expected_data, dtype=np.float32)
    assert isinstance(result, Tensor)
    assert result.dtype == expected.dtype
    assert_allclose(result.data, expected.data)

def test_matmul_tensor(tensor_a, tensor_b):
    result = tensor_a @ tensor_b
    expected_data = np.matmul(tensor_a.data, tensor_b.data)
    expected = Tensor(expected_data, dtype=np.float32)
    assert isinstance(result, Tensor)
    assert result.shape == expected.shape
    assert result.dtype == expected.dtype
    assert_allclose(result.data, expected.data)

def test_matmul_vector():
    t1 = Tensor([1, 2, 3])
    t2 = Tensor([4, 5, 6])
    result = t1 @ t2 # Dot product
    expected = Tensor(np.dot(t1.data, t2.data))
    assert isinstance(result, Tensor)
    assert result.shape == () # Scalar result
    assert_allclose(result.data, expected.data)

    t3 = Tensor([[1, 2], [3, 4]])
    t4 = Tensor([5, 6])
    result = t3 @ t4 # Matrix-vector
    expected = Tensor(np.matmul(t3.data, t4.data))
    assert isinstance(result, Tensor)
    assert result.shape == (2,)
    assert_allclose(result.data, expected.data)

def test_matmul_incompatible_shapes():
    t1 = Tensor(np.zeros((2, 3)))
    t2 = Tensor(np.zeros((4, 5)))
    with pytest.raises(ValueError): # NumPy raises ValueError for matmul shape mismatch
        t1 @ t2

def test_matmul_0d():
    t1 = Tensor(5)
    t2 = Tensor(np.zeros((2,2)))
    with pytest.raises(ValueError):
        t1 @ t2
    with pytest.raises(ValueError):
        t2 @ t1

# --- 测试打印 ---
def test_repr(tensor_a):
    rep = repr(tensor_a)
    assert "Tensor" in rep
    assert "[[1. 2.]" in rep # NumPy default repr inside
    assert "[3. 4.]]" in rep
    assert f"dtype={tensor_a.dtype}" in rep
    assert f"device='{tensor_a.device}'" in rep

def test_str(tensor_a):
    s = str(tensor_a)
    # np.array_str output
    assert "[[1. 2.]" in s
    assert " [3. 4.]]" in s
    # Should not contain dtype or device
    assert "dtype" not in s
    assert "device" not in s