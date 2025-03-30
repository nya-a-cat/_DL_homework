# 让 core 成为一个子包
# 可以选择性地将 Tensor 类暴露到 my_nn_lib.core 命名空间
from .tensor import Tensor, tensor, zeros, ones, randn, empty, from_numpy