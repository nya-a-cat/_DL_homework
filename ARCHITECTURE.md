# 神经网络库架构设计与开发计划

## 1. 目标

创建一个高性能、对标 PyTorch/TensorFlow 的神经网络库，用于实际项目，支持多种网络类型和 GPU 加速。

## 2. 核心架构

采用分层架构，包含以下核心模块：

1.  **底层计算引擎 (Backend Engine)**: 高效的、设备无关的张量操作。
2.  **自动微分引擎 (Autograd Engine)**: 自动计算梯度，实现反向传播。
3.  **神经网络层/模块 (NN)**: 构建神经网络的基本组件。
4.  **优化器 (Optim)**: 根据梯度更新模型参数。
5.  **数据加载与处理 (Data)**: 高效加载、预处理和组织数据。
6.  **高级接口与工具 (High-Level API & Utilities)**: 便捷接口和常用工具。

### 架构图 (Mermaid)

```mermaid
graph TD
    subgraph "用户接口 (User Interface)"
        A[模型定义 (Model Definition)]
        B[训练脚本 (Training Script)]
        C[推理应用 (Inference Application)]
    end

    subgraph "高级接口与工具 (High-Level API & Utilities)"
        HL1[模型保存/加载]
        HL2[训练循环辅助]
        HL3[分布式训练]
        HL4[设备管理]
        HL5[指标计算]
    end

    subgraph "核心库 (Core Library)"
        NN[神经网络层/模块 (NN)]
        Optim[优化器 (Optim)]
        Data[数据加载/处理 (Data)]
        Autograd[自动微分 (Autograd)]
    end

    subgraph "底层计算引擎 (Backend Engine)"
        BE1[张量操作 (Tensor Ops)]
        BE2[内存管理 (Memory Mgmt)]
        BE3[设备抽象 (Device Abstraction - CPU/GPU)]
        BE4[外部库集成 (BLAS, cuBLAS, cuDNN)]
    end

    A --> NN
    B --> NN
    B --> Optim
    B --> Data
    B --> Autograd
    B --> HL1
    B --> HL2
    B --> HL3
    B --> HL4
    B --> HL5
    C --> NN
    C --> HL1
    C --> HL4

    NN -- 使用 --> Autograd
    NN -- 包含 --> BE1
    Optim -- 使用 --> Autograd
    Optim -- 更新 --> NN(参数)
    Autograd -- 依赖 --> BE1
    Data -- 提供数据给 --> NN
    Data -- 可能使用 --> BE1(进行预处理)

    Autograd -- 计算梯度 --> Optim

    NN -- 运行在 --> BE3
    Optim -- 运行在 --> BE3
    Autograd -- 运行在 --> BE3
    Data -- 运行在 --> BE3

    BE1 -- 执行在 --> BE3
    BE2 -- 管理 --> BE3
    BE1 -- 可能调用 --> BE4

    HL1 -- 操作 --> NN
    HL2 -- 协调 --> NN & Optim & Data & Autograd
    HL3 -- 扩展 --> NN & Optim & Data
    HL4 -- 控制 --> BE3
    HL5 -- 评估 --> NN & Data
```

## 3. 核心模块细化设计

*   **底层计算引擎 (Backend Engine)**:
    *   `Tensor` 类: 包含 `data`, `shape`, `dtype`, `device`。提供创建、运算、形状、规约、索引、设备传输等方法。
    *   后端抽象: 根据设备分派 CPU/CUDA 实现。
    *   内存管理: 考虑内存池。
*   **自动微分引擎 (Autograd Engine)**:
    *   `Tensor` 扩展: `requires_grad`, `grad`。
    *   计算图: 动态图，记录 `Function` 和输入。
    *   `Function` 基类: `forward(ctx, *inputs)`, `backward(ctx, *grad_outputs)`。
    *   梯度计算入口: `tensor.backward()`。
*   **神经网络层/模块 (NN)**:
    *   `Module` 基类: `__init__`, `forward`, `parameters`, `to`, `train`/`eval`。
    *   `Parameter` 类: 继承 `Tensor`，`requires_grad=True`。
    *   关键层: `Linear`, `Conv2d`, `ReLU`, `Sequential` 等。
*   **优化器 (Optim)**:
    *   `Optimizer` 基类: `__init__`, `step`, `zero_grad`。
    *   具体实现: `SGD`, `Adam` 等。
*   **数据加载 (Data)**:
    *   `Dataset` 基类: `__len__`, `__getitem__`。
    *   `DataLoader`: `__init__`, 迭代返回批次数据。

## 4. 技术选型

*   **主要语言**: Python
*   **性能关键部分**: C++ / CUDA C++
*   **绑定**: `pybind11` 或 `Cython`
*   **计算图**: 动态计算图
*   **CPU 后端**: NumPy (初期), 后续考虑 Eigen/MKL/自研 C++
*   **GPU 后端**: CUDA, 集成 cuBLAS, cuDNN

## 5. 项目结构

```
my_nn_lib/
├── backend/
│   ├── cpu/
│   └── cuda/
├── core/
│   ├── autograd.py
│   ├── function.py
│   └── tensor.py
├── nn/
│   ├── functional.py
│   ├── init.py
│   ├── module.py
│   └── modules/
├── optim/
├── data/
├── utils/
└── _C/

csrc/
├── cpu/
├── cuda/
└── binding.cpp

tests/
examples/

setup.py
README.md
requirements.txt
```

## 6. 开发路线图 (Milestones)

*   **M1: 核心张量与 CPU 后端 (NumPy)**: 实现 `Tensor` 类基础，NumPy 后端，单元测试。
*   **M2: 动态计算图与自动微分**: 扩展 `Tensor`，实现 `Function` 基类和 `backward`，梯度检查。
*   **M3: 基础神经网络层与优化器**: 实现 `Module`, `Parameter`, `Linear`, `ReLU`, `SGD`, `MSELoss`，简单 MLP 示例。
*   **M4: GPU 支持 (CUDA) - 核心**: 设置 CUDA 环境，实现 CUDA 后端基础运算，`tensor.to('cuda')`，测试 GPU MLP。
*   **M5: 卷积层与数据加载**: 实现 `Conv2d`, `MaxPool2d` (CPU/CUDA)，`Dataset`, `DataLoader`，CNN 示例 (MNIST)。
*   **M6: 扩展与优化**: 更多层、优化器、激活函数、损失函数，优化 `DataLoader` 和内存管理。
*   **M7: 高级功能**: 模型保存/加载，学习率调度器，简单分布式，文档。