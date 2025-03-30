import numpy as np

# 假设 my_nn_lib 已经通过 pip install -e . 安装
# 或者需要调整 sys.path
try:
    from my_nn_lib.core import tensor
    import my_nn_lib.nn as nn
    import my_nn_lib.optim as optim
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from my_nn_lib.core import tensor
    import my_nn_lib.nn as nn
    import my_nn_lib.optim as optim

# --- 1. 准备数据 ---
# 目标函数: y = x^2
# 生成数据点
np.random.seed(42) # 为了可复现性
n_samples = 100
X_np = np.random.rand(n_samples, 1) * 10 - 5 # X 在 [-5, 5] 之间
y_np = X_np**2

# 转换为 Tensor
# 注意：需要 float 类型用于梯度计算
X = tensor(X_np.astype(np.float32))
y = tensor(y_np.astype(np.float32))

# --- 2. 定义模型 ---
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# --- 3. 实例化模型、损失和优化器 ---
input_size = 1
hidden_size = 10 # 隐藏层大小
output_size = 1

model = SimpleMLP(input_size, hidden_size, output_size)

# 使用 MSELoss，注意 mean reduction 尚未完全实现，使用 sum
# criterion = nn.MSELoss(reduction='mean') 
criterion = nn.MSELoss(reduction='sum') 

# 使用 SGD 优化器
learning_rate = 1e-5 # 可能需要调整
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# --- 4. 训练循环 ---
epochs = 200000

print("开始训练...")
for epoch in range(epochs):
    # --- 前向传播 ---
    y_pred = model(X)

    # --- 计算损失 ---
    loss = criterion(y_pred, y)

    # --- 梯度清零 ---
    optimizer.zero_grad()

    # --- 反向传播 ---
    loss.backward()

    # --- 参数更新 ---
    optimizer.step()

    # --- 打印损失 ---
    if (epoch + 1) % 100 == 0:
        # loss 是一个标量 Tensor，需要 .data 获取 NumPy 值
        loss_value = loss.data.item() # 获取标量值
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss_value:.4f}')

print("训练完成!")

# --- 5. (可选) 测试模型 ---
# 创建一些测试点
X_test_np = np.arange(-5, 5, 0.5).reshape(-1, 1).astype(np.float32)
X_test = tensor(X_test_np)

# 模型预测 (切换到评估模式，虽然我们还没有 Dropout/BN)
model.eval() 
with np.printoptions(precision=2): # 打印选项
    y_test_pred = model(X_test)
    print("\n测试输入:")
    print(X_test_np.flatten())
    print("模型预测:")
    print(y_test_pred.data.flatten())
    print("真实值 (x^2):")
    print((X_test_np**2).flatten())