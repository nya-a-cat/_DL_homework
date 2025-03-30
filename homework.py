import numpy as np
import matplotlib.pyplot as plt

# ==================================================
# 1. Explanation of Optimizers (Textual Answer)
# ==================================================
print("="*20 + "\n1. Optimizer Explanations\n" + "="*20)
print("""
**SGD (Stochastic Gradient Descent):**
- **Principle:** Updates parameters in the opposite direction of the gradient of the loss function, calculated on a small batch (or single example) of data.
- **Update Rule:** `param = param - learning_rate * gradient`
- **Pros:** Simple, computationally less expensive per update than batch gradient descent.
- **Cons:** Can have high variance in updates, leading to noisy convergence; may get stuck in local minima or oscillate; requires careful tuning of the learning rate.

**Momentum:**
- **Principle:** Accelerates SGD by adding a fraction (momentum term, gamma, typically ~0.9) of the previous update vector to the current gradient step. This helps overcome small local minima/saddles and dampens oscillations in directions with consistent gradients.
- **Update Rule:**
    `velocity = momentum * velocity - learning_rate * gradient`
    `param = param + velocity`
- **Pros:** Faster convergence than basic SGD, reduced oscillation.
- **Cons:** Introduces another hyperparameter (momentum).

**AdaGrad (Adaptive Gradient Algorithm):**
- **Principle:** Adapts the learning rate for each parameter individually, making smaller updates for parameters associated with frequently occurring features and larger updates for parameters associated with infrequent features. It accumulates the *square* of past gradients.
- **Update Rule:**
    `cache += gradient**2`
    `param = param - learning_rate * gradient / (sqrt(cache) + epsilon)`
- **Pros:** Eliminates the need to manually tune the learning rate as much; works well for sparse data.
- **Cons:** The learning rate can decay too aggressively and become infinitesimally small, effectively stopping learning prematurely.

**Adam (Adaptive Moment Estimation):**
- **Principle:** Combines ideas from Momentum (uses moving average of the gradient - first moment) and RMSprop/AdaGrad (uses moving average of the squared gradient - second moment). It also includes bias correction for these moving averages, especially important during the initial steps.
- **Update Rule (Simplified):**
    `m = beta1 * m + (1 - beta1) * gradient` (Update biased first moment estimate)
    `v = beta2 * v + (1 - beta2) * (gradient**2)` (Update biased second moment estimate)
    `m_hat = m / (1 - beta1**t)` (Bias-corrected first moment)
    `v_hat = v / (1 - beta2**t)` (Bias-corrected second moment)
    `param = param - learning_rate * m_hat / (sqrt(v_hat) + epsilon)`
    (where t is the timestep)
- **Pros:** Generally performs well across a wide range of problems, combines benefits of Momentum and adaptive learning rates, often converges quickly.
- **Cons:** More complex, introduces more hyperparameters (beta1, beta2, epsilon), although default values often work well.

**Key Differences Summary:**
- **SGD:** Basic gradient update.
- **Momentum:** Adds velocity based on past updates.
- **AdaGrad:** Parameter-specific learning rates based on accumulated *squared* gradients (learning rate monotonically decreases).
- **Adam:** Parameter-specific learning rates based on *moving averages* of both the gradient and its square, with bias correction.
""")

# ==================================================
# 2. Simple SGD Optimizer Class
# ==================================================
print("\n" + "="*20 + "\n2. SGD Optimizer Implementation\n" + "="*20)

class SGD:
    """
    Simple Stochastic Gradient Descent (SGD) optimizer.
    """
    def __init__(self, lr=0.01):
        """
        Initializes the SGD optimizer.

        Args:
            lr (float): Learning rate. Default is 0.01.
        """
        self.lr = lr
        print(f"SGD Optimizer initialized with learning rate: {self.lr}")

    def update(self, params, grads):
        """
        Updates parameters using the SGD rule.

        Args:
            params (dict): Dictionary containing model parameters (e.g., {'W1': W1, 'b1': b1, ...}).
                           Values are NumPy arrays.
            grads (dict): Dictionary containing gradients for the parameters
                          (e.g., {'W1': dW1, 'b1': db1, ...}). Values are NumPy arrays.
        """
        for key in params.keys():
            # Ensure gradient exists for the parameter
            if key in grads:
                params[key] -= self.lr * grads[key]
            else:
                print(f"Warning: Gradient for parameter '{key}' not found in grads dict.")

# Example Usage (Conceptual)
# params_example = {'W': np.random.randn(10, 5), 'b': np.zeros(5)}
# grads_example = {'W': np.random.randn(10, 5) * 0.1, 'b': np.random.randn(5) * 0.1}
# sgd_optimizer = SGD(lr=0.1)
# print("Params before update (first 2 elements):", params_example['W'][0, :2])
# sgd_optimizer.update(params_example, grads_example)
# print("Params after update (first 2 elements):", params_example['W'][0, :2])
# print("---")

# ==================================================
# 3. Simple Dropout Layer Class
# ==================================================
print("\n" + "="*20 + "\n3. Dropout Layer Implementation\n" + "="*20)

class Dropout:
    """
    Simple Dropout layer.
    """
    def __init__(self, dropout_ratio=0.5):
        """
        Initializes the Dropout layer.

        Args:
            dropout_ratio (float): Probability of dropping out a neuron (setting its output to zero).
                                   Must be between 0 and 1. Default is 0.5.
        """
        if not 0 <= dropout_ratio <= 1:
            raise ValueError("dropout_ratio must be between 0 and 1.")
        self.dropout_ratio = dropout_ratio
        self.mask = None # Mask used during forward pass (needed for backward pass if implemented)
        print(f"Dropout Layer initialized with dropout_ratio: {self.dropout_ratio}")

    def forward(self, x, train_flg=True):
        """
        Performs the forward pass for the Dropout layer.

        Args:
            x (np.ndarray): Input data.
            train_flg (bool): Flag indicating whether the model is in training mode (True)
                              or inference/test mode (False). Default is True.

        Returns:
            np.ndarray: Output data after applying dropout (if training) or scaling (if testing).
        """
        if train_flg:
            # Training mode: Apply dropout mask
            # Generate random numbers, compare with dropout_ratio.
            # Keep neurons where random number > dropout_ratio (probability 1 - dropout_ratio)
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            # Apply mask (neurons where mask is False become 0)
            # Note: Inverted dropout scales the output during training,
            # but the prompt asks for scaling during testing, so we'll follow that.
            return x * self.mask
        else:
            # Inference/Test mode: Scale the output
            # Multiply by (1 - dropout_ratio) to keep the expected output scale
            # consistent with the training phase (where only a fraction of neurons were active).
            return x * (1.0 - self.dropout_ratio)

# Example Usage (Conceptual)
# dropout_layer = Dropout(dropout_ratio=0.3)
# input_data = np.random.randn(3, 4)
# print("Input data:\n", input_data)
# output_train = dropout_layer.forward(input_data, train_flg=True)
# print("Output (train mode):\n", output_train)
# print("Mask used:\n", dropout_layer.mask)
# output_test = dropout_layer.forward(input_data, train_flg=False)
# print("Output (test mode):\n", output_test)
# print("---")

# ==================================================
# 4. Batch Normalization Explanation (Textual Answer)
# ==================================================
print("\n" + "="*20 + "\n4. Batch Normalization Explanation\n" + "="*20)
print("""
**Purpose of Batch Normalization (BN):**
Batch Normalization is a technique used to improve the speed, performance, and stability of training deep neural networks. Its main goals are:
1.  **Accelerate Training:** Allows for higher learning rates and faster convergence.
2.  **Stabilize Training:** Reduces the sensitivity to parameter initialization and makes training more robust.
3.  **Regularization:** Acts as a slight regularizer, sometimes reducing the need for other regularization techniques like Dropout.

**Internal Covariate Shift (ICS):**
ICS refers to the phenomenon where the distribution of activations (inputs) for a specific layer changes during the training process. This happens because the parameters of the preceding layers are constantly being updated. Each layer must continually adapt to these changing input distributions, which slows down the learning process, similar to trying to hit a moving target.

**How Batch Normalization Addresses ICS:**
BN tackles ICS by normalizing the activations within each mini-batch during training. For a given layer's input (pre-activation):
1.  **Calculate Mini-Batch Statistics:** It computes the mean and variance of the activations across the current mini-batch.
2.  **Normalize:** It subtracts the mini-batch mean and divides by the mini-batch standard deviation (plus a small epsilon for numerical stability). This step ensures that the input to the activation function (within that layer, for that mini-batch) has approximately zero mean and unit variance.
3.  **Scale and Shift:** Since forcing activations to always have zero mean and unit variance might limit the network's representational power (e.g., a sigmoid might need inputs outside this range), BN introduces two learnable parameters per feature/channel:
    *   `gamma` (scale): Multiplies the normalized activation.
    *   `beta` (shift): Adds to the scaled, normalized activation.
    These parameters allow the network to learn the optimal scale and mean for the activations, potentially even undoing the normalization if that's beneficial for that layer.

By stabilizing the distribution of layer inputs (reducing ICS), BN allows subsequent layers to learn more effectively and enables the use of higher learning rates without causing divergence or gradient explosion/vanishing. During inference (testing), BN uses aggregated statistics (running averages of mean and variance calculated during training) instead of mini-batch statistics to ensure deterministic output.
""")

# ==================================================
# 5. Simple Batch Normalization Layer Class
# ==================================================
print("\n" + "="*20 + "\n5. Batch Normalization Implementation\n" + "="*20)

class BatchNormalization:
    """
    Simple Batch Normalization layer.
    Assumes input 'x' has shape (N, D) where N is batch size, D is feature dimension.
    Normalization is performed over the batch dimension (N).
    gamma and beta have shape (D,).
    """
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None, eps=1e-5):
        """
        Initializes the Batch Normalization layer.

        Args:
            gamma (np.ndarray): Scale parameter (learnable), shape (D,).
            beta (np.ndarray): Shift parameter (learnable), shape (D,).
            momentum (float): Momentum for updating running mean and variance. Default 0.9.
            running_mean (np.ndarray): Running average of mean, shape (D,). Usually initialized to zeros.
            running_var (np.ndarray): Running average of variance, shape (D,). Usually initialized to ones or zeros.
            eps (float): Small constant added to variance for numerical stability. Default 1e-5.
        """
        self.gamma = gamma # Scale parameter (learnable)
        self.beta = beta   # Shift parameter (learnable)
        self.momentum = momentum
        self.input_shape = None # Store input shape for potential backward pass

        # Running mean and variance for inference mode
        self.running_mean = running_mean
        self.running_var = running_var

        self.eps = eps
        self.batch_size = None
        self.xc = None # Input centered around zero
        self.std = None # Standard deviation used for normalization

        print(f"BatchNormalization Layer initialized with momentum={momentum}, eps={eps}")
        # Initialize running stats if not provided (based on gamma/beta shape)
        D = gamma.shape[0]
        if self.running_mean is None:
            self.running_mean = np.zeros(D, dtype=np.float64)
            print("Initialized running_mean to zeros.")
        if self.running_var is None:
            self.running_var = np.zeros(D, dtype=np.float64) # Usually starts at 0 or 1, 0 seems common
            print("Initialized running_var to zeros.")


    def forward(self, x, train_flg=True):
        """
        Performs the forward pass for Batch Normalization.

        Args:
            x (np.ndarray): Input data, expected shape (N, D).
            train_flg (bool): Flag indicating training (True) or inference (False) mode.

        Returns:
            np.ndarray: Output data after Batch Normalization.
        """
        self.input_shape = x.shape
        if x.ndim != 2:
            # Reshape for convolutional layers if needed, but sticking to (N, D) based on prompt
             # For now, raise error if not (N, D)
             # N, C, H, W = x.shape
             # x = x.reshape(N, -1)
             print(f"Warning: Input shape {x.shape} not (N, D). Assuming (N=batch_size, D=features).")
             if x.ndim > 2: # Simple attempt to flatten features
                  x = x.reshape(x.shape[0], -1)
                  # Note: Proper BN for Conv layers normalizes per channel (N, C, H, W) -> stats over (N, H, W) for each C.
                  # This implementation normalizes over N for all D=C*H*W features combined if flattened.
                  # For the prompt's simplicity, we'll proceed assuming D matches gamma/beta.


        if self.running_mean is not None and x.shape[1] != self.running_mean.shape[0]:
             raise ValueError(f"Input feature dimension {x.shape[1]} doesn't match running_mean dimension {self.running_mean.shape[0]}")


        if train_flg:
            # Training mode: Use mini-batch statistics
            mu = np.mean(x, axis=0) # Mini-batch mean (shape D)
            xc = x - mu             # Center data
            var = np.mean(xc**2, axis=0) # Mini-batch variance (shape D)
            std = np.sqrt(var + self.eps) # Mini-batch standard deviation
            xn = xc / std           # Normalize

            self.batch_size = x.shape[0]
            self.xc = xc
            self.std = std
            self.var = var # Store variance for potential backward pass

            # Update running averages (using population statistics formula is slightly different,
            # but often moving average is used like this)
            # Using simple moving average:
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
            # Note: For unbiased variance estimate correction in backward pass, Bessel's correction might be needed.

            # Scale and shift
            out = self.gamma * xn + self.beta

        else:
            # Inference mode: Use running averages
            # Check if running averages have been computed
            # if self.running_mean is None or self.running_var is None:
            #    raise Exception("Running mean/variance not computed. Run in training mode first.")
            # Use stored running mean and variance estimates
            xc = x - self.running_mean
            # Use running variance for std deviation
            # Note: running_var is variance, not std dev
            std_inf = np.sqrt(self.running_var + self.eps)
            xn = xc / std_inf

            # Scale and shift
            out = self.gamma * xn + self.beta

        # Reshape back if input was flattened Conv layer (not implemented fully here)
        # out = out.reshape(*self.input_shape) if x.ndim != 2 else out
        return out

# Example Usage (Conceptual)
# D_features = 4
# gamma_init = np.ones(D_features)
# beta_init = np.zeros(D_features)
# bn_layer = BatchNormalization(gamma=gamma_init, beta=beta_init)
# input_data_bn = np.random.randn(100, D_features) + 5 # Batch of 100, 4 features, shifted mean
#
# # Training pass
# print("Input mean (axis 0):", np.mean(input_data_bn, axis=0))
# output_train_bn = bn_layer.forward(input_data_bn, train_flg=True)
# print("Output mean (train, axis 0):", np.mean(output_train_bn, axis=0)) # Should be close to beta (0)
# print("Output var (train, axis 0):", np.var(output_train_bn, axis=0)) # Should be close to gamma^2 (1)
# print("Running mean after train:", bn_layer.running_mean)
# print("Running var after train:", bn_layer.running_var)
#
# # Testing pass with new data
# input_data_test_bn = np.random.randn(20, D_features) + 5 # Different batch size and data
# print("\nTest Input mean (axis 0):", np.mean(input_data_test_bn, axis=0))
# output_test_bn = bn_layer.forward(input_data_test_bn, train_flg=False)
# print("Output mean (test, axis 0):", np.mean(output_test_bn, axis=0)) # Should use running stats, closer to 0
# print("Output var (test, axis 0):", np.var(output_test_bn, axis=0))   # Should use running stats, closer to 1
# print("---")


# ==================================================
# 6. Learning Rate Comparison Experiment
# ==================================================
print("\n" + "="*20 + "\n6. Learning Rate Experiment\n" + "="*20)

# --- Helper Functions & Basic Layers ---
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    # Subtract max for numerical stability
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def cross_entropy_error(y, t):
    """
    Compute cross entropy error.
    y: predictions (output of softmax), shape (N, C)
    t: true labels (one-hot encoded), shape (N, C)
    """
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # Handle case where t is label indices instead of one-hot
    if t.size == y.shape[0]:
         t = np.eye(y.shape[1])[t] # Convert labels to one-hot

    # Batch size
    batch_size = y.shape[0]
    # Add small epsilon for numerical stability (to avoid log(0))
    delta = 1e-7
    return -np.sum(t * np.log(y + delta)) / batch_size

class Affine:
    """Simple Affine (fully connected) layer."""
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.original_x_shape = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.original_x_shape = x.shape
        # Flatten input if it's multi-dimensional (e.g., from Conv layer)
        self.x = x.reshape(x.shape[0], -1)
        out = np.dot(self.x, self.W) + self.b
        return out

    def backward(self, dout):
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        dx = np.dot(dout, self.W.T)
        # Reshape dx to original input shape
        dx = dx.reshape(*self.original_x_shape)
        return dx

class Relu:
    """ReLU activation function layer."""
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

class SoftmaxWithLoss:
     """Softmax activation combined with Cross-Entropy Loss."""
     def __init__(self):
        self.loss = None
        self.y = None # output of softmax
        self.t = None # true labels (one-hot)

     def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

     def backward(self, dout=1):
        batch_size = self.t.shape[0]
        # Handle case where t is label indices
        if self.t.size == self.y.size: # if t is already one-hot
            dx = (self.y - self.t) / batch_size
        else: # if t contains indices
             dx = self.y.copy()
             dx[np.arange(batch_size), self.t] -= 1
             dx = dx / batch_size
        return dx

# --- TwoLayerNet Definition ---
class TwoLayerNet:
    """
    A simple two-layer neural network (Input - Affine - ReLU - Affine - Softmax).
    """
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        """
        Initializes the network.

        Args:
            input_size (int): Size of the input layer.
            hidden_size (int): Size of the hidden layer.
            output_size (int): Size of the output layer.
            weight_init_std (float): Standard deviation for weight initialization.
        """
        # Initialize weights and biases
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # Create layers
        self.layers = {
            'Affine1': Affine(self.params['W1'], self.params['b1']),
            'Relu1': Relu(),
            'Affine2': Affine(self.params['W2'], self.params['b2'])
        }
        # Last layer handles Softmax and Loss calculation
        self.lastLayer = SoftmaxWithLoss()

        print(f"TwoLayerNet initialized: Input({input_size}) -> Hidden({hidden_size}) -> Output({output_size})")


    def predict(self, x):
        """
        Performs forward pass to get predictions (scores before softmax).
        """
        x_in = x
        for layer_name in ['Affine1', 'Relu1', 'Affine2']:
            x_in = self.layers[layer_name].forward(x_in)
        return x_in

    def loss(self, x, t):
        """
        Calculates the loss for given data and labels.
        """
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        """
        Calculates the accuracy.
        t can be indices or one-hot vectors.
        """
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: # If t is one-hot encoded
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        """
        Calculates gradients using backpropagation.
        """
        # Forward pass
        self.loss(x, t)

        # Backward pass
        dout = 1
        dout = self.lastLayer.backward(dout) # Gradient from SoftmaxWithLoss

        layers_list = ['Affine2', 'Relu1', 'Affine1']
        for layer_name in layers_list:
            dout = self.layers[layer_name].backward(dout)

        # Store gradients
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads


# --- Generate Synthetic Data ---
# Example: Simple classification task
np.random.seed(42) # for reproducibility
N = 100 # number of data points per class
D = 2   # dimensionality
K = 3   # number of classes
X = np.zeros((N*K, D)) # data matrix (each row = single example)
y_indices = np.zeros(N*K, dtype='uint8') # class labels (indices)
for j in range(K):
    ix = range(N*j, N*(j+1))
    r = np.linspace(0.0, 1, N) # radius
    theta = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N)*0.2 # theta
    X[ix] = np.c_[r*np.sin(theta), r*np.cos(theta)]
    y_indices[ix] = j

# Convert labels to one-hot encoding
y_one_hot = np.eye(K)[y_indices]

print(f"Generated synthetic data: {X.shape[0]} samples, {D} features, {K} classes.")


# --- Training Setup ---
input_size = D
hidden_size = 10 # Small hidden layer
output_size = K
iterations = 100
batch_size = 30 # Use mini-batches
learning_rates = [1.0, 0.1, 0.01] # Different learning rates to compare

# Dictionary to store loss history for each learning rate
loss_histories = {}

print("\nStarting training experiment...")

# --- Training Loop ---
for lr in learning_rates:
    print(f"\n--- Training with Learning Rate: {lr} ---")

    # Initialize network and optimizer for each LR
    network = TwoLayerNet(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    optimizer = SGD(lr=lr)

    current_loss_history = []

    for i in range(iterations):
        # Create a mini-batch
        batch_mask = np.random.choice(X.shape[0], batch_size)
        x_batch = X[batch_mask]
        y_batch = y_one_hot[batch_mask] # Use one-hot labels for loss calculation

        # Calculate gradients
        grads = network.gradient(x_batch, y_batch)

        # Update parameters
        optimizer.update(network.params, grads)

        # Calculate and record loss on the current batch
        loss = network.loss(x_batch, y_batch)
        current_loss_history.append(loss)

        if (i + 1) % (iterations // 5) == 0 or i == 0:
            # Optionally calculate accuracy on the whole dataset (can be slow)
            # train_acc = network.accuracy(X, y_indices)
            print(f"Iteration {i+1}/{iterations}, Loss: {loss:.4f}") #, Accuracy: {train_acc:.3f}")

    loss_histories[lr] = current_loss_history
    print(f"Finished training for LR={lr}.")


# --- Plotting Results ---
print("\nPlotting loss curves...")
plt.figure(figsize=(10, 6))

for lr, loss_list in loss_histories.items():
    plt.plot(range(1, iterations + 1), loss_list, label=f'LR = {lr}')

plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss vs. Iteration for Different Learning Rates (SGD)")
plt.legend()
plt.grid(True)
plt.ylim(0, 3.0) # Adjust Y limit if necessary based on loss values
plt.show()

print("\nExperiment finished.")