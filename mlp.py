# %%
import numpy as np

def sigmoid(x):
    x = x.astype(float)
    pos_mask = x >= 0
    result = np.zeros_like(x, dtype=float)
    
    # For positive values: 1 / (1 + exp(-x))
    result[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))
    
    # For negative values: exp(x) / (1 + exp(x))
    neg_mask = ~pos_mask
    exp_x = np.exp(x[neg_mask])
    result[neg_mask] = exp_x / (1 + exp_x)

    return result

def sigmoid_derivative(x):
    # force x to be a float
    x = x.astype(float)
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    # force x to be a float
    x = x.astype(float)
    return np.maximum(0, x)

def relu_derivative(x):
    # force x to be a float
    x = x.astype(float)
    return np.where(x > 0, 1, 0)


class MLP:

    def __init__(self, layer_sizes, beta1=0.9, beta2=0.999, epsilon=1e-8, random_seed=42):
        # layer_sizes includes input and output layers
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        self.n_layers = len(layer_sizes)
        
        # Adam optimizer parameters
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # time step
        
        # Initialize first and second moment estimates
        self.m_weights = []
        self.v_weights = []
        self.m_biases = []
        self.v_biases = []

        # Glorot initialization
        np.random.seed(random_seed)
        for i in range(1, self.n_layers):
            scale = np.sqrt(2.0 / (layer_sizes[i] + layer_sizes[i-1]))
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i-1]) * scale)
            self.biases.append(np.zeros((layer_sizes[i], 1)))
            
            # Initialize Adam parameters
            self.m_weights.append(np.zeros_like(self.weights[-1]))
            self.v_weights.append(np.zeros_like(self.weights[-1]))
            self.m_biases.append(np.zeros_like(self.biases[-1]))
            self.v_biases.append(np.zeros_like(self.biases[-1]))
        

    def forward(self, X):
        activations = [X]  # Input 
        z_values = []      
        
        # Forward pass
        for i in range(self.n_layers - 2):
            z = self.weights[i] @ activations[i] + self.biases[i]
            z_values.append(z)
            
            activation = relu(z)
            activations.append(activation)
        
        z = self.weights[-1] @ activations[-1] + self.biases[-1]
        z_values.append(z)
        activation = sigmoid(z)
        activations.append(activation)
        
        return activations, z_values
    
    def backward(self, X, y):
        activations, z_values = self.forward(X)

        # Compute gradients
        m = X.shape[0]

        # Initialize gradients
        d_weights = [np.zeros_like(w) for w in self.weights]
        d_biases = [np.zeros_like(b) for b in self.biases]

        # Compute gradients for output layer
        delta = (activations[-1] - y)
        for l in reversed(range(self.n_layers - 1)):
            d_weights[l] = delta @ activations[l].T / m
            d_biases[l] = np.sum(delta, axis=1, keepdims=True) / m

            if l == 0:
                continue

            if l == self.n_layers - 2:
                delta = (self.weights[l].T @ delta) * sigmoid_derivative(z_values[l-1])
            else:
                delta = (self.weights[l].T @ delta) * relu_derivative(z_values[l-1])

        return d_weights, d_biases
        

    def update_parameters(self, d_weights, d_biases, learning_rate):
        self.t += 1
        
        for l in range(self.n_layers - 1):
            # Update biased first moment estimate
            self.m_weights[l] = self.beta1 * self.m_weights[l] + (1 - self.beta1) * d_weights[l]
            self.m_biases[l] = self.beta1 * self.m_biases[l] + (1 - self.beta1) * d_biases[l]
            
            # Update biased second raw moment estimate
            self.v_weights[l] = self.beta2 * self.v_weights[l] + (1 - self.beta2) * (d_weights[l] ** 2)
            self.v_biases[l] = self.beta2 * self.v_biases[l] + (1 - self.beta2) * (d_biases[l] ** 2)
            
            # Compute bias-corrected first moment estimate
            m_weights_hat = self.m_weights[l] / (1 - self.beta1 ** self.t)
            m_biases_hat = self.m_biases[l] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_weights_hat = self.v_weights[l] / (1 - self.beta2 ** self.t)
            v_biases_hat = self.v_biases[l] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            self.weights[l] -= learning_rate * m_weights_hat / (np.sqrt(v_weights_hat) + self.epsilon)
            self.biases[l] -= learning_rate * m_biases_hat / (np.sqrt(v_biases_hat) + self.epsilon)


    def cross_entropy(self, y_pred, y_true):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def train(self, X, y, learning_rate, n_epochs):
        X = X.T
        y = y.T

        loss_history = []
        for epoch in range(n_epochs):
            activations, _ = self.forward(X)
            y_pred = activations[-1]
            loss = self.cross_entropy(y_pred, y)
            loss_history.append(loss)

            d_weights, d_biases = self.backward(X, y)
            self.update_parameters(d_weights, d_biases, learning_rate)

            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss:.4f}")

        return loss_history

    def predict(self, X):
        X = X.T
        y_pred, _ = self.forward(X)
        return y_pred[-1].T
    

def generate_xor_data(n):
    X = np.random.randn(n, 2)
    y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).reshape(-1, 1)
    return X, y


# %%
if __name__ == "__main__":
    X, y = generate_xor_data(200)
    print(f"X.shape: {X.shape}")
    print(f"y.shape: {y.shape}")

    model = MLP(layer_sizes=[2, 64, 1])
    losses = model.train(X, y, learning_rate=0.1, n_epochs=300)

    predictions = model.predict(X)

    accuracy = np.mean((predictions > 0.5).astype(int) == y)
    print(f"Accuracy: {accuracy:.4f}")

    import matplotlib.pyplot as plt

    # plot the losses
    plt.plot(losses)
    plt.show()
# %%
