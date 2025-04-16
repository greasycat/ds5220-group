# %%
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)


class MLP:

    def __init__(self, layer_sizes):
        # layer_sizes includes input and output layers
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        self.n_layers = len(layer_sizes)



        # Glorot initialization
        for i in range(1, self.n_layers):
            scale = np.sqrt(2.0 / (layer_sizes[i] + layer_sizes[i-1]))
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i-1]) * scale)
            self.biases.append(np.zeros((layer_sizes[i], 1)))
        

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
        for l in range(self.n_layers - 1):
            self.weights[l] -= learning_rate * d_weights[l]
            self.biases[l] -= learning_rate * d_biases[l]


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
    losses = model.train(X, y, learning_rate=0.1, n_epochs=1000)

    predictions = model.predict(X)

    accuracy = np.mean((predictions > 0.5).astype(int) == y)
    print(f"Accuracy: {accuracy:.4f}")
# %%
