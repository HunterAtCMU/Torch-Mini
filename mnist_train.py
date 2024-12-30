import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self.prev = []
        self._saved_tensors = {}

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)
        
        if out.requires_grad:
            out.prev = [self, other]
            # Save shapes for backward pass
            out._saved_tensors['self_shape'] = self.data.shape
            out._saved_tensors['other_shape'] = other.data.shape
            
            def _backward():
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = np.zeros_like(self.data)
                    # Sum gradients over broadcasted dimensions if necessary
                    grad = out.grad
                    if self.data.shape != out.grad.shape:
                        # Sum over broadcasted dimensions
                        axis = tuple(range(len(out.grad.shape) - len(self.data.shape)))
                        grad = np.sum(out.grad, axis=axis if axis else None)
                        # Reshape if necessary
                        if grad.shape != self.data.shape:
                            grad = np.sum(grad, axis=tuple(
                                i for i, (a, b) in enumerate(zip(grad.shape[::-1], self.data.shape[::-1]))
                                if a != b
                            ))
                    self.grad += grad

                if other.requires_grad:
                    if other.grad is None:
                        other.grad = np.zeros_like(other.data)
                    # Sum gradients over broadcasted dimensions if necessary
                    grad = out.grad
                    if other.data.shape != out.grad.shape:
                        # Sum over broadcasted dimensions
                        axis = tuple(range(len(out.grad.shape) - len(other.data.shape)))
                        grad = np.sum(out.grad, axis=axis if axis else None)
                        # Reshape if necessary
                        if grad.shape != other.data.shape:
                            grad = np.sum(grad, axis=tuple(
                                i for i, (a, b) in enumerate(zip(grad.shape[::-1], other.data.shape[::-1]))
                                if a != b
                            ))
                    other.grad += grad
            out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)
        
        if out.requires_grad:
            out.prev = [self, other]
            def _backward():
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = np.zeros_like(self.data)
                    grad = other.data * out.grad
                    if grad.shape != self.data.shape:
                        axis = tuple(range(len(grad.shape) - len(self.data.shape)))
                        grad = np.sum(grad, axis=axis if axis else None)
                    self.grad += grad

                if other.requires_grad:
                    if other.grad is None:
                        other.grad = np.zeros_like(other.data)
                    grad = self.data * out.grad
                    if grad.shape != other.data.shape:
                        axis = tuple(range(len(grad.shape) - len(other.data.shape)))
                        grad = np.sum(grad, axis=axis if axis else None)
                    other.grad += grad
            out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.matmul(self.data, other.data), requires_grad=self.requires_grad or other.requires_grad)
        
        if out.requires_grad:
            out.prev = [self, other]
            def _backward():
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = np.zeros_like(self.data)
                    self.grad += np.matmul(out.grad, other.data.T)
                if other.requires_grad:
                    if other.grad is None:
                        other.grad = np.zeros_like(other.data)
                    other.grad += np.matmul(self.data.T, out.grad)
            out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        
        def build_topo(tensor):
            if tensor not in visited:
                visited.add(tensor)
                for prev_tensor in tensor.prev:
                    if prev_tensor.requires_grad:
                        build_topo(prev_tensor)
                topo.append(tensor)
        
        build_topo(self)
        
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        
        for tensor in reversed(topo):
            tensor._backward()

class Linear:
    def __init__(self, in_features, out_features, requires_grad=True):
        self.weights = Tensor(np.random.randn(in_features, out_features) * np.sqrt(2.0/(in_features)), requires_grad=requires_grad)
        self.bias = Tensor(np.zeros((1, out_features)), requires_grad=requires_grad)  # Note the shape change here

    def __call__(self, x):
        out = x @ self.weights + self.bias
        out._saved_tensors['input'] = x
        return out

    def zero_grad(self):
        self.weights.grad = None
        self.bias.grad = None

class ReLU:
    def __call__(self, x):
        out = Tensor(np.maximum(0, x.data), requires_grad=x.requires_grad)
        
        if out.requires_grad:
            out.prev = [x]
            out._saved_tensors['mask'] = (x.data > 0).astype(np.float64)
            
            def _backward():
                if x.requires_grad:
                    if x.grad is None:
                        x.grad = np.zeros_like(x.data)
                    x.grad += out._saved_tensors['mask'] * out.grad
            out._backward = _backward
        return out

class MSELoss:
    def __call__(self, predicted, target):
        diff = predicted.data - target.data
        out = Tensor(np.mean(diff ** 2), requires_grad=predicted.requires_grad)
        
        if out.requires_grad:
            out.prev = [predicted]
            def _backward():
                if predicted.requires_grad:
                    if predicted.grad is None:
                        predicted.grad = np.zeros_like(predicted.data)
                    predicted.grad += (2.0 / predicted.data.size) * diff
            out._backward = _backward
        return out

class MNISTLoader:
    def __init__(self):
        print("Loading MNIST from scikit-learn...")
        # Load MNIST from scikit-learn
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
        
        # Convert to float32 and normalize to [0,1]
        self.X = X.astype(np.float32) / 255.0
        # Convert labels to integers - this is the key fix
        self.y = y.astype(np.int64)  # Changed from int32 to ensure proper indexing

    def load_data(self, num_train=10000, num_test=100):
        # Split into train and test
        train_images = self.X[:60000][:num_train]
        train_labels = self.y[:60000][:num_train]
        test_images = self.X[60000:][:num_test]
        test_labels = self.y[60000:][:num_test]
        
        return train_images, train_labels, test_images, test_labels

# class CrossEntropyLoss:
#     def __call__(self, predicted, target):
#         batch_size = predicted.data.shape[0]
#         # Add small epsilon to avoid log(0)
#         eps = 1e-7
        
#         # Apply softmax
#         exp_pred = np.exp(predicted.data - np.max(predicted.data, axis=1, keepdims=True))
#         softmax_pred = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
        
#         # Convert target to one-hot encoding
#         target_one_hot = np.zeros_like(softmax_pred)
#         target_one_hot[np.arange(batch_size), target.data] = 1
        
#         # Compute cross entropy loss
#         loss = -np.sum(target_one_hot * np.log(softmax_pred + eps)) / batch_size
#         out = Tensor(loss, requires_grad=predicted.requires_grad)
        
#         if out.requires_grad:
#             out.prev = [predicted]
#             def _backward():
#                 if predicted.requires_grad:
#                     if predicted.grad is None:
#                         predicted.grad = np.zeros_like(predicted.data)
#                     # Gradient of cross entropy with softmax
#                     predicted.grad += (softmax_pred - target_one_hot) / batch_size
#             out._backward = _backward
#         return out

class CrossEntropyLoss:
    def __call__(self, predicted, target):
        batch_size = predicted.data.shape[0]
        # Add small epsilon to avoid log(0)
        eps = 1e-7
        
        # Apply softmax
        exp_pred = np.exp(predicted.data - np.max(predicted.data, axis=1, keepdims=True))
        softmax_pred = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
        
        # Convert target to one-hot encoding
        target_indices = target.data.astype(np.int64)  # Ensure target is integer
        target_one_hot = np.zeros_like(softmax_pred)
        target_one_hot[np.arange(batch_size), target_indices] = 1
        
        # Compute cross entropy loss
        loss = -np.sum(target_one_hot * np.log(softmax_pred + eps)) / batch_size
        out = Tensor(loss, requires_grad=predicted.requires_grad)
        
        if out.requires_grad:
            out.prev = [predicted]
            def _backward():
                if predicted.requires_grad:
                    if predicted.grad is None:
                        predicted.grad = np.zeros_like(predicted.data)
                    # Gradient of cross entropy with softmax
                    predicted.grad += (softmax_pred - target_one_hot) / batch_size
            out._backward = _backward
        return out


def accuracy(predictions, targets):
    return np.mean(np.argmax(predictions.data, axis=1) == targets.data)

class BatchLoader:
    def __init__(self, X, y, batch_size):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.n_samples = X.shape[0]
        self.n_batches = (self.n_samples + batch_size - 1) // batch_size
        self.current_batch = 0

    def __iter__(self):
        self.current_batch = 0
        self.indices = np.random.permutation(self.n_samples)
        return self

    def __next__(self):
        if self.current_batch >= self.n_batches:
            raise StopIteration
        
        start_idx = self.current_batch * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.n_samples)
        batch_indices = self.indices[start_idx:end_idx]
        
        self.current_batch += 1
        return (Tensor(self.X[batch_indices], requires_grad=True),
                Tensor(self.y[batch_indices]))

# def train_mnist():
#     # Load MNIST data
#     print("Loading MNIST data...")
#     loader = MNISTLoader()
#     train_images, train_labels, test_images, test_labels = loader.load_data(num_train=5000, num_test=1000)
    
#     # Network parameters
#     input_size = 784  # 28x28 pixels
#     hidden_size = 256
#     output_size = 10  # 10 digits
#     batch_size = 100
#     epochs = 150
#     learning_rate = 0.05

#     # Initialize network
#     linear1 = Linear(input_size, hidden_size)
#     relu = ReLU()
#     linear2 = Linear(hidden_size, output_size)
#     loss_fn = CrossEntropyLoss()
    
#     print(f"Training on {len(train_images)} samples, testing on {len(test_images)} samples")
    
#     # Training loop
#     for epoch in range(epochs):
#         total_loss = 0
#         total_acc = 0
#         n_batches = 0
        
#         # Create batch iterator
#         batch_loader = BatchLoader(train_images, train_labels, batch_size)
        
#         for batch_X, batch_y in batch_loader:
#             # Forward pass
#             out1 = linear1(batch_X)
#             out2 = relu(out1)
#             output = linear2(out2)
            
#             # Compute loss and accuracy
#             loss = loss_fn(output, batch_y)
#             acc = accuracy(output, batch_y)
            
#             # Zero gradients
#             linear1.zero_grad()
#             linear2.zero_grad()
            
#             # Backward pass
#             loss.backward()
            
#             # Update weights
#             linear1.weights.data -= learning_rate * linear1.weights.grad
#             linear1.bias.data -= learning_rate * linear1.bias.grad
#             linear2.weights.data -= learning_rate * linear2.weights.grad
#             linear2.bias.data -= learning_rate * linear2.bias.grad
            
#             total_loss += loss.data
#             total_acc += acc
#             n_batches += 1
        
#         # Compute epoch metrics
#         avg_loss = total_loss / n_batches
#         avg_acc = total_acc / n_batches
        
#         print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")
        
#         # Evaluate on test set
#         if (epoch + 1) % 5 == 0:
#             test_loader = BatchLoader(test_images, test_labels, batch_size)
#             test_acc = 0
#             n_test_batches = 0
            
#             for test_X, test_y in test_loader:
#                 # Forward pass without computing gradients
#                 out1 = linear1(Tensor(test_X.data, requires_grad=False))
#                 out2 = relu(out1)
#                 test_output = linear2(out2)
#                 test_acc += accuracy(test_output, test_y)
#                 n_test_batches += 1
            
#             test_acc /= n_test_batches
#             print(f"Test Accuracy: {test_acc:.4f}")

# if __name__ == "__main__":
#     train_mnist()

def train_mnist():
    # Load MNIST data
    print("Loading MNIST data...")
    loader = MNISTLoader()
    train_images, train_labels, test_images, test_labels = loader.load_data(num_train=5000, num_test=1000)
    
    # Network parameters
    input_size = 784  # 28x28 pixels
    hidden_size = 128
    output_size = 10  # 10 digits
    batch_size = 100
    epochs = 150
    learning_rate = 0.08

    # Initialize network
    linear1 = Linear(input_size, hidden_size)
    relu = ReLU()
    linear2 = Linear(hidden_size, output_size)
    loss_fn = CrossEntropyLoss()
    
    print(f"Training on {len(train_images)} samples, testing on {len(test_images)} samples")

    # Lists to store accuracy for plotting
    training_accuracies = []
    test_accuracies = []

    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        total_acc = 0
        n_batches = 0
        
        # Create batch iterator
        batch_loader = BatchLoader(train_images, train_labels, batch_size)
        
        for batch_X, batch_y in batch_loader:
            # Forward pass
            out1 = linear1(batch_X)
            out2 = relu(out1)
            output = linear2(out2)
            
            # Compute loss and accuracy
            loss = loss_fn(output, batch_y)
            acc = accuracy(output, batch_y)
            
            # Zero gradients
            linear1.zero_grad()
            linear2.zero_grad()
            
            # Backward pass
            loss.backward()
            
            # Update weights
            linear1.weights.data -= learning_rate * linear1.weights.grad
            linear1.bias.data -= learning_rate * linear1.bias.grad
            linear2.weights.data -= learning_rate * linear2.weights.grad
            linear2.bias.data -= learning_rate * linear2.bias.grad
            
            total_loss += loss.data
            total_acc += acc
            n_batches += 1
        
        # Compute epoch metrics
        avg_loss = total_loss / n_batches
        avg_acc = total_acc / n_batches
        training_accuracies.append(avg_acc)  # Log training accuracy
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")
        
        # Evaluate on test set
        if (epoch + 1) % 5 == 0:
            test_loader = BatchLoader(test_images, test_labels, batch_size)
            test_acc = 0
            n_test_batches = 0
            
            for test_X, test_y in test_loader:
                # Forward pass without computing gradients
                out1 = linear1(Tensor(test_X.data, requires_grad=False))
                out2 = relu(out1)
                test_output = linear2(out2)
                test_acc += accuracy(test_output, test_y)
                n_test_batches += 1
            
            test_acc /= n_test_batches
            test_accuracies.append(test_acc)  # Log test accuracy
            print(f"Test Accuracy: {test_acc:.4f}")

    # Plot training and test accuracy
    plot_accuracies(training_accuracies, test_accuracies, epochs)

def plot_accuracies(train_acc, test_acc, epochs):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_acc, label="Training Accuracy", marker='o', color='blue')
    test_epochs = list(range(5, epochs + 1, 5))  # Test accuracy is logged every 5 epochs
    plt.plot(test_epochs, test_acc, label="Test Accuracy", marker='s', color='orange')
    plt.title("Training and Test Accuracy vs Epoch", fontsize=16)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

if __name__ == "__main__":
    train_mnist()
