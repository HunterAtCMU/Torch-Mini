import numpy as np

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

class CrossEntropyLoss:
     def __call__(self, predicted, target):
         batch_size = predicted.data.shape[0]
         # Add small epsilon to avoid log(0)
         eps = 1e-7
        
         # Apply softmax
         exp_pred = np.exp(predicted.data - np.max(predicted.data, axis=1, keepdims=True))
         softmax_pred = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
        
         # Convert target to one-hot encoding
         target_one_hot = np.zeros_like(softmax_pred)
         target_one_hot[np.arange(batch_size), target.data] = 1
        
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