# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 14:26:17 2024

@author: Hunter
"""

import numpy as np

# Tensor class as you defined...
class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=np.float64)  # Convert data to numpy array
        self.requires_grad = requires_grad
        self.grad = None  # Gradients start as None
        self._backward = lambda: None  # Default backward is a no-op
        self.left = None  # Left operand for binary operations
        self.right = None  # Right operand for binary operations

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)

        # Assign left and right operands for backpropagation
        out.left, out.right = self, other

        def _backward():
            print(f"Add backward: out.grad = {out.grad}")
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += out.grad
            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                other.grad += out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = self + Tensor(-other.data, requires_grad=self.requires_grad or other.requires_grad)

        # Assign left and right operands for backpropagation
        out.left, out.right = self, other

        def _backward():
            print(f"Sub backward: out.grad = {out.grad}")
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += out.grad
            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                other.grad -= out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        if isinstance(other, Tensor):
            out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)

            # Assign left and right operands for backpropagation
            out.left, out.right = self, other

            def _backward():
                print(f"Mul backward: out.grad = {out.grad}")
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = np.zeros_like(self.data)
                    self.grad += other.data * out.grad
                if other.requires_grad:
                    if other.grad is None:
                        other.grad = np.zeros_like(other.data)
                    other.grad += self.data * out.grad

            out._backward = _backward
        else:
            out = Tensor(self.data * other, requires_grad=self.requires_grad)

            # Assign left operand and set right to None
            out.left, out.right = self, None

            def _backward():
                print(f"Scalar Mul backward: out.grad = {out.grad}")
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = np.zeros_like(self.data)
                    self.grad += other * out.grad

            out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.matmul(self.data, other.data), requires_grad=self.requires_grad or other.requires_grad)

        # Assign left and right operands for backpropagation
        out.left, out.right = self, other

        def _backward():
            print(f"Matmul backward: out.grad = {out.grad}")
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

    def mean(self):
        out = Tensor(self.data.mean(), requires_grad=self.requires_grad)

        def _backward():
            print(f"Mean backward: out.grad = {out.grad}")
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += (1 / self.data.size) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        # Initialize the gradient if it's not already set (start with 1 for scalar loss)
        if self.grad is None:
            self.grad = np.ones_like(self.data, dtype=np.float64)

        #print(f"Backward pass started: self.grad = {self.grad}")

        # Apply the backward function for this tensor
        self._backward()

        # Backpropagate through left and right operands (if they require gradients)
        if self.left and self.left.requires_grad:
            print(f"Recursively calling backward on left operand")
            self.left.backward()
        if self.right and self.right.requires_grad:
            print(f"Recursively calling backward on right operand")
            self.right.backward()
            
# class Linear:
#     def __init__(self, in_features, out_features, requires_grad=True):
#         # Initialize weights and biases with requires_grad
#         self.weights = Tensor(np.random.randn(in_features, out_features), requires_grad=requires_grad)
#         self.bias = Tensor(np.zeros(out_features), requires_grad=requires_grad)
    
class Linear:
    def __init__(self, in_features, out_features, requires_grad=True):
        # Initialize weights and biases with requires_grad
        self.weights = Tensor(np.random.randn(in_features, out_features), requires_grad=requires_grad)
        self.bias = Tensor(np.zeros(out_features), requires_grad=requires_grad)

    def __call__(self, x):
        # Perform linear transformation: y = xW + b
        out = x @ self.weights + self.bias
        #print(f"Linear forward: x = {x.data}, weights = {self.weights.data}, bias = {self.bias.data}, out = {out.data}")

        # Define backward propagation
        def _backward():
            print("Linear Backprop invoked")
            if out.grad is None:
                raise ValueError("Backward called on a tensor with no gradient (out.grad is None)")

            if x.requires_grad:
                if x.grad is None:
                    x.grad = np.zeros_like(x.data)
                x.grad += np.matmul(out.grad, self.weights.data.T)
                #print(f"x.grad updated to {x.grad}")

            if self.weights.requires_grad:
                if self.weights.grad is None:
                    self.weights.grad = np.zeros_like(self.weights.data)
                self.weights.grad += np.matmul(x.data.T, out.grad)
                #print(f"weights.grad updated to {self.weights.grad}")

            if self.bias.requires_grad:
                if self.bias.grad is None:
                    self.bias.grad = np.zeros_like(self.bias.data)
                self.bias.grad += np.sum(out.grad, axis=0)
                #print(f"bias.grad updated to {self.bias.grad}")

        # Assign the backward function to the output tensor
        out._backward = _backward

        # Ensure that left and right are set for backpropagation purposes
        out.left, out.right = x, None

        return out



class ReLU:
    def __init__(self):
        pass

    def __call__(self, x):
        # Perform ReLU activation: y = max(0, x)
        out = Tensor(np.maximum(0, x.data), requires_grad=x.requires_grad)

        # Assign left operand for backpropagation
        out.left = x

        # Define backward function
        def _backward():
            print("ReLU back Invoked")
            if x.requires_grad:
                if x.grad is None:
                    x.grad = np.zeros_like(x.data)
                # ReLU backward: gradient is 0 for x <= 0, and 1 for x > 0
                print(f"ReLU backward out.data: {out.data}")
                print(f"ReLU backward x.grad before: {x.grad}")
                x.grad += (out.data > 0) * out.grad
                print(f"ReLU backward x.grad after: {x.grad}")

            # Call backward on x's dependencies
            if hasattr(x, '_backward'):
                print("x has back")
                x._backward()

        # Assign the backward function to the output tensor
        out._backward = _backward

        return out


    
class MSELoss:
    def __init__(self):
        pass

    def __call__(self, predicted, target):
        # Compute MSE loss
        loss = ((predicted.data - target.data) ** 2).mean()

        # Store the loss for backward propagation
        out = Tensor(loss, requires_grad=True)

        # Define backward function for MSE
        def _backward():
            #print("MSE back invoked")
            if predicted.requires_grad:
                if predicted.grad is None:
                    predicted.grad = np.zeros_like(predicted.data)
                # Gradient of the MSE loss with respect to the predicted values
                predicted.grad += 2 * (predicted.data - target.data) / target.data.size
                predicted._backward()
                
            if target.requires_grad:
                if target.grad is None:
                    target.grad = np.zeros_like(target.data)
                target.grad += -2 * (predicted.data - target.data) / target.data.size
                predicted._backward()
        out._backward = _backward
        return out


# Helper test functions for each section:

def test_combined_operations():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

    # Test (x + y) * (x - y)
    z = (x + y) * (x - y)
    z.backward()
    print(f"Result of (x + y) * (x - y):\n{z.data}")
    print(f"Gradients for x:\n{x.grad}")
    print(f"Gradients for y:\n{y.grad}")

    # Test (x @ y) + (x * y)
    z = ((x @ y) + (x * y) * 2).mean()
    z.backward()
    print(f"Result of (x @ y) + (x * y) * 2:\n{z.data}")
    print(f"Gradients for x:\n{x.grad}")
    print(f"Gradients for y:\n{y.grad}")

def test_linear_layer():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    linear_layer = Linear(in_features=2, out_features=1)

    output = linear_layer(x)

    print("Output from Linear Layer (forward pass):")
    print(output.data)

    output.backward()

    print("\nGradients after backward pass:")
    print(f"Gradients for input x:\n{x.grad}")
    print(f"Gradients for weights:\n{linear_layer.weights.grad}")
    print(f"Gradients for bias:\n{linear_layer.bias.grad}")

def test_combined_operations_with_linear():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

    # Apply (x + y) @ (x - y)
    z = (x + y) @ (x - y)
    print(f"Result of (x + y) @ (x - y):\n{z.data}")

    # Apply the linear layer
    linear_layer = Linear(in_features=2, out_features=1)
    output = linear_layer(z)
    print(f"Output from linear layer:\n{output.data}")

    # Perform backpropagation
    output.backward()

    print("\nGradients after backward pass:")
    print(f"Gradients for z (input): {z.grad}")
    print(f"Gradients for x: {x.grad}")
    print(f"Gradients for y: {y.grad}")
    print(f"Gradients for weights: {linear_layer.weights.grad}")
    print(f"Gradients for bias: {linear_layer.bias.grad}")

def test_combined_operations_with_relu():
    x = Tensor([[1.0, -2.0], [3.0, -4.0]], requires_grad=True)
    y = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

    # Apply (x + y) * (x - y)
    z = (x + y) * (x - y)
    #print(f"Result of (x + y) * (x - y):\n{z.data}")

    # Apply ReLU activation
    relu = ReLU()
    relu_output = relu(z)
    #print(f"ReLU output:\n{relu_output.data}")

    # Perform backpropagation
    relu_output.backward()

    print("\nGradients after backward pass:")
    print(f"Gradients for x (after backward): {x.grad}")
    print(f"Gradients for y (after backward): {y.grad}")

def test_mse_loss():
    predicted = Tensor([[0.5, 0.8], [0.2, 0.9]], requires_grad=True)
    target = Tensor([[1.0, 0.0], [0.5, 1.0]], requires_grad=False)

    loss_fn = MSELoss()
    loss = loss_fn(predicted, target)

    #print(f"Loss: {loss.data}")

    loss.backward()

    #print(f"Gradients for predicted:\n{predicted.grad}")

# # Run all tests
# print("**********Combined Operations Test**********")
# test_combined_operations()
# print("\n**********Linear Layer Test**********")
# test_linear_layer()
# print("\n**********Linear with Operations Test**********")
# test_combined_operations_with_linear()
print("\n**********ReLU Activation Test**********")
test_combined_operations_with_relu()
# print("\n**********MSE Loss Test**********")
# test_mse_loss()

print("*******Single Layer Linear Test********")
print("")
# Generate synthetic data (y = 2 * x + 1)
np.random.seed(0)
x_data = np.random.randn(100, 1)  # 100 samples, 1 feature
y_data = 2 * x_data + 1  # Linear function: y = 2 * x + 1

# Training the model
learning_rate = 0.01
epochs = 1000
input_size = 1
output_size = 1

# Initialize model and loss
model = Linear(input_size, output_size)
loss_fn = MSELoss()

# Convert data to Tensor
x_tensor = Tensor(x_data)
y_tensor = Tensor(y_data)

# Training loop
for epoch in range(epochs):
    # Forward pass
    y_pred = model(x_tensor)

    # Compute loss
    loss = loss_fn(y_pred, y_tensor)

    # Zero gradients (initialize to None before backprop)
    model.weights.grad = None
    model.bias.grad = None
    x_tensor.grad = None

    # Backward pass
    loss._backward()

    # Update weights and biases (Gradient Descent)
    if model.weights.grad is not None:
        model.weights.data -= learning_rate * model.weights.grad
    if model.bias.grad is not None:
        model.bias.data -= learning_rate * model.bias.grad

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data}")

# # Define model
# class TwoLayerNN:
#     def __init__(self, input_size, hidden_size, output_size):
#         self.layer1 = Linear(input_size, hidden_size)
#         self.activation = ReLU()
#         self.layer2 = Linear(hidden_size, output_size)

#     def __call__(self, x):
#         # Forward pass: Linear -> ReLU -> Linear
#         out = self.layer1(x)
#         out = self.activation(out)
#         out = self.layer2(out)
#         return out

# # Training parameters
# learning_rate = 0.01
# epochs = 1000

# # Initialize model, loss, and data
# model = TwoLayerNN(input_size=1, hidden_size=4, output_size=1)
# loss_fn = MSELoss()

# # Example training data: y = 2x
# x_train = Tensor(np.array([[1.0], [2.0], [3.0], [4.0]]), requires_grad=True)
# y_train = Tensor(np.array([[2.0], [4.0], [6.0], [8.0]]))

# # Training loop
# for epoch in range(epochs):
#     # Forward pass
#     y_pred = model(x_train)
#     loss = loss_fn(y_pred, y_train)

#     # Backward pass
#     loss._backward()

#     # Update weights and biases
#     for layer in [model.layer1, model.layer2]:
#         layer.weights.data -= learning_rate * layer.weights.grad
#         layer.bias.data -= learning_rate * layer.bias.grad

#         # Clear gradients after the update
#         layer.weights.grad = None
#         layer.bias.grad = None

#     # Print loss every 100 epochs
#     if epoch % 100 == 0:
#         print(f"Epoch {epoch}, Loss: {loss.data}")

# # Final evaluation
# print("Training complete")
# print(f"Final Weights Layer 1: {model.layer1.weights.data}")
# print(f"Final Bias Layer 1: {model.layer1.bias.data}")
# print(f"Final Weights Layer 2: {model.layer2.weights.data}")
# print(f"Final Bias Layer 2: {model.layer2.bias.data}")

def test_combined_operations_with_relu():
    # Define input tensors with positive values for testing
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

    # Apply operations (x + y) * (x - y)
    z = (x + y) * (x + y)
    print(f"Result of (x + y) * (x - y):\n{z.data}")

    # Apply ReLU activation
    relu = ReLU()
    relu_output = relu(z)
    print(f"ReLU output:\n{relu_output.data}")

    # Target values (arbitrary example)
    target = Tensor([[1.0, 1.0], [1.0, 1.0]], requires_grad=False)
    
    # Calculate Mean Squared Error loss
    loss = MSELoss()(relu_output, target)
    print(f"Loss: {loss.data}")

    # Perform backpropagation
    print("Starting backward pass...")
    loss.backward()

    # Print gradients
    print("\nGradients after backward pass:")
    print(f"Gradients for x (after backward): {x.grad}")
    print(f"Gradients for y (after backward): {y.grad}")

# Run the test
test_combined_operations_with_relu()

def train_nn():
    # Initialize layers
    linear1 = Linear(2, 3)  # 2 inputs, 3 outputs
    linear2 = Linear(3, 1)  # 3 inputs, 1 output (final output)
    mse_loss = MSELoss()

    # Input and target data
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    target = Tensor([[5.0], [7.0]], requires_grad=False)

    # Learning rate for gradient descent
    learning_rate = 0.01

    for epoch in range(1000):
        # Forward pass: pass input through the network
        out1 = linear1(x)  # Linear layer 1
        output = linear2(out1)  # Linear layer 2

        # Compute loss
        loss = mse_loss(output, target)
        print(f"Epoch {epoch}, Loss: {loss.data}")

        # Backward pass: calculate gradients
        loss.backward()

        # Update weights and biases using gradient descent
        linear1.weights.data -= learning_rate * linear1.weights.grad
        linear1.bias.data -= learning_rate * linear1.bias.grad
        linear2.weights.data -= learning_rate * linear2.weights.grad
        linear2.bias.data -= learning_rate * linear2.bias.grad

        # Zero the gradients after each update
        linear1.weights.grad = np.zeros_like(linear1.weights.data)
        linear1.bias.grad = np.zeros_like(linear1.bias.data)
        linear2.weights.grad = np.zeros_like(linear2.weights.data)
        linear2.bias.grad = np.zeros_like(linear2.bias.data)

train_nn()