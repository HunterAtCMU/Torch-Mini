# mytorch

## Homebrew Machine Learning Library using Python and NumPy

This version works with arbitrary neural network structures and achieved 91% test accuracy on the MNIST dataset without fine-tuning.

Feel free to use as you wish, or add anything that may be of use. I intend this to be educational material for anyone interested in learning the basics of ML. Doing this project taught me a lot about the practical aspects, so I'll describe the process as thoroughly as possible without diving into too much theory.

If you don't want to go through the guide, there are scripts which contain the library, a training loop for synthetic data and MNIST data, as well as some debugging scripts. They should work without any adjustments as long as you have NumPy, sklearn, and matplotlib. The library only requires NumPy.

### Introduction.

Machine learning is the backbone of Artificial Intelligence and the cool part is, it's actually pretty easy to understand. The thing that makes AI exciting (or scary) is scaling-- meaning the more layers and training data, the less interpretable, yet powerful, the model will become. This is why tools like ChatGPT can behave in nebulous ways. There are simply too many parameters for humans to understand (look up mechanistic interpretability).

However, the algorithm that powers such tools has been around for several decades and is quite intuitive. This project strips away all of the state-of-the-art inventions (like transformers) used by large language models and focuses on the essence of machine learning. 

This implementation includes tensors, a linear class, a ReLU class, two cost functions, as well as some examples and tests that can be run to better understand how neural networks function.

### How ML Works.

The goal of machine learning, generally, is to provide some sort of data to a computer, extract patterns or relationships from the data, and use the extracted information to make predictions, generate new data, etc. 

Neural networks are a popular way to implement this behavior and can be thought of as "arbitrary function approximaters". I like this description because it gets at the fact that computers only understand numbers, making tools like ChatGPT less intimidating. The idea is to have labeled data, such as animal pictures with the species labeled. We can imagine $x$ as the picture and $y$ as the label. We assume there is some unknown function $f(x)$ such that $f(x)=y$. If we knew the function, we could put in a picture of an animal and the function would spit out the species. Neural networks use a large amount of labeled training data to approximate $f(x)$.

In practice, $x$ would be a high-dimensional array (imagine an array of numbers representing the RGB values of each pixel in an image), $y$ would be an array of probabilities that the image contains a certain species, and $f(x)$ would be a tranformation between the spaces containing the images and the probabilities. If you find this interesting, I would encourage you to dig into some advanced linear algebra, but it's not really necessary here.

### Approximating an Arbitrary Function

The way neural networks approximate functions is relatively straightforward. The structure of the neural network defines what features our function includes. For example, we could construct a neural network which implements the function $f(x)=ax+b$. 

![image](https://github.com/user-attachments/assets/c7dc4b0e-3899-4e66-8ba2-ef3dc57c32f2)

In the image above, the points represent the training data and the line is a function that approximates the relationship between $x$ and $y$. We can see that there is a value for $a$, the slope, and $b$, the intercept, which best represents the data. We want our neural network to search for those values.

To do this, we need a way to test one choice of $a$ and $b$ against another. This is called a cost function and calculates how far the data points are from the line, known as the mean-squared error. Each time the network is trained, the cost (or error) should decrease until we find the best value for $a$ and $b$. There are other ways to implement the cost function, but mean-square error is the most straightforward.

In order to make the error decrease, we need an optimization algorithm. For this implementation, I've chosen gradient descent. To understand gradient descent, imagine you're standing on a topographical map, each coordinate on the map represents a choice for $a$ and $b$, and the elevation represents the error. The goal is to find the lowest point on the map. 

![image](https://github.com/user-attachments/assets/bef9a22b-3a97-4ec7-99c7-6bf97cd32682)

We could blindly step in a direction, see whether we went up or down, and continue moving in that direction until we hit a low point (minimum) on the map. Though this will lead to finding a good choice of $a$ and $b$, it is tedious and unecessary. Thankfully, due to something called backpropagation, we can actually calculate which direction we should go. 

Backpropagation is a relatively complicated algorithm, made possible by computational graphs and recursion-- important concepts but unnecessary to understand the basics. To get the gist, just know that each layer of a neural network represents some mathematical operator within our function. Using the chain rule from calculus, we can find the contribution to the slope of the map from each individual operation. In other words, we can calculate how much $a$ and $b$ in $f(x)=ax+b$ individually effect the shape of the map around a specific point. This tells us which direction we should step in order to decrease the cost most efficiently.

An important parameter for running this algorithm is the Learning Rate. The learning rate determines how far we move each time we take a step. If the learning rate is too big, we run the risk of stepping over the lowest point on the map and never finding the right values for $a$ and $b$, but if the learning rate is too small, we'll barely move at all.

Now we know the pieces required to create and train a neural network:

1. Define a function such as $f(x)=ax+b$
2. Pass a data point, $x$, to the network
3. Compare the output to the correct value of $y$ with the cost function
4. Backpropagate the output to update the parameters $a$ and $b$
5. Repeat until the cost is minimized

Using this framework, we can train the network using any labeled data to approximate a function that maps the input to the correct output. An important consideration is that not all datasets will have a linear relationship that can be described by $f(x)=ax+b$, so how can we expand the model to fit these datasets?

Activation functions are functions that are often placed at the output of linear layers to introduce non-linearity. A commonly used activation function is known as the Rectified Linear Unit (ReLU), which clamps all negative outputs to zero, as shown in the figure below.

![image](https://github.com/user-attachments/assets/14c14271-6bb2-4a90-803f-6f8405969ae8)

Adding a ReLU layer allows our network to approximate non-linear datasets, as seen in the figure below.

![image](https://github.com/user-attachments/assets/742d94b9-b4d8-492c-b6ce-d374e7afdcdd)

With an activation function included, we can now train a network to approximate non-linear relationships. Note that these examples show the behavior with a single-dimensional input and output. The challenge in  machine learning is to scale this algorithm to work well with high-dimensional datasets.

### Implementing the Algorithm in Python

The first step to implement this algorithm in Python is to define a Tensor class. A tensor is an algebraic object which describes the relationships between objects, such as arrays, regardless of the basis. If it's unclear what this means, the Wikipedia page on Tensors has a lot of great information. For our application, we need a tensor to store data, such as the input or the intermediate results after an operation, and to encode the gradient or slope of the map at each layer of the network.

```
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
```
Let's look at each section of the Tensor class to better understand what it does. 

#### init
The init function takes in the data we are turning into a tensor and a flag which determines whether the tensor requires gradient calculations. The next few lines zero the gradients (self.grad = None), sets the backward function to do nothing by default (self._backward = lambda: None), creates a list to save the previous tensors in the network for backpropagation (self.prev = []), and saves the shapes of the tensors for backpropagation (self._saved_tensors = {}).

#### Operations
The next few functions define operations that we can perform with tensor objects. The important thing to note here is that each function behaves as expected (such as adding two arrays), but also attaches a function called _backward which implements the chain rule to determine the gradient contribution from each operation. This functionality is the reason we must define things as tensors.

#### Backward Function
The final part is the backward function. This step creates a computational graph using the list of previous tensors to link each tensor to its neighbor in the network. When the backward function runs, it propagates the result of the cost function back through each layer of the network to determine the gradients and update the parameters.

### Linear Layer

The linear layer is very straightforward. It simply defines random weights and biases in the shape of the input, applies them to the input, and provides the result as the output. It also provides a method for handling tensors with no gradients.

```
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
```

### ReLU

The ReLU function behaves very similarly to the linear function, but applies a max function to zero any inputs that are negative. The ReLU also requires a _backward function for the gradients to propagate through the network.

```
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
```

### Cost Functions

Two cost functions are defined. First is the mean-squared error. This simply takes the result from the network and computes the error between the output and the target (correct answer). Think of this as computing the distance from each point to the line $f(x)=ax+b$ in the sections above.

```
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
```

The second cost function is cross entropy loss, which is a bit more complicated but ends up making the results more interpretable for lots of cases. Basically, cross entropy loss behaves similar to MSE, but provides the outputs as a likelihood that the input belongs to a certain set. In our discussion of identifying the species of an animal, the output may contain a vector with $n$ entries where each entry represents a species, and the value of the entry is the likelihood that the image contains the corresponding species. In this implementation, values close to zero (low cost) represent high probability.

```
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
```

### Training a Simple Network

With that, we have all of the pieces together to train a basic neural network. This can be accomplished with the following code.

```
def train_network(epochs=1000):
    # Initialize network
    linear1 = Linear(2, 3)  # 2 inputs -> 3 hidden units
    relu = ReLU()
    linear2 = Linear(3, 1)  # 3 hidden units -> 1 output
    
    # Create synthetic data with proper shapes
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)  # (2, 2)
    target = Tensor([[5.0], [7.0]])  # (2, 1)
    
    learning_rate = 0.01
    for epoch in range(epochs):
        # Forward pass
        out1 = linear1(x)         # Shape: (2, 3)
        out2 = relu(out1)         # Shape: (2, 3)
        output = linear2(out2)    # Shape: (2, 1)
        
        # Compute loss
        loss = MSELoss()(output, target)
        
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
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.data}")

if __name__ == "__main__":
    train_network()
```

This code defines a basic network structure with a linear layer, a ReLU activation function, a second linear layer, and the MSE cost function. The most important aspect to getting the network to train is verifying the input and output sizes of each layer. In this case, the input and output sizes are determined by the synthetic data we create. We can see that the input, $x$, is $2x2$ and the output (target) is $2x1$. Therfore, we need the first layer to accept a $2x2$ input and we need the final layer to output a single value for the cost function to work.

Some important pieces are the hyperparameters. In this case, the hyperparameters are the learning rate and the number of epochs. Each epoch is a pass through the training loop and the learning rate determines how strongly the gradient is applied to update the parameters of each layer.

We can see five steps in the training loop. First the input is passed forward through each layer consecutively. Next, the cost is calculated using MSELoss. Then, the gradients are set to zero to prepare for backpropagation. Backpropagation is then invoked, which recursively calls the Tensor backward function and uses the linkage between the tensors in the computational graph to call each layer's _backward function and compute all of the gradients. Finally, once the gradients are computed, each gradient is multiplied by the learning rate and is subtracted from the weights and biases before repeating the process.

We should see the cost decrease with each epoch as the biases and weights are updated to find the function that maps the input to the output. The results are:

```
Epoch 0, Loss: 28.488428331583762
Epoch 100, Loss: 0.6536107833634941
Epoch 200, Loss: 0.3132867062447044
Epoch 300, Loss: 0.15382633910791707
Epoch 400, Loss: 0.07535157966693172
Epoch 500, Loss: 0.036038541655822286
Epoch 600, Loss: 0.016730325202864743
Epoch 700, Loss: 0.007531049636108783
Epoch 800, Loss: 0.0032988132583287714
Epoch 900, Loss: 0.0014139331992504647
```

### Using a Real Dataset

A popular problem in machine learning is to achieve good performance on the MNIST dataset. MNIST is a set of digitized, handwritten numbers which are labeled. The goal is to get the neural network to properly identify the number in the image. MNIST is included in the scikit Python package. The following code can be used to load the dataset.

```
from sklearn.datasets import fetch_openml
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
```

This code loads the MNIST dataset and divides it into a training set and a test set. The training set is used for calculations in the training loop and the test set is used to evaluate accuracy once the model is trained. This is an important concept in ML and it's of utmost importance that the training and test sets do not overlap. This will lead to inaccurate results. We must assume that the test data is similar enough to the training data that the parameters produced by the training loop can map the test inputs to the test labels. 

This relationship can be improved with batching. Batching arbitrarily divides the training data into batches to avoid having unwanted relationships in the data. For example, if the last 500 training points all contained the number 9, the model would show a preference for the number 9. 

Batching improves accuracy without increasing complexity and helps to avoid local minima in the cost function. Imagine if there was a small crater at the top of a hill in the topographical map. Gradient descent may get stuck in that crater because the cost increases in every direction around the crater, yet the lowest cost lies in a valley outside the crater. There are other methods to avoid this, such as injecting noise into the model or using a more complex method to randomly initialize the parameters at the beginning of the training loop. The following code implements batch loading.

```
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
```

We're almost ready to train on the MNIST dataset, but we need a way to evaluate the accuracy. There are many ways to do this, but the following code is a very basic implementation which takes an average of how many times the model gives the correct answer.

```
def accuracy(predictions, targets):
    return np.mean(np.argmax(predictions.data, axis=1) == targets.data)
```

Now, we can define our network and run the training loop.

```
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
    epochs = 100
    learning_rate = 0.05

    # Initialize network
    linear1 = Linear(input_size, hidden_size)
    relu = ReLU()
    linear2 = Linear(hidden_size, output_size)
    loss_fn = CrossEntropyLoss()
    
    print(f"Training on {len(train_images)} samples, testing on {len(test_images)} samples")
    
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
            print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    train_mnist()
```

This code works identically to the previous training example, but implements the MNIST loader, the batch loader, and provides evaluation results. The results of training are shown below for 10 epochs.

```
Loading MNIST data...
Loading MNIST from scikit-learn...
Training on 5000 samples, testing on 1000 samples
Epoch 1/10, Loss: 2.3536, Accuracy: 0.1074
Epoch 2/10, Loss: 2.1103, Accuracy: 0.2880
Epoch 3/10, Loss: 1.8898, Accuracy: 0.4978
Epoch 4/10, Loss: 1.6721, Accuracy: 0.6222
Epoch 5/10, Loss: 1.4759, Accuracy: 0.6888
Test Accuracy: 0.6660
Epoch 6/10, Loss: 1.3108, Accuracy: 0.7332
Epoch 7/10, Loss: 1.1755, Accuracy: 0.7622
Epoch 8/10, Loss: 1.0667, Accuracy: 0.7836
Epoch 9/10, Loss: 0.9784, Accuracy: 0.8004
Epoch 10/10, Loss: 0.9057, Accuracy: 0.8126
Test Accuracy: 0.7790
```

This can be improved greatly by tuning the learning rate and using epochs. Shown below are the results when using 2000 epochs with the same learning rate.

```
Epoch 1996/2000, Loss: 0.0064, Accuracy: 1.0000
Epoch 1997/2000, Loss: 0.0064, Accuracy: 1.0000
Epoch 1998/2000, Loss: 0.0064, Accuracy: 1.0000
Epoch 1999/2000, Loss: 0.0064, Accuracy: 1.0000
Epoch 2000/2000, Loss: 0.0064, Accuracy: 1.0000
Test Accuracy: 0.9210
```

As we can see, the accuracy greatly improved, but there are some important points to take away from these results. The training accuracy remains at 1, meaning the model has perfectly learned the training data. Because the training accuracy is 1, the the loss will remain constant and the test accuracy will not increase. The goal of tuning these training loops is generally to have the training accuracy slowly increase to 1 as the cost decreases. We can achieve this by either using fewer epochs or decreasing the learning rate. If the training accuracy never reaches 1, we should use more epochs or increase the learning rate. 

There's no purpose in training beyond the point where the training accuracy reaches 1 because the model has essentially learned everything it can about the data. To achieve better performance, we can make the model more complex by adding more layers or making the hidden layers bigger. Let's double the hidden layer size and see what happens.

```
Epoch 146/150, Loss: 0.0224, Accuracy: 0.9994
Epoch 147/150, Loss: 0.0221, Accuracy: 0.9994
Epoch 148/150, Loss: 0.0218, Accuracy: 0.9992
Epoch 149/150, Loss: 0.0215, Accuracy: 0.9996
Epoch 150/150, Loss: 0.0213, Accuracy: 0.9996
Test Accuracy: 0.9160
```

We can now see that the training accuracy approaches 1 and achieved similar accuracy with fewer epochs, but it would be nice to visualize what's happening here. The following updated training loop produces a plot of the training and test accuracy versus epoch.

```
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
```

As we can see in the plot below, the training accuracy very quickly approaches 1, preventing the test accuracy from increasing beyond a certain point. I would encourage you to try implementing some methods for improving this behavior, such as injecting noise or using a different batching or random initialization method.

![image](https://github.com/user-attachments/assets/733b301c-1abb-4c77-b0a9-2c9fe6fbac75)

These plots can be incredibly useful for figuring out what's preventing your model from achieving high test accuracy. You can easily identify over/under-training and use them as a metric for tweaking parameters.

### Wrapping Up

Thank you for following along and I hope this provided some useful information-- even if it just gives more insight into what tools like PyTorch have going on under the hood. I've included the library, scripts for training on a synthetic dataset and the MNIST set, as well as a bunch of debugging scripts that may be of help if you run into trouble doing your own project. 

#### P.S. - I take responsibility for all of the code presented here. There are certainly many other resources which contain similar approaches and code structures, but I did not reference any other piece of work when producing this. ChatGPT was used frequently for debugging, giving tips to resolve issues with backpropagation, and producing the plots presented above. Because ChatGPT is trained on resources available online, methods for overcoming certain challenges may be similar to other works of which I am unaware. For that reason, I intend this to be purely educational and not a tool or idea that I have invented.



