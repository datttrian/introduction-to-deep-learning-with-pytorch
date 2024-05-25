# Introduction to Deep Learning with PyTorch

## Introduction to PyTorch, a Deep Learning Library

### Creating tensors and accessing attributes

Tensors are the primary data structure in PyTorch and will be the
building blocks for our deep learning models. They share many
similarities with NumPy arrays but have some unique attributes too.

In this exercise, you'll practice creating a tensor from a Python list
and displaying some of its attributes.

**Instructions**

- Begin by importing PyTorch.
- Create a tensor from the Python list `list_a`.

**Answer**

```{python}

```

### Creating tensors from NumPy arrays

Tensors are the fundamental data structure of PyTorch. You can create
complex deep learning algorithms by learning how to manipulate them.

The `torch` package has been imported, and two NumPy arrays have been
created, named `array_a` and `array_b`. Both arrays have the same
dimensions.

**Instructions**

- Create two tensors, `tensor_a` and `tensor_b`, from the NumPy arrays
  `array_a` and `array_b`, respectively.
- Subtract `tensor_b` from `tensor_a`.
- Perform an element-wise multiplication of `tensor_a` and `tensor_b`.
- Add the resulting tensors from the two previous steps together.

**Answer**

```{python}

```

### Your first neural network

In this exercise, you will implement a small neural network containing
two **linear** layers. The first layer takes an eight-dimensional input,
and the last layer outputs a one-dimensional tensor.

The `torch` package and the `torch.nn` package have already been
imported for you.

**Instructions**

- Create a neural network of linear layers that takes a tensor of
  dimensions \\1\times8\\ as input and outputs a tensor of dimensions
  \\1\times1\\.
- Use any output dimension for the first layer you want.

**Answer**

```{python}

```

### The sigmoid and softmax functions

The sigmoid and softmax functions are two of the most popular activation
functions in deep learning. They are both usually used as the last step
of a neural network. Sigmoid functions are used for binary
classification problems, whereas softmax functions are often used for
multi-class classification problems. This exercise will familiarize you
with creating and using both functions.

Let's say that you have a neural network that returned the values
contained in the `score` tensor as a pre-activation output. You will
apply activation functions to this output.

`torch.nn` is already imported as `nn`.

**Instructions**

Create a sigmoid function and apply it on `input_tensor` to generate a
probability.

Create a softmax function and apply it on `input_tensor` to generate a
probability.

**Answer**

```{python}

```

## Training Our First Neural Network with PyTorch

### Building a binary classifier in PyTorch

Recall that a small neural network with a single linear layer followed
by a sigmoid function is a binary classifier. It acts just like a
logistic regression.

In this exercise, you'll practice building this small network and
interpreting the output of the classifier.

The `torch` package and the `torch.nn` package have already been
imported for you.

**Instructions**

- Create a neural network that takes a tensor of dimensions 1x8 as
  input, and returns an output of the correct shape for binary
  classification.
- Pass the output of the linear layer to a sigmoid, which both takes in
  and return a single float.

**Answer**

```{python}

```

### From regression to multi-class classification

Recall that the models we have seen for binary classification,
multi-class classification and regression have all been similar, barring
a few tweaks to the model.

In this exercise, you'll start by building a model for regression, and
then tweak the model to perform a multi-class classification.

**Instructions**

- Create a neural network with exactly four linear layers, which takes
  the input tensor as input, and outputs a regression value, using any
  shapes you like for the hidden layers.

<!-- -->

- A similar neural network to the one you just built is provided,
  containing four linear layers; update this network to perform a
  multi-class classification with four outputs.

**Answer**

```{python}

```

### Creating one-hot encoded labels

One-hot encoding is a technique that turns a single integer label into a
vector of N elements, where N is the number of classes in your dataset.
This vector only contains zeros and ones. In this exercise, you'll
create the one-hot encoded vector of the label `y` provided.

You'll practice doing this manually, and then make your life easier by
leveraging the help of PyTorch! Your dataset contains three classes.

NumPy is already imported as `np`, and `torch.nn.functional` as `F`. The
`torch` package is also imported.

**Instructions**

- Manually create a one-hot encoded vector of the ground truth label `y`
  by filling in the NumPy array provided.
- Create a one-hot encoded vector of the ground truth label `y` using
  PyTorch.

**Answer**

```{python}

```

### Calculating cross entropy loss

Cross entropy loss is the most used loss for classification problems. In
this exercise, you will create inputs and calculate cross entropy loss
in PyTorch. You are provided with the ground truth label `y` and a
vector of `scores` predicted by your model.

You'll start by creating a one-hot encoded vector of the ground truth
label `y`, which is a required step to compare `y` with the scores
predicted by your model. Next, you'll create a cross entropy loss
function. Last, you'll call the loss function, which takes `scores`
(model predictions before the final softmax function), and the one-hot
encoded ground truth label, as inputs. It outputs a single float, the
loss of that sample.

`torch`, `torch.nn` as `nn`, and `torch.nn.functional` as `F` have
already been imported for you.

**Instructions**

- Create the one-hot encoded vector of the ground truth label `y` and
  assign it to `one_hot_label`.

**Answer**

```{python}

```

### Estimating a sample

In previous exercises, you used linear layers to build networks.

Recall that the operation performed by `nn.Linear()` is to take an input
\\X\\ and apply the transformation \\W\*X + b\\ ,where \\W\\ and \\b\\
are two tensors (called the weight and bias).

A critical part of training PyTorch models is to calculate gradients of
the weight and bias tensors with respect to a loss function.

In this exercise, you will calculate weight and bias tensor gradients
using cross entropy loss and a sample of data.

The following tensors are provded:

- `weight`: a \\2 \times 9\\-element tensor
- `bias`: a \\2\\-element tensor
- `preds`: a \\1 \times 2\\-element tensor containing the model
  predictions
- `target`: a \\1 \times 2\\-element one-hot encoded tensor containing
  the ground-truth label

**Instructions**

- Use the criterion you have defined to calculate the loss value with
  respect to the predictions and target values.
- Compute the gradients of the cross entropy loss.
- Display the gradients of the weight and bias tensors, in that order.

**Answer**

```{python}

```

### Accessing the model parameters

A PyTorch model created with the `nn.Sequential()` is a module that
contains the different layers of your network. Recall that each layer
parameter can be accessed by indexing the created model directly. In
this exercise, you will practice accessing the parameters of different
**linear** layers of a neural network. You won't be accessing the
sigmoid.

**Instructions**

- Access the `weight` parameter of the first **linear** layer.
- Access the `bias` parameter of the second **linear** layer.

**Answer**

```{python}

```

### Updating the weights manually

Now that you know how to access weights and biases, you will manually
perform the job of the PyTorch optimizer. PyTorch functions can do what
you're about to do, but it's helpful to do the work manually at least
once, to understand what's going on under the hood.

A neural network of three layers has been created and stored as the
`model` variable. This network has been used for a forward pass and the
loss and its derivatives have been calculated. A default learning rate,
`lr`, has been chosen to scale the gradients when performing the update.

**Instructions**

- Create the gradient variables by accessing the local gradients of each
  weight tensor.

**Answer**

```{python}

```

### Using the PyTorch optimizer

In the previous exercise, you manually updated the weight of a network.
You now know what's going on under the hood, but this approach is not
scalable to a network of many layers.

Thankfully, the PyTorch SGD optimizer does a similar job in a handful of
lines of code. In this exercise, you will practice the last step to
complete the training loop: updating the weights using a PyTorch
optimizer.

A neural network has been created and provided as the `model` variable.
This model was used to run a forward pass and create the tensor of
predictions `pred`. The one-hot encoded tensor is named `target` and the
cross entropy loss function is stored as `criterion`.

`torch.optim` as `optim`, and `torch.nn` as `nn` have already been
loaded for you.

**Instructions**

- Use `optim` to create an SGD optimizer with a learning rate of your
  choice (must be less than one) for the `model` provided.

**Answer**

```{python}

```

### Using the MSELoss

Recall that we can't use cross-entropy loss for regression problems. The
mean squared error loss (MSELoss) is a common loss function for
regression problems. In this exercise, you will practice calculating and
observing the loss using NumPy as well as its PyTorch implementation.

The `torch` package has been imported as well as `numpy` as `np` and
`torch.nn` as `nn`.

**Instructions**

- Calculate the MSELoss using NumPy.
- Create a MSELoss function using PyTorch.
- Convert `y_hat` and `y` to tensors and then float data types, and then
  use them to calculate MSELoss using PyTorch as `mse_pytorch`.

**Answer**

```{python}

```

### Writing a training loop

In `scikit-learn`, the whole training loop is contained in the `.fit()`
method. In PyTorch, however, you implement the loop manually. While this
provides control over loop's content, it requires a custom
implementation.

You will write a training loop every time you train a deep learning
model with PyTorch, which you'll practice in this exercise. The
`show_results()` function provided will display some sample ground truth
and the model predictions.

The package imports provided are: pandas as `pd`, `torch`, `torch.nn` as
`nn`, `torch.optim` as `optim`, as well as `DataLoader` and
`TensorDataset` from `torch.utils.data`.

The following variables have been created: `dataloader`, containing the
dataloader; `model`, containing the neural network; `criterion`,
containing the loss function, `nn.MSELoss()`; `optimizer`, containing
the SGD optimizer; and `num_epochs`, containing the number of epochs.

**Instructions**

- Write a for loop that iterates over the `dataloader`; this should be
  nested within a for loop that iterates over a range equal to the
  number of epochs.
- Set the gradients of the optimizer to zero.

**Answer**

```{python}

```

## Neural Network Architecture and Hyperparameters

### Implementing ReLU

The rectified linear unit (or ReLU) function is one of the most common
activation functions in deep learning.

It overcomes the training problems linked with the sigmoid function you
learned, such as the **vanishing gradients problem**.

In this exercise, you'll begin with a ReLU implementation in PyTorch.
Next, you'll calculate the gradients of the function.

The `nn` module has already been imported for you.

**Instructions**

- Create a ReLU function in PyTorch.

**Answer**

```{python}

```

### Implementing leaky ReLU

You've learned that ReLU is one of the most used activation functions in
deep learning. You will find it in modern architecture. However, it does
have the inconvenience of outputting null values for negative inputs and
therefore, having null gradients. Once an element of the input is
negative, it will be set to zero for the rest of the training. Leaky
ReLU overcomes this challenge by using a multiplying factor for negative
inputs.

In this exercise, you will implement the leaky ReLU function in NumPy
and PyTorch and practice using it. The `numpy` as `np` package, the
`torch` package as well as the `torch.nn` as `nn` have already been
imported.

**Instructions**

- Create a leaky ReLU function in PyTorch with a negative slope of 0.05.
- Call the function on the tensor `x`, which has already been defined
  for you.

**Answer**

```{python}

```

### Counting the number of parameters

Deep learning models are famous for having a lot of parameters. Recent
language models have billions of parameters. With more parameters comes
more computational complexity and longer training times, and a deep
learning practitioner must know how many parameters their model has.

In this exercise, you will calculate the number of parameters in your
model, first using PyTorch then manually.

The `torch.nn` package has been imported as `nn`.

**Instructions**

- Iterate through the model's parameters to update the total variable
  with the total number of parameters in the model.

**Answer**

```{python}

```

### Manipulating the capacity of a network

In this exercise, you will practice creating neural networks with
different capacities. The capacity of a network reflects the number of
parameters in said network. To help you, a `calculate_capacity()`
function has been implemented, as follows:

    def calculate_capacity(model):
      total = 0
      for p in model.parameters():
        total += p.numel()
      return total

This function returns the number of parameters in the your model.

The dataset you are training this network on has `n_features` features
and `n_classes` classes. The `torch.nn` package has been imported as
`nn`.

**Instructions**

Create a neural network with exactly three linear layers and less than
120 parameters, which takes `n_features` as inputs and outputs
`n_classes`.

Create a neural network with exactly four linear layers and more than
120 parameters, which takes `n_features` as inputs and outputs
`n_classes`.

**Answer**

```{python}

```

### Experimenting with learning rate

In this exercise, your goal is to find the optimal learning rate such
that the optimizer can find the minimum of the non-convex function
\\x^{4} + x^{3} - 5x^{2}\\ in ten steps.

You will experiment with three different learning rate values. For this
problem, try learning rate values between 0.001 to 0.1.

You are provided with the `optimize_and_plot()` function that takes the
learning rate for the first argument. This function will run 10 steps of
the SGD optimizer and display the results.

**Instructions**

- Try a small learning rate value such that the optimizer isn't able to
  get past the first minimum on the right.

<!-- -->

- Try a large learning rate value such that the optimizer skips past the
  global minimum at -2.

<!-- -->

- Based on the previous results, try a better learning rate value.

**Answer**

```{python}

```

### Experimenting with momentum

In this exercise, your goal is to find the optimal momentum such that
the optimizer can find the minimum of the following non-convex function
\\x^{4} + x^{3} - 5x^{2}\\ in 20 steps. You will experiment with two
different momentum values. For this problem, the learning rate is fixed
at 0.01.

You are provided with the `optimize_and_plot()` function that takes the
learning rate for the first argument. This function will run 20 steps of
the SGD optimizer and display the results.

**Instructions**

- Try a first value for the momentum such that the optimizer gets stuck
  in the first minimum.

<!-- -->

- Try a second value for the momentum such that the optimizer finds the
  global optimum.

**Answer**

```{python}

```

### Freeze layers of a model

You are about to fine-tune a model on a new task after loading
pre-trained weights. The model contains three linear layers. However,
because your dataset is small, you only want to train the last linear
layer of this model and freeze the first two linear layers.

The model has already been created and exists under the variable
`model`. You will be using the `named_parameters` method of the model to
list the parameters of the model. Each parameter is described by a name.
This name is a string with the following naming convention: `x.name`
where `x` is the index of the layer.

Remember that a linear layer has two parameters: the `weight` and the
`bias`.

**Instructions**

- Use an `if` statement to determine if the parameter should be frozen
  or not based on its name.
- Freeze the parameters of the first two layers of this model.

**Answer**

```{python}

```

### Layer initialization

The initialization of the weights of a neural network has been the focus
of researchers for many years. When training a network, the method used
to initialize the weights has a direct impact on the final performance
of the network.

As a machine learning practitioner, you should be able to experiment
with different initialization strategies. In this exercise, you are
creating a small neural network made of two layers and you are deciding
to initialize each layer's weights with the uniform method.

**Instructions**

- For each layer (`layer0` and `layer1`), use the uniform initialization
  method to initialize the weights.

**Answer**

```{python}

```

## Evaluating and Improving Models

### Using the TensorDataset class

In practice, loading your data into a PyTorch dataset will be one of the
first steps you take in order to create and train a neural network with
PyTorch.

The `TensorDataset` class is very helpful when your dataset can be
loaded directly as a NumPy array. Recall that `TensorDataset()` can take
one or more NumPy arrays as input.

In this exercise, you'll practice creating a PyTorch dataset using the
TensorDataset class.

`torch` and `numpy` have already been imported for you, along with the
`TensorDataset` class.

**Instructions**

- Convert the NumPy arrays provided to PyTorch tensors.
- Create a TensorDataset using the `torch_features` and the
  `torch_target` tensors provided (in this order).
- Return the last element of the dataset.

**Answer**

```{python}

```

### From data loading to running a forward pass

In this exercise, you'll create a PyTorch `DataLoader` from a pandas
DataFrame and call a model on this dataset. Specifically, you'll run a
**forward pass** on a neural network. You'll continue working with fully
connected neural networks, as you have done so far.

You'll begin by subsetting a loaded DataFrame called `dataframe`,
converting features and targets NumPy arrays, and converting to PyTorch
tensors in order to create a PyTorch dataset.

This dataset can be loaded into a PyTorch `DataLoader`, batched,
shuffled, and used to run a forward pass on a custom fully connected
neural network.

NumPy as `np`, pandas as `pd`, `torch`, `TensorDataset()`, and
`DataLoader()` have been imported for you.

**Instructions**

- Extract the features (`ph`, `Sulfate`, `Conductivity`,
  `Organic_carbon`) and target (`Potability`) values and load them into
  the appropriate tensors to represent features and targets.
- Use both tensors to create a PyTorch dataset using the dataset class
  that's quickest to use when tensors don't require any additional
  preprocessing.

**Answer**

```{python}

```

### Writing the evaluation loop

In this exercise, you will practice writing the evaluation loop. Recall
that the evaluation loop is similar to the training loop, except that
you will not perform the gradient calculation and the optimizer step.

The `model` has already been defined for you, along with the object
`validationloader`, which is a dataset.

**Instructions**

- Set the model to evaluation mode.
- Sum the current batch loss to the `validation_loss` variable.

**Answer**

```{python}

```

### Calculating accuracy using torchmetrics

In addition to the losses, you should also be keeping track of the
accuracy during training. By doing so, you will be able to select the
epoch when the model performed the best.

In this exercise, you will practice using the `torchmetrics` package to
calculate the accuracy. You will be using a sample of the facemask
dataset. This dataset contains three different classes. The
`plot_errors` function will display samples where the model predictions
do not match the ground truth. Performing such error analysis will help
you understand your model failure modes.

The `torchmetrics` package is already imported. The model `outputs` are
the probabilities returned by a softmax as the last step of the model.
The `labels` tensor contains the labels as one-hot encoded vectors.

**Instructions**

- Create an accuracy metric for a `"multiclass"` problem with three
  classes.
- Calculate the accuracy for each batch of the dataloader.

**Answer**

```{python}

```

### Experimenting with dropout

The dropout layer randomly zeroes out elements of the input tensor.
Doing so helps fight overfitting. In this exercise, you'll create a
small neural network with at least two linear layers, two dropout
layers, and two activation functions.

The `torch.nn` package has already been imported as `nn`. An
`input_tensor` of dimensions \\1 \times 3072\\ has been created for you.

**Instructions**

- Create a small neural network with one linear layer, one ReLU
  function, and one dropout layer, in that order.
- The model should take `input_tensor` as input and return an output of
  size 16.

<!-- -->

- Using the same neural network, set the probability of zeroing out
  elements in the dropout layer to `0.8`.

**Answer**

```{python}

```

### Implementing random search

Hyperparameter search is a computationally costly approach to experiment
with different hyperparameter values. However, it can lead to
performance improvements. In this exercise, you will implement a random
search algorithm.

You will randomly sample 10 values of the learning rate and momentum
from the uniform distribution. To do so, you will use the
`np.random.uniform()` function.

The `numpy` package has already been imported as `np`.

**Instructions**

- Randomly sample a learning rate factor between `2` and `4` so that the
  learning rate (`lr`) is bounded between \\10^{-2}\\ and \\10^{-4}\\.
- Randomly sample a momentum between 0.85 and 0.99.

**Answer**

```{python}

```
