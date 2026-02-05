# https://medium.com/@rossedwards_14988/visualizing-a-neural-network-using-manim-part-1-664387704a49
# Relevant imports
from manim import * # added this
import numpy as np
import pandas as pd

# Activation functions
def relu(X):
    return np.maximum(0,X)

# def softmax(X):
#     return np.exp(X)/sum(np.exp(X))

# stable version of the softmax
def softmax(X):
    Z = X - max(X)
    numerator = np.exp(Z)
    denominator = np.sum(numerator)
    return numerator/denominator

# Calculates the output of a given layer
def calculate_layer_output(w, prev_layer_output, b, activation_type="relu"):
    # Steps 1 & 2
    g = w @ prev_layer_output + b

    # Step 3
    if activation_type == "relu":
        return relu(g)
    if activation_type == "softmax":
        return softmax(g)

# Initialize weights & biases
def init_layer_params(row, col):
    w = np.random.randn(row, col)
    b = np.random.randn(row, 1)
    return w, b

# Calculate ReLU derivative
def relu_derivative(g):
    derivative = g.copy()
    derivative[derivative <= 0] = 0
    derivative[derivative > 0] = 1
    return np.diag(derivative.T[0])

def layer_backprop(previous_derivative, layer_output, previous_layer_output
                   , w, activation_type="relu"):
    # 1. Calculate the derivative of the activation func
    dh_dg = None
    if activation_type == "relu":
        dh_dg = relu_derivative(layer_output)
    elif activation_type == "softmax":
        dh_dg = softmax_derivative(layer_output)

    # 2. Apply chain rule to get derivative of Loss function with respect to:
    dL_dg = dh_dg @ previous_derivative # activation function

    # 3. Calculate the derivative of the linear function with respect to:
    dg_dw = previous_layer_output.T     # a) weight matrix
    dg_dh = w.T                         # b) previous layer output
    dg_db = 1.0                         # c) bias vector

    # 4. Apply chain rule to get derivative of Loss function with respect to:
    dL_dw = dL_dg @ dg_dw               # a) weight matrix
    dL_dh = dg_dh @ dL_dg               # b) previous layer output
    dL_db = dL_dg * dg_db               # c) bias vector

    return dL_dw, dL_dh, dL_db

def gradient_descent(w, b, dL_dw, dL_db, learning_rate):
    w -= learning_rate * dL_dw
    b -= learning_rate * dL_db
    return w, b

def get_prediction(o):
    return np.argmax(o)

# Compute Accuracy (%) across all training data
def compute_accuracy(train, label, w1, b1, w2, b2, w3, b3):
    # Set params
    correct = 0
    total = train.shape[0]

    # Iterate through training data
    for index in range(0, total):
        # Select a single data point (image)
        X = train[index: index+1,:].T

        # Forward pass: compute Output/Prediction (o)
        h1 = calculate_layer_output(w1, X, b1, activation_type="relu")
        h2 = calculate_layer_output(w2, h1, b2, activation_type="relu")
        o = calculate_layer_output(w3, h2, b3, activation_type="softmax")

        # If prediction matches label Increment correct count
        if label[index] == get_prediction(o):
            correct+=1

    # Return Accuracy (%)
    return (correct / total) * 100


# Calculate Softmax derivative
def softmax_derivative(o):
    derivative = np.diag(o.T[0])

    for i in range(len(derivative)):
        for j in range(len(derivative)):
            if i == j:
                derivative[i][j] = o[i] * (1 - o[i])
            else:
                derivative[i][j] = -o[i] * o[j]
    return derivative

class VisualiseNeuralNetwork(Scene):

    def construct(self):
        ### INITIALISE NEURAL NET PARAMETERS ###
        # Extract MNIST csv data into train & test variables
        # Load MNIST data from mnist.npz
        with np.load('mnist.npz') as data:
            train = np.concatenate((data['x_train'].reshape(-1, 28*28), data['y_train'].reshape(-1, 1)), axis=1)
            test = np.concatenate((data['x_test'].reshape(-1, 28*28), data['y_test'].reshape(-1, 1)), axis=1)

        # Extract the first column of the training dataset into a label array
        label = train[:, 0]
        # The train dataset now becomes all columns except the first
        train = train[:, 1:]

        # Initialise vector of all zeroes with 10 columns and the same number
        # of rows as the label array
        Y = np.zeros((label.shape[0], 10))

        # assign a value of 1 to each column index matching the label value
        Y[np.arange(0, label.shape[0]), label] = 1.0

        # Normalize test & training dataset
        train = train / 255
        test = test / 255

        # Set hyperparameter(s)
        learning_rate = 0.01

        # Set other params
        epoch = 0
        previous_accuracy = 100
        accuracy = 0

        # Randomly initialize weights & biases
        w1, b1 = init_layer_params(10, 784)  # Hidden Layer 1
        w2, b2 = init_layer_params(10, 10)  # Hidden Layer 2
        w3, b3 = init_layer_params(10, 10)  # Output Layer

        training_image = train[0:1, :].T
        input_image = self.create_input_image(training_image, left_shift=5)

        # Play Create animation and then wait for 2 seconds
        self.play(Create(input_image))
        self.wait(2)

    def create_input_image(self, training_image, left_shift):
        # Initialise params
        square_count = training_image.shape[0]
        rows = np.sqrt(square_count)

        # Create list of squares to represent pixels
        squares = [
            Square(fill_color=WHITE
                , fill_opacity=training_image[i]
                , stroke_width=0.5).scale(0.03)
            for i in range(square_count)
        ]

        # Place all the squares into a VGroup and arrange into a 28x28 grid
        group = VGroup(*squares).arrange_in_grid(rows=int(rows), buff=0)

        # Shift into correct position in the scene
        group.shift(left_shift * LEFT)

        return group