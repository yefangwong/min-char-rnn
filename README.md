# SimpleRNN

A minimal character-level Recurrent Neural Network (RNN) implementation in Java.

## Overview

This project implements a simple RNN from scratch to learn character sequences and generate text. It demonstrates the core concepts of RNNs including forward propagation, backpropagation through time (BPTT), and parameter updates using gradient descent.

## Features

- Character-level modeling with a vocabulary built from input text.
- Single hidden layer with tanh activation.
- Training with cross-entropy loss and smooth loss tracking.
- Forward and backward passes implemented manually without external ML libraries.
- Sampling method to generate text from learned probabilities.

## Usage

- Initialize the RNN with training data (a string).
- Call the `train` method with the number of iterations to train the model.
- Use the trained model to generate text by sampling from output probabilities.

## Key Components

- Weight matrices: `wxh` (input to hidden), `whh` (hidden to hidden), `why` (hidden to output).
- Bias vectors: `bh` (hidden layer), `by` (output layer).
- Forward pass computes hidden states and output probabilities.
- Backward pass computes gradients for all parameters.
- Parameter updates use gradient descent with a fixed learning rate.

## Notes

- Sequence length is fixed to 1 for simplicity.
- The model uses one-hot encoding for input characters.
- Random initialization of weights with small Gaussian noise.
- Designed for educational purposes to understand RNN internals.

## License

MIT License
