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

This project can be run in two modes: training and inference.

### Prerequisites

- A Java Development Kit (JDK).

First, compile all the Java source files:

```bash
javac -d out src/*.java
```

### Training

To train a new model, run the `SimpleRNN` class without any arguments. The training data and number of iterations are currently hardcoded in the `main` method of `src/SimpleRNN.java`.

```bash
java -cp out SimpleRNN
```

This will train the model and save the weights to a file (e.g., `rnn_model_12000.dat`) in the project's root directory.

### Inference

To generate text with a pre-trained model, use the `--inference` flag. You need to provide the model file path and a seed character.

**Syntax:**
```bash
java -cp out SimpleRNN --inference <model_path> <seed_char> [generate_length]
```

- `<model_path>`: Path to the `.dat` model file (e.g., `rnn_model_12000.dat`).
- `<seed_char>`: The starting character for text generation.
- `[generate_length]` (Optional): The number of characters to generate. Defaults to 4.

**Example:**

To generate sequences such as "鮭魚生魚片" (5 characters starting with "鮭"), or "生魚片" (3 characters starting with "生"), ensure `rnn_model_12000.dat` is in the root directory and run:

```bash
java -cp out SimpleRNN --inference rnn_model_12000.dat 鮭 4
```

## Key Components

- Weight matrices: `wxh` (input to hidden), `whh` (hidden to hidden), `why` (hidden to output).
- Bias vectors: `bh` (hidden layer), `by` (output layer).
- Forward pass computes hidden states and output probabilities.
- Backward pass computes gradients for all parameters.
- Parameter updates use gradient descent with a fixed learning rate.

## Notes
*  Sequence length is fixed to 4 for simplicity.
*  The recurrent logic in the `forward` pass has been corrected to ensure proper propagation of the hidden state across time steps.
- The model uses one-hot encoding for input characters.
- Weight Initialization: Switched from random Gaussian noise to Xavier initialization (Glorot initialization) to optimize gradient flow for the Tanh activation function.
- Designed for educational purposes to understand RNN internals.

## License

BSD 3-Clause License (see `src/SimpleRNN.java` for details).
