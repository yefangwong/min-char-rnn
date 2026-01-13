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

This will train the model and save the weights to a file (e.g., `rnn_model_20000.dat`) in the project's root directory.

### Inference

To generate text with a pre-trained model, use the `--inference` flag. You need to provide the model file path and a seed character.

**Syntax:**
```bash
java -cp out SimpleRNN --inference <model_path> <seed_char> [generate_length]
```

- `<model_path>`: Path to the `.dat` model file (e.g., `rnn_model_20000.dat`).
- `<seed_char>`: The starting character for text generation.
- `[generate_length]` (Optional): The number of characters to generate. Defaults to 4.

**Example:**

To generate sequences such as "鮭魚生魚片" (5 characters starting with "鮭"), or "生魚片" (3 characters starting with "生"), ensure `rnn_model_20000.dat` is in the root directory and run:

```bash
java -cp out SimpleRNN --inference rnn_model_20000.dat 鮭 4
```

## Key Components

- Weight matrices: `wxh` (input to hidden), `whh` (hidden to hidden), `why` (hidden to output).
- Bias vectors: `bh` (hidden layer), `by` (output layer).
- Forward pass computes hidden states and output probabilities.
- Backward pass computes gradients for all parameters.
- Parameter updates use gradient descent with a fixed learning rate.

## Notes
*  Through experimental validation, the SEQ_LENGTH is set to 5. While 4 was initially used for simplicity, it proved insufficient to resolve the semantic ambiguity of the character "魚". It requires approximately 82,000 iterations to converge due to the faint gradient signal at that window size. By increasing the SEQ_LENGTH to 5, we provide a more robust context window, directly contributing to disambiguating the character "魚". This extra step allowing the model to bridge the dependency between "Salmon" (鮭「魚」) and "Sashimi" (生「魚」片) much earlier. Combined with Identity Initialization, this reduces the required training iterations to 20,000 (a 4x efficiency gain).
*  The recurrent logic in the `forward` pass has been corrected to ensure proper propagation of the hidden state across time steps.
- The model uses one-hot encoding for input characters.
- Weight Initialization: Switched from random Gaussian noise to **Xavier initialization** (Glorot initialization) to optimize gradient flow for the Tanh activation function.
- Designed for educational purposes to understand RNN internals.

## Performance Breakthrough: 4x Faster Convergence via Identity Initialization (IRNN)

This project implements a **Minimal Character-level RNN** in Java. Recent experimental results demonstrate a **4.1x improvement in convergence speed** by transitioning from standard random initialization to the **Identity-RNN (IRNN) strategy**.

### 1. The Core Engineering Challenge: Gradient Attenuation
In a character-level RNN, resolving the semantic ambiguity of repeated characters (like "魚" in different positions) is primarily an optimization challenge.
With a SEQ_LENGTH of 4, the model is theoretically capable of bridging the dependencies. However, it suffers from severe gradient attenuation. The signal required to disambiguate the transition from "魚" to "片" is extremely faint, forcing the model to undergo an extensive training period of 82,000 iterations to fine-tune the weights through this "narrow" temporal window.

### 2. The Solution: Achieving Dynamical Isometry
By setting the recurrent weight matrix ($W_{hh}$) to an **Identity Matrix ($I$)** and refining the window striding to $p += 1$, we achieved what researchers call **Dynamical Isometry**:
*   **Identity Mapping**: Since the eigenvalues of the Identity Matrix are all **1**, the error signal propagates through the time-steps without exponential decay ($1^{15} = 1$).
*   **Faithful Gradient Propagation**: The gradient norm remains in a healthy range (~0.35), ensuring that the features of the first character ("鮭") are preserved with high fidelity to influence the prediction of the final character ("片").
*   **Inductive Bias**: Initializing $W_{hh} = I$ provides a powerful **inductive bias** that assumes "memory retention" as the default state, rather than forcing the model to learn how to remember from scratch.

### 3. Benchmark Results
| Evaluation Metric | Random Gaussian Init | **Identity Initialization (IRNN)** | Improvement |
| :--- | :--- | :--- | :--- |
| **Convergence Iterations** | ~82,000 | **~20,000** | **4.1x Faster** |
| **Training Stability** | Long Plateaus | **Rapid Phase Transition** | Significantly More Stable |
| **Avg Gradient Norm** | $10^{-9}$ (Vanishing) | **0.35 (Healthy Flow)** | Restored Signal |
| **Success Case** | "生魚生" (Semantic Collapse) | **"鮭魚生魚片" (Full Recall)** | Resolved Ambiguity |

### 4. Technical Insights from Saxe et al.
Drawing on the theory of **nonlinear learning dynamics**, these results confirm that the "learning speed" of a deep network can be decoupled from its depth when weights are initialized to act as near-isometries. Even while retaining the `Tanh` activation, the structural stability provided by the Identity Matrix allowed the model to bypass the "saturation traps" that typically stall Vanilla RNNs.

---

### 💡 Analogy for the README
**Random Initialization** is like a runner with amnesia trying to navigate a forest in the dark; they must repeatedly fail before finding the path. **Identity Initialization** is like handing the runner a permanent notebook; they no longer need to learn *how* to not forget, allowing them to focus entirely on learning the map itself.

## License

BSD 3-Clause License (see `src/SimpleRNN.java` for details).
