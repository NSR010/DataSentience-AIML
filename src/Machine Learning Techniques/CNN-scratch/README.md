# Convolutional Neural Network (CNN) from Scratch

This repository contains a from-scratch implementation of a Convolutional Neural Network (CNN) in Python using **NumPy**. The project is designed for educational purposes to demonstrate the fundamental mechanics of a CNN, including the detailed implementation of forward and backward propagation through both convolutional and max-pooling layers.

The model is trained to classify handwritten digits from the `scikit-learn` `load_digits` dataset.

## Features
- **Pure NumPy Implementation:** No high-level deep learning frameworks like TensorFlow or PyTorch are used.
- **Full Backpropagation:** Includes the complete backward pass implementation for:
    - Convolutional Layers
    - MaxPooling Layers
    - Fully Connected Layers
- **Weight Initialization:** Uses He initialization for better training performance.
- **Standard Layers & Functions:** Implements ReLU activation, Softmax for output, and Cross-Entropy for loss calculation.

## Model Architecture
The network is built with the following architecture:
1.  `Input (8x8x1 image)`
2.  `->` **Conv Layer 1** (4 filters of 4x4, stride=1, pad=1) `->` **ReLU**
3.  `->` **Conv Layer 2** (8 filters of 2x2, stride=1, pad=0) `->` **ReLU**
4.  `->` **MaxPool Layer** (2x2 filter, stride=2)
5.  `->` **Flatten**
6.  `->` **Fully Connected Layer** (72 -> 60 units) `->` **ReLU**
7.  `->` **Fully Connected Layer** (60 -> 30 units) `->` **ReLU**
8.  `->` **Fully Connected Layer** (30 -> 10 units)
9.  `->` **Softmax Output**

## How to Run

### Prerequisites
Make sure you have Python and the following libraries installed:
```bash
pip install numpy matplotlib scikit-learn jupyterlab
```

### Execution
1.  Clone this repository to your local machine.
2.  Navigate to the src/Machine Learning Techniques/CNN-sratch directory
3.  Launch Jupyter Lab:
    ```bash
    jupyter lab
    ```
4.  Open and run the `cnn.ipynb` notebook to see the training process and evaluation.

## File Structure
```bash
├── convLayer.py         # Forward and Backward pass for Convolutional Layer
├── poolLayer.py         # Forward and Backward pass for MaxPooling Layer
└── cnn.ipynb            # Main notebook with model definition, training, and evaluation
```