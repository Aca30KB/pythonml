# PythonML: Fundamental Machine Learning Algorithms

## 🤖 Overview
This repository contains fundamental Machine Learning algorithms implemented in **pure Python** (using NumPy for numerical operations). The primary goal is to provide a comprehensive reference for the core mathematical principles, **numerical stability**, and implementation details of common ML techniques.

The implementation focuses on high code clarity and serves as a crucial component of my **cross-language benchmarking** study, comparing performance and convergence behavior against C/C++, Julia, and Go implementations.

---

## 🔬 Implemented Algorithms
The project covers a range of supervised and unsupervised learning algorithms:

### Supervised Learning
* **Linear Regression:** Implementation of Ordinary Least Squares (OLS) and Gradient Descent solvers.
* **Logistic Regression:** Binary classification model focusing on the sigmoid function and cross-entropy loss.
* **Neural Networks (MLP):** Multi-Layer Perceptron implemented with customizable hidden layers, ReLU/Sigmoid activation, and backpropagation from scratch.

### Unsupervised Learning
* **K-Means Clustering:** Standard iterative clustering algorithm.
* **Principal Component Analysis (PCA):** Implementation for dimensionality reduction and data visualization.

---

## 🚀 Key Implementation Features
The code is designed to demonstrate deep understanding beyond standard library calls:

* **From-Scratch Implementation:** All core calculations (e.g., gradient calculation, backpropagation, and loss functions) are implemented directly without relying on high-level ML frameworks (like Scikit-learn or PyTorch).
* **NumPy Optimization:** Leveraging NumPy's vectorized operations for efficient matrix algebra, which is crucial for maximizing Python's numerical performance.
* **Numerical Rigor:** Emphasis is placed on managing **numerical stability** (e.g., handling log operations in cross-entropy loss) and ensuring accurate **gradient computation** for training loops.
* **Modular Design:** Code is structured to easily integrate into larger systems, facilitating cross-language comparisons and deployment in automated workflows.

---

## 🛠️ Requirements and Setup

The project requires Python 3.8+ and the following library:

```bash
pip install numpy
