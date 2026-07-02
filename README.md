# ✍️ Handwritten Digit Recognizer

> **MLP built from scratch (NumPy backprop + Adam) vs CNN (PyTorch LeNet) on MNIST**

A deep learning portfolio project that implements a full multi-layer perceptron from mathematical foundations, trains a CNN using PyTorch, compares both models, and visualizes what the CNN actually learns.

---

## 🔥 Results

| Model | Test Accuracy | Parameters |
|---|---|---|
| **MLP (NumPy from scratch)** | **98.24%** | ~235K |
| **CNN (PyTorch LeNet)** | **98.73%** | ~44K |

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python train.py        # downloads MNIST, trains both models (~3 min)
streamlit run app.py
```

---

## 🧠 What's built from scratch

### MLP (`my_library/mlp.py`)
Everything manually implemented in NumPy:

- **Forward pass** — Linear layer, ReLU, Softmax
- **Loss** — Categorical cross-entropy
- **Backpropagation** — Full chain rule: `dZ = ŷ − y`, `dW = A.T · dZ / m`, `dA = dZ · W.T`
- **Optimizers** — SGD, SGD+Momentum, Adam (with bias correction)
- **Regularization** — L2 weight decay + Dropout
- **Weight init** — He initialization for ReLU layers

### Metrics (`my_library/metrics.py`)
- Accuracy, Confusion Matrix, Per-class Precision/Recall/F1 — all from scratch

### CNN (`my_library/cnn.py`)
- LeNet-style architecture built with PyTorch
- Filter + feature map extraction for visualization
- Training loop with Adam + LR scheduler

---

## 🖥️ App Pages

| Page | What you see |
|---|---|
| **✍️ Draw & Predict** | Browse test set samples, view probability distributions from both models |
| **📊 Model Dashboard** | Learning curves, confusion matrices, per-class F1 comparison |
| **🔬 CNN Internals** | Conv1 learned filters, Conv1 + Conv2 feature maps per digit, model mistakes |
| **📖 Architecture** | Math behind backprop + Adam, project structure, highlights |

---

## 🗂️ Project Structure

```
digit_recognizer/
│
├── my_library/
│   ├── mlp.py          # MLP: DenseLayer, DropoutLayer, MLPClassifier (NumPy only)
│   ├── cnn.py          # CNN: DigitCNN (PyTorch), train_cnn, predict_cnn
│   └── metrics.py      # accuracy, confusion_matrix, precision_recall_f1_per_class
│
├── data/               # MNIST downloaded here by train.py
├── models/             # mlp_scratch.pkl + cnn_pytorch.pth + eval_data.pkl
│
├── train.py            # Full training pipeline
├── app.py              # Streamlit app
├── requirements.txt
└── README.md
```

---

## ⚙️ Hyperparameters

### MLP
| Parameter | Value |
|---|---|
| Architecture | 784 → 256 → 128 → 10 |
| Activation | ReLU (hidden), Softmax (output) |
| Optimizer | Adam (β₁=0.9, β₂=0.999) |
| Learning rate | 0.001 |
| L2 regularization | 1e-4 |
| Dropout keep prob | 0.8 |
| Epochs | 20 |
| Batch size | 128 |

### CNN
| Parameter | Value |
|---|---|
| Architecture | Conv(6) → Pool → Conv(16) → Pool → FC(120) → FC(84) → FC(10) |
| Filter size | 5×5 |
| Optimizer | Adam |
| Learning rate | 0.001 (step decay ×0.5 every 5 epochs) |
| Dropout | 0.3 |
| Epochs | 10 |

---

## 👤 Author

**Sajid Ansari** — BTech CSE  
Built as a portfolio project for Microsoft ML internship applications.

---

## 📄 License
MIT
