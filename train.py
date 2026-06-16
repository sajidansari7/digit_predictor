"""
Train MLP (from scratch) + CNN (PyTorch) on MNIST.
Saves models + evaluation data for the Streamlit app.

Author : Sajid Ansari
Project: Handwritten Digit Recognizer
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import joblib
import torch
from torchvision import datasets, transforms

from my_library.mlp import MLPClassifier
from my_library.cnn import train_cnn, predict_cnn, DigitCNN
from my_library.metrics import accuracy_score, confusion_matrix, precision_recall_f1_per_class

os.makedirs("data",   exist_ok=True)
os.makedirs("models", exist_ok=True)

# ── Load MNIST ─────────────────────────────────────────────────────────────
print("Loading MNIST...")
transform = transforms.ToTensor()
train_ds  = datasets.MNIST("data", train=True,  download=True, transform=transform)
test_ds   = datasets.MNIST("data", train=False, download=True, transform=transform)

X_train_raw = train_ds.data.numpy().astype(np.float32) / 255.0
y_train     = train_ds.targets.numpy()
X_test_raw  = test_ds.data.numpy().astype(np.float32)  / 255.0
y_test      = test_ds.targets.numpy()

X_train_flat = X_train_raw.reshape(-1, 784)
X_test_flat  = X_test_raw.reshape(-1, 784)

X_val_flat  = X_train_flat[-10000:];  y_val   = y_train[-10000:]
X_tr_flat   = X_train_flat[:-10000];  y_tr    = y_train[:-10000]
X_val_raw   = X_train_raw[-10000:];   X_tr_raw = X_train_raw[:-10000]

print(f"Train: {X_tr_flat.shape}  Val: {X_val_flat.shape}  Test: {X_test_flat.shape}")

# ── MLP from scratch ───────────────────────────────────────────────────────
print("\nTraining MLP from scratch (NumPy)...")
mlp = MLPClassifier(
    layer_sizes=[784, 256, 128, 10], activations=["relu", "relu", "softmax"],
    optimizer="adam", learning_rate=0.001, l2=1e-4, dropout_rate=0.8,
    epochs=20, batch_size=128, random_state=42, verbose=True,
)
mlp.fit(X_tr_flat, y_tr, X_val=X_val_flat, y_val=y_val)

mlp_test_preds = mlp.predict(X_test_flat)
mlp_acc  = accuracy_score(y_test, mlp_test_preds)
mlp_cm   = confusion_matrix(y_test, mlp_test_preds)
mlp_per  = precision_recall_f1_per_class(y_test, mlp_test_preds)
print(f"\nMLP Test Accuracy: {mlp_acc:.4f}")

# ── CNN (PyTorch) ──────────────────────────────────────────────────────────
print("\nTraining CNN (PyTorch)...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model, cnn_history = train_cnn(
    X_tr_raw, y_tr, X_val_raw, y_val,
    epochs=10, batch_size=128, lr=0.001, device=device, verbose=True,
)
cnn_test_preds, _ = predict_cnn(cnn_model, X_test_raw)
cnn_acc = accuracy_score(y_test, cnn_test_preds)
cnn_cm  = confusion_matrix(y_test, cnn_test_preds)
cnn_per = precision_recall_f1_per_class(y_test, cnn_test_preds)
print(f"\nCNN Test Accuracy: {cnn_acc:.4f}")

# ── Save ───────────────────────────────────────────────────────────────────
print("\nSaving models...")

# Save MLP weights as .npz (few MB) instead of pickling (131MB)
mlp_weights = {}
for i, layer in enumerate(mlp.layers):
    if hasattr(layer, 'W'):
        mlp_weights[f"W_{i}"] = layer.W
        mlp_weights[f"b_{i}"] = layer.b
np.savez_compressed("models/mlp_weights.npz", **mlp_weights)

mlp_config = {
    "layer_sizes":  mlp.layer_sizes,
    "activations":  mlp.activations,
    "train_losses": mlp.train_losses,
    "val_losses":   mlp.val_losses,
    "train_accs":   mlp.train_accs,
    "val_accs":     mlp.val_accs,
}
with open("models/mlp_config.json", "w") as f:
    json.dump(mlp_config, f)

torch.save(cnn_model.state_dict(), "models/cnn_pytorch.pth")

filters    = cnn_model.get_conv_filters()
sample_idx = np.random.choice(len(y_test), 200, replace=False)

eval_data = {
    "mlp": {
        "accuracy":   mlp_acc,  "cm": mlp_cm.tolist(), "per_class": mlp_per,
        "train_loss": mlp.train_losses, "val_loss": mlp.val_losses,
        "train_acc":  mlp.train_accs,   "val_acc":  mlp.val_accs,
        "layer_sizes": mlp.layer_sizes,
    },
    "cnn": {
        "accuracy":   cnn_acc,  "cm": cnn_cm.tolist(), "per_class": cnn_per,
        "train_loss": cnn_history["train_loss"], "val_loss": cnn_history["val_loss"],
        "train_acc":  cnn_history["train_acc"],  "val_acc":  cnn_history["val_acc"],
        "filters":    filters.tolist(),
    },
    "sample_images":   X_test_raw[sample_idx].tolist(),
    "sample_labels":   y_test[sample_idx].tolist(),
    "sample_mlp_pred": mlp_test_preds[sample_idx].tolist(),
    "sample_cnn_pred": cnn_test_preds[sample_idx].tolist(),
}
joblib.dump(eval_data, "models/eval_data.pkl")

print(f"\nMLP weights size: {os.path.getsize('models/mlp_weights.npz')/1e6:.1f} MB")
print(f"CNN weights size: {os.path.getsize('models/cnn_pytorch.pth')/1e6:.1f} MB")
print(f"Eval data size  : {os.path.getsize('models/eval_data.pkl')/1e6:.1f} MB")
print(f"\nMLP: {mlp_acc*100:.2f}%   CNN: {cnn_acc*100:.2f}%")
print("Done!")
