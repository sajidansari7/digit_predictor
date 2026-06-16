
"""
CNN for MNIST digit classification using PyTorch.
Architecture: Conv → Pool → Conv → Pool → FC → FC → Softmax
 
Author : Sajid Ansari
Project: Handwritten Digit Recognizer
"""
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
 
 
class DigitCNN(nn.Module):
    """
    LeNet-style CNN for MNIST.
 
    Architecture
    ------------
    Input:  (B, 1, 28, 28)
    Conv1:  6 filters, 5×5  → (B, 6, 24, 24)
    Pool1:  2×2 MaxPool     → (B, 6, 12, 12)
    Conv2:  16 filters, 5×5 → (B, 16, 8, 8)
    Pool2:  2×2 MaxPool     → (B, 16, 4, 4)
    Flatten: (B, 256)
    FC1:    256 → 120
    FC2:    120 → 84
    FC3:    84  → 10
    """
 
    def __init__(self, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),       # → (B, 6, 24, 24)
            nn.ReLU(),
            nn.MaxPool2d(2),                       # → (B, 6, 12, 12)
            nn.Conv2d(6, 16, kernel_size=5),       # → (B, 16, 8, 8)
            nn.ReLU(),
            nn.MaxPool2d(2),                       # → (B, 16, 4, 4)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(84, 10),
        )
 
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
 
    def get_feature_maps(self, x):
        """Returns conv1 and conv2 feature maps for visualization."""
        maps = {}
        out = x
        for i, layer in enumerate(self.features):
            out = layer(out)
            if isinstance(layer, nn.Conv2d):
                maps[f"conv{len(maps)+1}"] = out.detach()
        return maps
 
    def get_conv_filters(self):
        """Returns conv1 weights for filter visualization."""
        return self.features[0].weight.detach().cpu().numpy()
 
 
def train_cnn(X_train, y_train, X_val, y_val,
              epochs=10, batch_size=64, lr=0.001,
              device=None, verbose=True):
    """
    Train DigitCNN and return (model, history_dict).
 
    X_train: numpy (N, 784) or (N, 28, 28)
    y_train: numpy (N,)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    # Reshape to (N, 1, 28, 28)
    def prep(X):
        X = X.reshape(-1, 1, 28, 28).astype(np.float32)
        return torch.tensor(X)
 
    Xt  = prep(X_train);  yt  = torch.tensor(y_train.astype(np.int64))
    Xv  = prep(X_val);    yv  = torch.tensor(y_val.astype(np.int64))
 
    train_ds = TensorDataset(Xt, yt)
    val_ds   = TensorDataset(Xv, yv)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=256)
 
    model     = DigitCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
 
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
 
    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            correct    += (logits.argmax(1) == yb).sum().item()
            total      += xb.size(0)
        scheduler.step()
        tr_loss = total_loss / total
        tr_acc  = correct / total
 
        # ── Validate ──
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits  = model(xb)
                loss    = criterion(logits, yb)
                val_loss    += loss.item() * xb.size(0)
                val_correct += (logits.argmax(1) == yb).sum().item()
                val_total   += xb.size(0)
        vl_loss = val_loss / val_total
        vl_acc  = val_correct / val_total
 
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)
 
        if verbose:
            print(f"Epoch {epoch:>2}/{epochs}  "
                  f"loss: {tr_loss:.4f}  acc: {tr_acc:.4f}  "
                  f"val_loss: {vl_loss:.4f}  val_acc: {vl_acc:.4f}")
 
    return model, history
 
 
def predict_cnn(model, X, device=None, batch_size=256):
    # Always infer device from where model weights actually live
    # This prevents cuda/cpu mismatch when model is loaded on cpu in the app
    device = next(model.parameters()).device
    model.eval()
    X_t  = torch.tensor(X.reshape(-1, 1, 28, 28).astype(np.float32))
    ds   = TensorDataset(X_t)
    dl   = DataLoader(ds, batch_size=batch_size)
    preds, probas = [], []
    with torch.no_grad():
        for (xb,) in dl:
            logits = model(xb.to(device))
            prob   = F.softmax(logits, dim=1).cpu().numpy()
            probas.append(prob)
            preds.append(np.argmax(prob, axis=1))
    return np.concatenate(preds), np.concatenate(probas, axis=0)
