import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import Tensor


class MLP(nn.Module):
    """Simple MLP"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def train_model(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    X: Tensor,
    y: Tensor,
    epochs: int = 100,
) -> list:
    loss_log = []
    for _ in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        loss_log.append(loss.item())
    return loss_log


def plot_loss_curves(losses_adam: list, losses_adamw: list):
    plt.figure(figsize=(10, 5))
    plt.plot(losses_adam, label="Adam")
    plt.plot(losses_adamw, label="AdamW")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig("assets/loss_curve_comparison.png")


def plot_decision_boundary(model: nn.Module, X: Tensor, y: Tensor, title: str = ""):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.01
    xx, yy = torch.meshgrid(
        torch.arange(x_min, x_max, h), torch.arange(y_min, y_max, h), indexing="ij"
    )
    grid = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], dim=1)
    with torch.no_grad():
        Z = model(grid).argmax(dim=1).reshape(xx.shape)
    plt.figure(figsize=(10, 5))
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", cmap=plt.cm.coolwarm)
    plt.title(title)
    plt.savefig(f"assets/{title.strip().lower()}.png")


if __name__ == "__main__":

    torch.manual_seed(42)

    X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)

    model_adam = MLP()
    model_adamw = MLP()
    criterion = nn.CrossEntropyLoss()

    adam = optim.Adam(model_adam.parameters(), lr=1e-2, weight_decay=1e-2)
    adamw = optim.AdamW(model_adamw.parameters(), lr=1e-2, weight_decay=1e-2)

    adam_loss = train_model(model_adam, adam, criterion, X_train, y_train)
    adamw_loss = train_model(model_adamw, adamw, criterion, X_train, y_train)

    plot_loss_curves(adam_loss, adamw_loss)
    plot_decision_boundary(model_adam, X_train, y_train, "Decision Boudnary - Adam")
    plot_decision_boundary(model_adamw, X_train, y_train, "Decision Boundary - AdamW")
