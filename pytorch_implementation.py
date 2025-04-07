import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Callable
from torch.optim import Optimizer


class CustomAdamW(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        weight_decay: float = 0.0,
        eps: float = 1e-8,
    ):
        if lr <= 0.0:
            raise ValueError("Learning rate must be positive.")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data

                # state init
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                # update biased first and second moment estimates
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # compute bias-corrected moments
                bias_corr1 = 1 - beta1**t
                bias_corr2 = 1 - beta2**t
                denom = (exp_avg_sq / bias_corr2).sqrt_().add_(eps)
                step_size = lr / bias_corr1

                # gradient step
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # decoupled weight decay
                if wd != 0:
                    p.data.add_(p.data, alpha=-lr * wd)
        return loss


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


if __name__ == "__main__":
    X = torch.randn(1000, 10)
    y = torch.randn(1000, 1)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = Model()
    loss_fn = nn.MSELoss()
    optimizer = CustomAdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    for epoch in range(50):
        for data, target in dataloader:
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
