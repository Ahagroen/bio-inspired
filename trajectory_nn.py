import numpy as np
import torch
from torch import nn, optim

from util import generate_parabola


class TrajectoryNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  # output: elevation
        )

    def forward(self, x):
        return torch.clamp(self.net(x), 0)


def generate_training_data(n_samples=600, seq_len=200):
    X = []
    Y = []
    for _ in range(n_samples):
        x2 = np.random.uniform(20, 180)
        E_max = np.random.uniform(20, 90)  # sample peak elevation
        parabola = generate_parabola(x2, E_max)

        azimuths = np.linspace(0, x2, seq_len)
        elevations = [parabola(x) for x in azimuths]

        # Normalize for NN training
        Y.extend(np.array(elevations) / 90.0)
        X.extend(azimuths / 180.0)
    return np.array(X, dtype=np.float32).reshape(-1, 1), np.array(
        Y, dtype=np.float32
    ).reshape(-1, 1)


def train_nn():
    X, Y = generate_training_data()
    X, Y = torch.tensor(X), torch.tensor(Y)

    model = TrajectoryNN()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.MSELoss()
    for epoch in range(250):
        optimizer.zero_grad()
        out = model(X)
        loss = loss_fn(out, Y)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss {loss.item():.4f}")
    return model
