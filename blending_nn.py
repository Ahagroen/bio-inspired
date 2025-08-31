from torch import nn, optim
import numpy as np
import torch
from actor import Actor
from trajectory_nn import train_nn
from util import generate_parabola


class BlenderPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, state):
        params = self.net(state)
        return params


def finish_episode(actor, gamma=0.99):
    R = 0
    returns = []
    for r in actor.rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32)
    returns = (returns - returns.mean()) / (returns.std() + 1e-6)  # normalize

    policy_loss = []
    for log_prob, rx in zip(actor.saved_blend, returns):
        policy_loss.append(-log_prob * rx)
    policy_loss = torch.stack(policy_loss).sum()
    actor.optimizer.zero_grad()
    policy_loss.backward()
    actor.optimizer.step()

    actor.saved_blend.clear()
    actor.rewards.clear()


def train_rl_blender(episodes=200):
    print("Training Trajectory Neural Network")
    model = train_nn()  # your trajectory NN
    policy = BlenderPolicy()
    optimizer = optim.Adam(policy.parameters(), lr=0.005)

    print("Training Blending Policy via Reinforcement Learning")
    for ep in range(episodes):
        x1 = np.random.uniform(20, 180)
        parabola = generate_parabola(x1, np.random.uniform(20, 90))
        actor = Actor(parabola, x1, model, policy, optimizer, warmup=25)
        actor.train(plot=False)  # rollout
        if ep % 10 == 0:
            mean_reward = np.mean(actor.rewards)
            print(f"Episode {ep}, mean reward: {mean_reward:.2f}")
        finish_episode(actor)

    return model, policy
