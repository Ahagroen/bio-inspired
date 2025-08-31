from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class State:
    azimuth: float
    elevation: float


# -----------------------------
# Neural network for trajectory
# -----------------------------
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


class Actor:
    def __init__(self, target, x1, model,policy,optimizer, warmup=25):
        self.state: State = State(0, 0)
        self.x2 = x1
        self.parabolic_target = target
        self.duration: int = 0
        self.azimuths = []
        self.last_elevation: list[float] = []
        self.last_strength: list[float] = []
        self.model = model
        self.warmup = warmup
        self.nn_predictions = []
        self.reg_predictions = []
        self.policy = policy
        self.optimizer = optimizer
        self.saved_blend = []
        self.rewards = []

    def train(self,plot=True):
        if self.model is not None:
            self.azimuths = np.linspace(self.state.azimuth, self.x2, 200)
            for i in range(self.warmup):
                with torch.no_grad():
                    az = torch.tensor([self.azimuths[i] / 180.0], dtype=torch.float32)
                    el = self.model(az).numpy()[0] * 90.0
                self.state = State(self.azimuths[i], el)
                self.last_elevation.append(self.state.elevation)
                true_val = self.parabolic_target(self.state.azimuth)
                self.nn_predictions.append(self.state.elevation)
                self.last_strength.append(
                    position_error(self.state, (self.state.azimuth, true_val))
                )
                self.duration += 1

        # Continue with hybrid prediction
        while self.duration < 200:
            self.determine_next_state()

        if plot:
            plt.plot(self.azimuths, self.last_elevation, label="Predicted")
            plt.plot(self.azimuths[self.warmup :], self.reg_predictions, label="Regression")
            plt.plot(self.azimuths, self.nn_predictions, label="Neural Net")
            x_true, y_true = [], []
            for i in self.azimuths:
                x_carry, y_carry = i, self.parabolic_target(i)
                x_true.append(x_carry)
                y_true.append(y_carry)
            plt.plot(x_true, y_true, label="True")
            plt.legend()
            plt.show()
        mse = np.mean([(el - self.parabolic_target(az)) ** 2 for az, el in zip(self.azimuths, self.last_elevation)])
        avg_blend = np.mean([x.detach().numpy() for x in self.saved_blend])
        return {"mse": mse, "avg_blend": avg_blend}


    def determine_next_state(self):
        # --- Save current state & error ---
        self.last_elevation.append(self.state.elevation)
        true_val = self.parabolic_target(self.state.azimuth)
        self.last_strength.append(position_error(self.state, (self.state.azimuth, true_val)))

        # --- Regression prediction ---
        if self.duration > self.warmup:
            errors = np.array([float(e.detach() if torch.is_tensor(e) else e)
                               for e in self.last_strength[:self.duration]])
            idx_sorted = np.argsort(errors)[:10]
            x_sel = np.array(self.azimuths[:self.duration])[idx_sorted]
            y_sel = np.array(self.last_elevation[:self.duration])[idx_sorted]
            w_sel = errors[idx_sorted]
            coeffs = weighted_quadratic_regression(x_sel, y_sel, w_sel)
            reg_pred = predict(self.azimuths[self.duration], coeffs)[0]
        else:
            reg_pred = self.state.elevation
        with torch.no_grad():
            nn_pred = self.model(
                torch.tensor([self.azimuths[self.duration] / 180.0], dtype=torch.float32)
            ).numpy()[0] * 90.0
        if 0 > reg_pred  or reg_pred > 90:
            reg_pred = nn_pred
        duration = self.duration/200.0
        state_feat = torch.tensor([[
            duration,
            self.azimuths[self.duration]/180.0,
            nn_pred/90.0,
            reg_pred/90.0,
            abs(nn_pred-reg_pred)/90.0
        ]], dtype=torch.float32)

        blend = self.policy(state_feat)
        self.saved_blend.append(blend)
        blended = blend.item() * nn_pred + (1-blend.item()) * reg_pred

        true_next = self.parabolic_target(self.azimuths[self.duration])
        reward = -((blended - true_next) / 90.0) ** 2
        self.rewards.append(reward)

        self.nn_predictions.append(nn_pred)
        self.reg_predictions.append(reg_pred)
        self.state = State(self.azimuths[self.duration], blended)
        self.duration += 1

def weighted_quadratic_regression(x, z, w):
    w = np.array(w)
    targets = z
    X = np.column_stack([x**2, x, np.ones_like(x)])
    W = np.diag(1.0 / (np.sqrt(w) + 1e-6))
    coeffs = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ targets)
    return coeffs


def predict(x_new, coeffs):
    x_new = np.atleast_1d(x_new)
    X_new = np.column_stack([x_new**2, x_new, np.ones_like(x_new)])
    return X_new @ coeffs


def position_error(pred: State, true):
    return (pred.elevation - true[1]) ** 2


def generate_parabola(x1, E_max: float = 90.0):
    if not (0 <= x1 <= 180):
        raise ValueError("x1 must be in range [0, 360]")
    if not (0 < E_max <= 90):
        raise ValueError("E_max must be in (0, 90]")
    denom = (x1 / 2) * (x1 / 2 - x1)
    if denom == 0:
        raise ValueError("Invalid x1: parabola degenerates")

    # Choose 'a' so that the peak elevation = E_max
    a = -E_max / abs(denom)

    def parabola(x):
        return a * (x) * (x - x1)

    return parabola


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


class BlenderPolicy(nn.Module):
    def __init__(self, input_dim=5, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),  
            nn.Sigmoid())

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
        x1 = np.random.uniform(20,180)
        parabola = generate_parabola(x1, np.random.uniform(20, 90))
        actor = Actor(parabola, x1, model, policy, optimizer, warmup=25)
        actor.train(plot=False)   # rollout
        if ep % 10 == 0:
            mean_reward = np.mean(actor.rewards)
            print(f"Episode {ep}, mean reward: {mean_reward:.2f}")
        finish_episode(actor)

    return model, policy

def sensitivity_analysis(model, policy):
    # 1. Sensitivity: Vary max azimuth with fixed elevation
    fixed_elevation = 60
    azimuths = np.linspace(20, 180, 45)  # vary max azimuth
    mse_az = []

    for x1 in azimuths:
        parabola = generate_parabola(x1, fixed_elevation)
        dummy_optimizer = optim.Adam(policy.parameters(), lr=0.001)
        actor = Actor(parabola, x1, model, policy, dummy_optimizer, warmup=25)
        metrics = actor.train(plot=False)
        mse_az.append(metrics['mse'])

    # 2. Sensitivity: Vary max elevation with fixed azimuth
    fixed_azimuth = 125
    elevations = np.linspace(20, 90, 45)  # vary max elevation
    mse_el = []

    for E_max in elevations:
        parabola = generate_parabola(fixed_azimuth, E_max)
        dummy_optimizer = optim.Adam(policy.parameters(), lr=0.001)
        actor = Actor(parabola, fixed_azimuth, model, policy, dummy_optimizer, warmup=25)
        metrics = actor.train(plot=False)
        mse_el.append(metrics['mse'])

    # 3. Plot results
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(azimuths, mse_az, marker='o')
    plt.xlabel("Max Azimuth")
    plt.ylabel("MSE")
    plt.title(f"Sensitivity: Vary Max Azimuth (Elevation={fixed_elevation})")

    plt.subplot(1,2,2)
    plt.plot(elevations, mse_el, marker='o', color='orange')
    plt.xlabel("Max Elevation")
    plt.ylabel("MSE")
    plt.title(f"Sensitivity: Vary Max Elevation (Azimuth={fixed_azimuth})")

    plt.tight_layout()
    plt.show()
def main():
    # 1. Train the trajectory NN
    np.random.seed = 20
    # 2. Train the blending policy
    model, policy = train_rl_blender(episodes=200)

    # 3. Evaluate on a test set
    print("Evaluating on Test Set")
    test_episodes = 5
    all_mse = []
    all_blends = []
    for i in range(test_episodes):
        x1 = np.random.uniform(25, 180)
        E_max = np.random.uniform(40, 85)
        parabola = generate_parabola(x1, E_max)

        # Use the trained model and policy for test
        dummy_optimizer = optim.Adam(policy.parameters(), lr=0.001)  # Not used for testing
        actor = Actor(parabola, x1, model, policy, dummy_optimizer, warmup=25)

        print(f"Test Episode {i+1}: x1 = {x1:.2f}, E_max = {E_max:.2f}")
        metrics = actor.train()  # Will plot and show results for each episode
        print(f"  MSE: {metrics['mse']:.4f}")
        print(f"  Avg Blend Factor (NN contribution): {metrics['avg_blend']:.4f}")
        all_mse.append(metrics['mse'])
        all_blends.append(metrics['avg_blend'])

    print("Overall Test Results")
    print(f"Average MSE: {np.mean(all_mse):.4f}")
    print(f"Average Blend Factor: {np.mean(all_blends):.4f}")
    print("Sensitivity Analysis")
    sensitivity_analysis(model,policy)
if __name__ == "__main__":
    main()
