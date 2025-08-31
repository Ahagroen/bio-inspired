import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import optim
from blending_nn import train_rl_blender
from util import (
    State,
    generate_parabola,
    position_error,
    predict,
    weighted_quadratic_regression,
)


# -----------------------------
# Neural network for trajectory
# -----------------------------


def sensitivity_analysis(model, policy):
    # 1. Sensitivity: Vary max azimuth with fixed elevation
    fixed_elevation = 60
    azimuths = np.linspace(20, 180, 45)  # vary max azimuth
    mse_az = []

    for x1 in azimuths:
        parabola = generate_parabola(x1, fixed_elevation)
        dummy_optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
        actor = Actor(parabola, x1, model, policy, dummy_optimizer, warmup=25)
        metrics = actor.train(plot=False)
        mse_az.append(metrics["mse"])

    # 2. Sensitivity: Vary max elevation with fixed azimuth
    fixed_azimuth = 125
    elevations = np.linspace(20, 90, 45)  # vary max elevation
    mse_el = []

    for E_max in elevations:
        parabola = generate_parabola(fixed_azimuth, E_max)
        dummy_optimizer = optim.Adam(policy.parameters(), lr=0.001)
        actor = Actor(
            parabola, fixed_azimuth, model, policy, dummy_optimizer, warmup=25
        )
        metrics = actor.train(plot=False)
        mse_el.append(metrics["mse"])

    # 3. Plot results
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(azimuths, mse_az, marker="o")
    plt.xlabel("Max Azimuth")
    plt.ylabel("MSE")
    plt.title(f"Sensitivity: Vary Max Azimuth (Elevation={fixed_elevation})")

    plt.subplot(1, 2, 2)
    plt.plot(elevations, mse_el, marker="o", color="orange")
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
        dummy_optimizer = optim.Adam(
            policy.parameters(), lr=0.001
        )  # Not used for testing
        actor = Actor(parabola, x1, model, policy, dummy_optimizer, warmup=25)

        print(f"Test Episode {i + 1}: x1 = {x1:.2f}, E_max = {E_max:.2f}")
        metrics = actor.train()  # Will plot and show results for each episode
        print(f"  MSE: {metrics['mse']:.4f}")
        print(f"  Avg Blend Factor (NN contribution): {metrics['avg_blend']:.4f}")
        all_mse.append(metrics["mse"])
        all_blends.append(metrics["avg_blend"])

    print("Overall Test Results")
    print(f"Average MSE: {np.mean(all_mse):.4f}")
    print(f"Average Blend Factor: {np.mean(all_blends):.4f}")
    print("Sensitivity Analysis")
    sensitivity_analysis(model, policy)


if __name__ == "__main__":
    main()
