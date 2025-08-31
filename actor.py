import torch
import numpy as np
import matplotlib.pyplot as plt
from util import State, position_error, predict, weighted_quadratic_regression


class Actor:
    def __init__(self, target, x1, model, policy, optimizer, warmup=25):
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

    def train(self, plot=True):
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
            plt.plot(
                self.azimuths[self.warmup :], self.reg_predictions, label="Regression"
            )
            plt.plot(self.azimuths, self.nn_predictions, label="Neural Net")
            x_true, y_true = [], []
            for i in self.azimuths:
                x_carry, y_carry = i, self.parabolic_target(i)
                x_true.append(x_carry)
                y_true.append(y_carry)
            plt.plot(x_true, y_true, label="True")
            plt.legend()
            plt.show()
        mse = np.mean(
            [
                (el - self.parabolic_target(az)) ** 2
                for az, el in zip(self.azimuths, self.last_elevation)
            ]
        )
        avg_blend = np.mean([x.detach().numpy() for x in self.saved_blend])
        return {"mse": mse, "avg_blend": avg_blend}

    def determine_next_state(self):
        # --- Save current state & error ---
        self.last_elevation.append(self.state.elevation)
        true_val = self.parabolic_target(self.state.azimuth)
        self.last_strength.append(
            position_error(self.state, (self.state.azimuth, true_val))
        )

        # --- Regression prediction ---
        if self.duration > self.warmup:
            errors = np.array(
                [
                    float(e.detach() if torch.is_tensor(e) else e)
                    for e in self.last_strength[: self.duration]
                ]
            )
            idx_sorted = np.argsort(errors)[:10]
            x_sel = np.array(self.azimuths[: self.duration])[idx_sorted]
            y_sel = np.array(self.last_elevation[: self.duration])[idx_sorted]
            w_sel = errors[idx_sorted]
            coeffs = weighted_quadratic_regression(x_sel, y_sel, w_sel)
            reg_pred = predict(self.azimuths[self.duration], coeffs)[0]
        else:
            reg_pred = self.state.elevation
        with torch.no_grad():
            nn_pred = (
                self.model(
                    torch.tensor(
                        [self.azimuths[self.duration] / 180.0], dtype=torch.float32
                    )
                ).numpy()[0]
                * 90.0
            )
        if 0 > reg_pred or reg_pred > 90:
            reg_pred = nn_pred
        duration = self.duration / 200.0
        state_feat = torch.tensor(
            [
                [
                    duration,
                    self.azimuths[self.duration] / 180.0,
                    nn_pred / 90.0,
                    reg_pred / 90.0,
                    abs(nn_pred - reg_pred) / 90.0,
                ]
            ],
            dtype=torch.float32,
        )

        blend = self.policy(state_feat)
        self.saved_blend.append(blend)
        blended = blend.item() * nn_pred + (1 - blend.item()) * reg_pred

        true_next = self.parabolic_target(self.azimuths[self.duration])
        reward = -(((blended - true_next) / 90.0) ** 2)
        self.rewards.append(reward)

        self.nn_predictions.append(nn_pred)
        self.reg_predictions.append(reg_pred)
        self.state = State(self.azimuths[self.duration], blended)
        self.duration += 1
