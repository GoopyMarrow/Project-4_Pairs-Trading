import numpy as np


class KalmanFilter:
    def __init__(self, n_dim: int, R: float, Q: np.ndarray, F: np.ndarray = None, P0: np.ndarray = None,
                 w0: np.ndarray = None):
        self.n_dim = n_dim
        self.R_obs_noise_var = R
        self.Q_proc_noise_cov = Q

        self.F_state_transition = F if F is not None else np.eye(n_dim)
        self.P_cov = P0 if P0 is not None else np.eye(n_dim) * 0.01
        self.w_state = w0 if w0 is not None else np.zeros(n_dim)

    def predict(self) -> tuple[np.ndarray, np.ndarray]:
        w_pred = self.F_state_transition @ self.w_state
        P_pred = self.F_state_transition @ self.P_cov @ self.F_state_transition.T + self.Q_proc_noise_cov
        return w_pred, P_pred

    def update(self, x_t_obs_matrix: np.ndarray, y_t_obs_value: float, w_pred: np.ndarray, P_pred: np.ndarray) -> tuple[
        np.ndarray, np.ndarray]:
        H_t = x_t_obs_matrix.reshape(1, -1)

        y_pred = H_t @ w_pred
        y_error = y_t_obs_value - y_pred

        S_t_innov_cov = H_t @ P_pred @ H_t.T + self.R_obs_noise_var

        if S_t_innov_cov == 0:
            K_t_gain = np.zeros_like(w_pred)
        else:
            K_t_gain = P_pred @ H_t.T / S_t_innov_cov

        w_upd = w_pred + K_t_gain.flatten() * y_error
        P_upd = (np.eye(self.n_dim) - K_t_gain @ H_t) @ P_pred

        self.w_state = w_upd
        self.P_cov = P_upd

        return w_upd, P_upd