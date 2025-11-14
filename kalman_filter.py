import numpy as np


class KalmanFilter:
    """
    A class for a general-purpose n-dimensional Kalman Filter.

    This implementation is designed for linear regression models where:
    y_t = H_t * w_t + R_t   (Observation Equation)
    w_t = F_t * w_{t-1} + Q_t (State Transition Equation)

    Attributes:
        n_dim (int): Number of state variables (e.g., 2 for [B0, B1]).
        R_obs_noise_var (float): Observation noise variance (R).
        Q_proc_noise_cov (np.ndarray): Process noise covariance matrix (Q).
        F_state_transition (np.ndarray): State transition matrix (F).
        P_cov (np.ndarray): State covariance matrix (P).
        w_state (np.ndarray): The current state vector (w).
    """

    def __init__(self, n_dim: int, R: float, Q: np.ndarray, F: np.ndarray = None, P0: np.ndarray = None,
                 w0: np.ndarray = None):
        """
        Initializes the Kalman Filter with given parameters.

        Args:
            n_dim: The dimension of the state vector.
            R: The observation noise variance (a scalar).
            Q: The process noise covariance matrix (n_dim x n_dim).
            F: The state transition matrix (n_dim x n_dim). Defaults to identity.
            P0: The initial state covariance matrix. Defaults to identity.
            w0: The initial state vector. Defaults to zeros.
        """
        self.n_dim = n_dim
        self.R_obs_noise_var = R
        self.Q_proc_noise_cov = Q

        # Set defaults if not provided
        self.F_state_transition = F if F is not None else np.eye(n_dim)
        self.P_cov = P0 if P0 is not None else np.eye(n_dim) * 0.01
        self.w_state = w0 if w0 is not None else np.zeros(n_dim)

    def predict(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Performs the "predict" step of the filter.

        Returns:
            A tuple of (w_pred, P_pred):
            - w_pred: The a priori predicted state.
            - P_pred: The a priori predicted covariance.
        """
        # Predict state: w_t|t-1 = F * w_t-1
        w_pred = self.F_state_transition @ self.w_state
        # Predict covariance: P_t|t-1 = F * P_t-1 * F^T + Q
        P_pred = self.F_state_transition @ self.P_cov @ self.F_state_transition.T + self.Q_proc_noise_cov
        return w_pred, P_pred

    def update(self, x_t_obs_matrix: np.ndarray, y_t_obs_value: float, w_pred: np.ndarray, P_pred: np.ndarray) -> tuple[
        np.ndarray, np.ndarray]:
        """
        Performs the "update" step of the filter.

        Args:
            x_t_obs_matrix: The observation matrix (H_t).
            y_t_obs_value: The observed value (y_t).
            w_pred: The predicted state from the predict() step.
            P_pred: The predicted covariance from the predict() step.

        Returns:
            A tuple of (w_upd, P_upd):
            - w_upd: The a posteriori updated state.
            - P_upd: The a posteriori updated covariance.
        """
        # Reshape H_t to be 1xN for matrix math
        H_t = x_t_obs_matrix.reshape(1, -1)

        # Calculate innovation (residual): y_t - H_t * w_pred
        y_pred = H_t @ w_pred
        y_error = y_t_obs_value - y_pred

        # Calculate innovation covariance: S_t = H_t * P_pred * H_t^T + R
        S_t_innov_cov = H_t @ P_pred @ H_t.T + self.R_obs_noise_var

        # Calculate Kalman Gain: K_t = P_pred * H_t^T * S_t^{-1}
        if S_t_innov_cov == 0:
            K_t_gain = np.zeros_like(w_pred)
        else:
            K_t_gain = P_pred @ H_t.T / S_t_innov_cov

        # Update state: w_t = w_pred + K_t * y_error
        w_upd = w_pred + K_t_gain.flatten() * y_error
        # Update covariance: P_t = (I - K_t * H_t) * P_pred
        P_upd = (np.eye(self.n_dim) - K_t_gain @ H_t) @ P_pred

        # Store the updated state and covariance for the next iteration
        self.w_state = w_upd
        self.P_cov = P_upd

        return w_upd, P_upd