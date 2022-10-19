import numpy as np

from control import care, dare

class LQRPolicy:
    def __init__(
        self,
        A,
        B,
        Q,
        R,
        reference_point,
        gamma=0.99,
        regularization_lambda=1.0,
        dt=1.0,
        is_continuous=False,
        seed=123,
    ):
        """
            Initialize LQR policy for an arbitrary system.

            INPUTS:
                A - Dynamics on state.
                B - Dynamics on action.
                Q - Cost coefficients for state.
                R - Cost coefficients for action.
                reference_point - Point to which system should tend.
                gamma - The discount factor of the system (assuming dt is 1.0).
                regularization_lambda - The regularization parameter of the policy.
                dt - The time step of the system.
                is_continuous - Boolean indicating whether or not A and B describe x or dx.
                seed - Random seed for reproducibility.

            OUTPUTS:
                Instance of LQR policy class.
        """

        self.seed = seed
        np.random.seed(self.seed)

        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.reference_point = reference_point
        self.gamma = gamma
        self.regularization_lambda = regularization_lambda
        self.dt = dt
        self.discount_factor = self.gamma**self.dt
        self.is_continuous = is_continuous

        self.discounted_A = self.A * np.sqrt(self.discount_factor)
        self.discounted_R = self.R / self.discount_factor

        if is_continuous:
            self.P = care(self.discounted_A, self.B, self.Q, self.discounted_R)[0]
        else:
            self.P = dare(self.discounted_A, self.B, self.Q, self.discounted_R)[0]

        self.C = np.linalg.inv(self.discounted_R + self.B.T @ self.P @ self.B) @ (self.B.T @ self.P @ self.discounted_A)
        self.sigma_t = self.regularization_lambda * np.linalg.inv(self.discounted_R + self.B.T @ self.P @ self.B)

    def get_action(self, x, is_entropy_regularized=True):
        """
            Compute the action given the current state.

            INPUTS:
                x - State of system as a column vector.
                is_entropy_regularized - Boolean indicating whether or not to sample from a (normal) distribution.

            OUTPUTS:
                Action from LQR policy.
        """

        # TODO: Check if this is returning the correct action dimensions
        if is_entropy_regularized:
            return np.random.normal(-self.C @ (x - self.reference_point), self.sigma_t)
        else:
            return -self.C @ (x - self.reference_point)