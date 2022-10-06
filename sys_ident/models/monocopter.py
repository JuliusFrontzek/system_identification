import numpy as np
from sys_ident.models.base_model import BaseModel


class Monocopter(BaseModel):
    def __init__(self):
        # Define known values
        self.g = 9.81
        self.m = 0.1
        self.l = 0.17
        self.alpha = 17.0 * np.pi / 180.0
        self.ode_parameter_combination = (
            self.l / 2 * self.m * self.g * np.sin(self.alpha)
        )

    def ode(self, x: np.ndarray, t: float, u: float, params: np.ndarray) -> np.ndarray:
        """
        Model parameters:
            b:  Combines the rotational moment of inertia of the monocopter
                arm combined with the proportionality factor of the motor constant
            c:  Combines our uncertainty on where the resulting gravitational force acts
                on the monocopter arm combined with the rotational moment of inertia of the monocopter
        """

        # Define states
        phi = x[0]
        phi_dot = x[1]

        # Define parameters
        b, c, d = params

        # Compute results
        dphi_dt = phi_dot
        dphi2_dt2 = (
            self.l * b * u**2
            - c * np.cos(phi) * self.ode_parameter_combination
            - d * phi_dot
        )
        return np.array([dphi_dt, dphi2_dt2])

    def measurement_equation(self, x: np.ndarray) -> float:
        # Define states
        phi = x[0]

        # Only the first state - the angle phi - is measured
        return phi


if __name__ == "__main__":
    model = Monocopter()
