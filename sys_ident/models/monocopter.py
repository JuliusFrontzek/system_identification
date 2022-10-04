import numpy as np
from sys_ident.models.base_model import BaseModel


class Monocopter(BaseModel):
    def __init__(self):
        # Define known values
        self.g = 9.81
        self.m = 0.1
        self.l = 0.17
        self.alpha = 17.0 * np.pi / 180.0

    def ode(self, t: float, x: np.ndarray, u: float, params: np.ndarray) -> np.ndarray:
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
        b = params[0]
        c = params[1]
        d = params[2]

        # Compute results
        dphi_dt = phi_dot
        dphi2_dt2 = (
            b * u**2 * self.l
            - self.l / 2 * c * self.m * self.g * np.cos(phi) * np.sin(self.alpha)
            - d * phi_dot
        )
        return np.array([dphi_dt, dphi2_dt2])

    def experiment_equation(self, x: np.ndarray) -> float:
        # Define states
        phi = x[0]

        # Only the first state - the angle phi - is measured
        return phi

    # def simulate(self, experiments: list[Experiment], params: np.ndarray):
    #     y_sim = []
    #     for experiment in experiments:
    #         # Simulation
    #         y_sim.append(simulate_experiment(experiment, self, params))

    #     return np.array(y_sim)


if __name__ == "__main__":
    model = Monocopter()
