import numpy as np
from sys_ident.models.base_model import BaseModel


class MonocopterGruppeG(BaseModel):
    def ode(self, x: np.ndarray, t: float, u: float, params: np.ndarray) -> np.ndarray:
        # Define states
        phi = x[0]
        phi_dot = x[1]

        # Define parameters
        p, k, r = params

        # Compute results
        dphi_dt = phi_dot
        dphi2_dt2 = p * (-np.cos(phi) + k * (np.sqrt(u) - phi_dot * r) ** 2)
        return np.array([dphi_dt, dphi2_dt2])

    def measurement_equation(self, x: np.ndarray) -> float:
        # Define states
        phi = x[0]

        # Only the first state - the angle phi - is measured
        return phi


if __name__ == "__main__":
    model = MonocopterGruppeG()
