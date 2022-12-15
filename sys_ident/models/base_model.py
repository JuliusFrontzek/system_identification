from abc import ABC, abstractmethod
import numpy as np
from sys_ident.utils import Experiment
from scipy.signal import lti, dlti
from scipy.integrate import odeint


class BaseModel(ABC):
    """
    Abstract base class for models used to identify systems.
    In addition to some of the subsequently specified methods, a subclass may contain attributes for model dimensions, etc.
    """

    def ode(self, x: np.ndarray, t: float, u: float, params: np.ndarray) -> np.ndarray:
        """
        Ode method which represents the model's ordinary differential equation.

        Params:
            x:          1D-Numpy array representing the model's state.
            t:          Float representing the current time
            u:          Float representing the current input into the model. Currently only SISO models are considered.
            params:     1D-Numpy array representing the model's parameter vector.
        """
        raise NotImplementedError

    def lti(self, params: np.ndarray) -> lti:
        raise NotImplementedError

    def dlti(self, params: np.ndarray) -> dlti:
        raise NotImplementedError

    def measurement_equation(self, x: np.ndarray) -> float:
        """
        Measurement equation method. Returns the part of the state vector which is being measured.

        Params:
            x:  A 1D-Numpy array representing the state of the model.

        Returns:
            Float that represents the measured value. Currently only SISO models are considered.
        """
        raise NotImplementedError

    def simulate_experiment(
        self, experiment: Experiment, params: np.ndarray
    ) -> np.ndarray:
        try:
            return self.lti(params).output(
                experiment.signal_handler.u, experiment.signal_handler.t, experiment.x_0
            )[1]
        except NotImplementedError:
            pass

        try:
            return self.dlti(params).output(
                experiment.signal_handler.u, experiment.signal_handler.t, experiment.x_0
            )[1]
        except NotImplementedError:
            pass

        try:
            x_0 = experiment.x_0
            y_sim = np.empty(len(experiment.signal_handler.t))
            y_sim[0] = self.measurement_equation(x_0)
            for idx_time in range(1, len(experiment.signal_handler.t)):
                tspan = (
                    experiment.signal_handler.t[idx_time - 1],
                    experiment.signal_handler.t[idx_time],
                )
                sol = odeint(
                    self.ode,
                    x_0,
                    tspan,
                    args=(experiment.signal_handler.u[idx_time], params),
                )
                x_0 = sol[1]
                y_sim[idx_time] = self.measurement_equation(x_0)

            return y_sim
        except NotImplementedError:
            raise NotImplementedError(
                "Could not simulate the model's response to the given input since none of the methods 'ode', 'lti' or 'dlti' is implemented."
            )
