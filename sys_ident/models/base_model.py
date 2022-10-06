from abc import ABC, abstractmethod
import numpy as np
from sys_ident.utils import Experiment
from scipy.signal import lti, dlti
from scipy.integrate import odeint


class BaseModel(ABC):
    """
    Abstract base class for models used to identify systems.
    In addition to the subsequently specified, obligatory methods, this class may contain attributes for model dimensions, etc.
    """

    def ode(self, x: np.ndarray, t: float, u: float, params: list) -> np.ndarray:
        """
        A model must implement an ode method which represents the model's ordinary differential equation.

        Params:
            x:      1D-Numpy array representing the model's state.
            t:      Float representing the current time
            u:      Float representing the current input into the model. Currently only SISO models are considered.
            params:      1D-Numpy array representing the model's parameter vector.
        """
        raise NotImplementedError

    def lti(self, params: list) -> lti:
        raise NotImplementedError

    def dlti(self, params: list) -> dlti:
        try:
            return self.lti(params).to_discrete()
        except NotImplementedError:
            raise

    @abstractmethod
    def measurement_equation(self, x: np.ndarray) -> float:
        """
        A model must implement a measurement equation method.

        Params:
            x:  A 1D-Numpy array representing the state of the model.

        Returns:
            Float that represents the measured value. Currently only SISO models are considered.
        """
        pass

    def simulate_experiment(self, experiment: Experiment, params: list):
        try:
            return self.lti(params).output(
                experiment.signal_handler.u, experiment.signal_handler.t, experiment.x_0
            )
        except NotImplementedError:
            pass

        try:
            return self.dlti(params).output(
                experiment.signal_handler.u, experiment.signal_handler.t, experiment.x_0
            )
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
