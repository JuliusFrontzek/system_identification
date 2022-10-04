from abc import ABC, abstractmethod
import numpy as np
from sys_ident.utils import Experiment


class BaseModel(ABC):
    """
    Abstract base class for models used to identify systems.
    In addition to the subsequently specified, obligatory methods, this class may contain attributes for model dimensions, etc.
    """

    @abstractmethod
    def ode(self, t: float, x: np.ndarray, u: float, params: np.ndarray) -> np.ndarray:
        """
        A model must implement an ode method which represents the model's ordinary differential equation.

        Params:
            t:      Float representing the current time
            x:      1D-Numpy array representing the model's state.
            u:      Float representing the current input into the model. Currently only SISO models are considered.
            params:      1D-Numpy array representing the model's parameter vector.
        """
        pass

    @abstractmethod
    def experiment_equation(self, x: np.ndarray) -> float:
        """
        A model must implement a measurement equation method.

        Params:


        Returns:
            Float that represents the measured value. Currently only SISO models are considered.
        """
        pass

    # @abstractmethod
    # def simulate(
    #     self, measurements: list[Experiment], params: np.ndarray
    # ) -> list[np.ndarray]:
    #     """
    #     A model must implement a simulate method. This extracts the initial state of the model from each of
    #     the experiments in the 'measurements' parameter and simulates the system's responses to each of the
    #     set of inputs which are also being extracted from the 'measurements'.
    #     """
    #     pass
