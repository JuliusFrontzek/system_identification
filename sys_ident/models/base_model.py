from abc import ABC, abstractmethod
import numpy as np
from sys_ident.utils import Experiment


class BaseModel(ABC):
    """
    Abstract base class for models used to identify systems.
    In addition to the subsequently specified, obligatory methods, this class may contain attributes for model dimensions, etc.
    """

    @abstractmethod
    def ode(self, x: np.ndarray, t: float, u: float, params: np.ndarray) -> np.ndarray:
        """
        A model must implement an ode method which represents the model's ordinary differential equation.

        Params:
            x:      1D-Numpy array representing the model's state.
            t:      Float representing the current time
            u:      Float representing the current input into the model. Currently only SISO models are considered.
            params:      1D-Numpy array representing the model's parameter vector.
        """
        pass

    @abstractmethod
    def experiment_equation(self, x: np.ndarray) -> float:
        """
        A model must implement a measurement equation method.

        Params:
            x:  A 1D-Numpy array representing the state of the model.

        Returns:
            Float that represents the measured value. Currently only SISO models are considered.
        """
        pass
