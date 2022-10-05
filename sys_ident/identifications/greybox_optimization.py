from dataclasses import dataclass
import numpy as np
from sys_ident.utils import Experiment
from scipy.optimize import minimize, Bounds
from sys_ident.cost_functions import cost_WLS, cost_MLE
from sys_ident.models import BaseModel


@dataclass
class Optimization:
    """
    Minimize the selected cost function by optimizing the parameter set.

    Params:
        p_0:                    Either a 1D-Numpy array containing the n starting values for the parameters to be found,
                                i.e. the starting point of the optimization, or a 2D-Numpy array containing multiple starting
                                points for the optimization.
        experiments:            A list of Experiment instances. These instances individually contain the experiments
                                taken during one experiment run.
        model:                  An instance of a class inheriting from the BaseModel class.
        cost_function:          Must be either 'WLS' for weighted least squares or 'MLE' for maximum likelihood estimation.
                                If using 'WLS' a covariance matrix for the experiments must be provided.
        cov_mat:                Optional. Covariance matrix for the experiments. Must be provided if using 'WLS' as cost function.
        p_bounds:               Optional. A nx2 2D-Numpy array containing the lower and upper bound for each parameter.
                                The first column contains the lower bounds, the second column contains the upper bounds.
                                Required for
    """

    p_0: np.ndarray
    experiments: list[Experiment]
    model: BaseModel
    cost_function: str
    cov_mat: np.ndarray = None
    p_bounds: np.ndarray = None
    cost_functions = {"WLS": cost_WLS, "MLE": cost_MLE}

    def run(self, max_iter: int = 1000) -> np.ndarray:
        """
        Run the optimization.

        Params:
            maxiter:    Integer representing the maximum number of iterations that the minimization algorithm may run.

        """
        if self.p_bounds is None:
            bounds = self.p_bounds
        else:
            bounds = Bounds(lb=self.p_bounds[:, 0], ub=self.p_bounds[:, 1])

        num_optimizations = 1
        if self.p_0.ndim == 1:
            self.p_0 = np.array([self.p_0])
        else:
            num_optimizations = self.p_0.shape[0]

        resulting_params = np.zeros((num_optimizations, self.p_0.shape[1]))

        for i in range(num_optimizations):
            optimization_result = minimize(
                self.cost_functions[self.cost_function],
                self.p_0[i],
                args=(self.experiments, self.model, self.cov_mat),
                method="Nelder-Mead",
                bounds=bounds,
                options={"maxiter": max_iter},
            )
            resulting_params[i] = optimization_result.x

        return resulting_params
