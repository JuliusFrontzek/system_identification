from dataclasses import dataclass
import numpy as np
from sys_ident.utils import Experiment
from scipy.optimize import minimize, Bounds
from sys_ident.cost_functions import cost_WLS, cost_MLE
from sys_ident.models import BaseModel
from multiprocessing import Pool


@dataclass
class OptimizationData:
    """
    Minimize the selected cost function by optimizing the parameter set.

    Params:
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

    experiments: list[Experiment]
    model: BaseModel
    cost_function: str
    cov_mat: np.ndarray = None
    p_bounds: np.ndarray = None
    cost_functions = {"WLS": cost_WLS, "MLE": cost_MLE}


def run_single_optimization(
    optimization_data: OptimizationData, p_0: np.ndarray, max_iter: int = 100
) -> np.ndarray:
    """
    Run the optimization.

    Params:
        p_0:        1D-Numpy array containing the n starting values for the parameters to be found,
                    i.e. the starting point of the optimization, or a 2D-Numpy array containing multiple starting
                    points for the optimization.
        maxiter:    Integer representing the maximum number of iterations that the minimization algorithm may run.

    """
    if optimization_data.p_bounds is None:
        bounds = optimization_data.p_bounds
    else:
        bounds = Bounds(
            lb=optimization_data.p_bounds[:, 0], ub=optimization_data.p_bounds[:, 1]
        )

    optimization_result = minimize(
        optimization_data.cost_functions[optimization_data.cost_function],
        p_0,
        args=(
            optimization_data.experiments,
            optimization_data.model,
            optimization_data.cov_mat,
        ),
        method="Nelder-Mead",
        bounds=bounds,
        options={"maxiter": max_iter},
    )
    return optimization_result.x


def run_multiple_optimizations(
    optimization_data: OptimizationData, p_0: np.ndarray, max_iter: int = 100
):
    arguments = [(optimization_data, p_0_, max_iter) for p_0_ in p_0]
    with Pool() as pool:
        results = pool.starmap(
            run_single_optimization,
            arguments,
        )
    return results
