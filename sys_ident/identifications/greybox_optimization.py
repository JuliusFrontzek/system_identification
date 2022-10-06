from dataclasses import dataclass
import numpy as np
from sys_ident.utils import Experiment
from scipy.optimize import minimize, Bounds
from sys_ident.cost_functions import cost_WLS, cost_MLE
from sys_ident.models import BaseModel
from multiprocessing import Pool
from tqdm import tqdm


@dataclass
class SingleOptimizationData:
    """
    Data required for single optimization.

    Params:
        experiments:            A list of Experiment instances. These instances individually contain the experiments
                                taken during one experiment run.
        model:                  An instance of a class inheriting from the BaseModel class.
        cost_function:          Must be either 'WLS' for weighted least squares or 'MLE' for maximum likelihood estimation.
                                If using 'WLS' a covariance matrix for the experiments must be provided.
        p_0:                    1D-Numpy array containing the n starting values for the parameters to be found,
                                i.e. the starting point of the optimization.
        max_iter:               Integer representing the maximum number of iterations that the minimization algorithm may run.
        cov_mat:                Optional. Covariance matrix for the experiments. Must be provided if using 'WLS' as cost function.
        p_bounds:               Optional. A nx2 2D-Numpy array containing the lower and upper bound for each parameter.
                                The first column contains the lower bounds, the second column contains the upper bounds.
                                Required for
    """

    experiments: list[Experiment]
    model: BaseModel
    cost_function: str
    p_0: np.ndarray
    max_iter: int
    cov_mat: np.ndarray = None
    p_bounds: np.ndarray = None
    cost_functions = {"WLS": cost_WLS, "MLE": cost_MLE}


@dataclass
class MultipleOptimizationsData:
    """
    Data required for multiple optimizations.
        Params:
        experiments:            A list of Experiment instances. These instances individually contain the experiments
                                taken during one experiment run.
        model:                  An instance of a class inheriting from the BaseModel class.
        cost_function:          Must be either 'WLS' for weighted least squares or 'MLE' for maximum likelihood estimation.
                                If using 'WLS' a covariance matrix for the experiments must be provided.
        p_0:                    2D-Numpy array containing the starting parameter sets for all optimizations to be run.
                                Parameter sets are defined in new rows.
        max_iter:               Integer representing the maximum number of iterations that the minimization algorithm may run.
        cov_mat:                Optional. Covariance matrix for the experiments. Must be provided if using 'WLS' as cost function.
        p_bounds:               Optional. A nx2 2D-Numpy array containing the lower and upper bound for each parameter.
                                The first column contains the lower bounds, the second column contains the upper bounds.
                                Required for
    """

    experiments: list[Experiment]
    model: BaseModel
    cost_function: str
    p_0: np.ndarray
    max_iter: int = 100
    cov_mat: np.ndarray = None
    p_bounds: np.ndarray = None


def run_single_optimization(
    optimization_data: SingleOptimizationData,
) -> np.ndarray:
    """
    Run the optimization.

    Params:
        optimization_data:  SingleOptimizationData
            Data required for single optimization.
    """
    if optimization_data.p_bounds is None:
        bounds = optimization_data.p_bounds
    else:
        bounds = Bounds(
            lb=optimization_data.p_bounds[:, 0], ub=optimization_data.p_bounds[:, 1]
        )

    optimization_result = minimize(
        optimization_data.cost_functions[optimization_data.cost_function],
        optimization_data.p_0,
        args=(
            optimization_data.experiments,
            optimization_data.model,
            optimization_data.cov_mat,
        ),
        method="Nelder-Mead",
        bounds=bounds,
        options={"maxiter": optimization_data.max_iter},
    )
    return optimization_result.x


def run_multiple_optimizations(optimization_data: MultipleOptimizationsData):
    single_optimization_data = [
        SingleOptimizationData(
            optimization_data.experiments,
            optimization_data.model,
            optimization_data.cost_function,
            optimization_data.p_0[i],
            optimization_data.max_iter,
            optimization_data.cov_mat,
            optimization_data.p_bounds,
        )
        for i in range(optimization_data.p_0.shape[0])
    ]

    with Pool() as pool:
        results = list(
            tqdm(
                pool.imap_unordered(run_single_optimization, single_optimization_data),
                total=len(single_optimization_data),
            )
        )

    return results
