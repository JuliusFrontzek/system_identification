from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sys_ident.utils import Experiment
from scipy.optimize import minimize, Bounds
from sys_ident.cost_functions import cost_WLS, cost_MLE
from sys_ident.models import BaseModel
from multiprocessing import Pool
from tqdm import tqdm
import warnings

# #suppress warnings_run_single_optimization
warnings.filterwarnings("ignore")


@dataclass
class GreyBoxIdentification:
    """
    Aims to identify a system by using a grey box approach.

    Params:
        identification_experiments: A list of Experiment instances. These instances individually contain the experiments
                                    taken during one experiment run that will be used for identification.
        model:                      An instance of a class inheriting from the BaseModel class.
        cost_function:              Must be either 'WLS' for weighted least squa2D-Numpy array containing the starting parameter sets for all optimizations to be run.
                                    Parameter sets are defined in new rows.res or 'MLE' for maximum likelihood estimation.
                                    If using 'WLS' a covariance matrix for the experiments must be provided.
        p_0:                        1D- or 2D-Numpy array containing either the n starting values for the parameters to be found,
                                    i.e. the starting point of the optimization or the starting parameter sets for all optimizations to be run.
                                    Parameter sets are defined in new rows.
        max_iter:                   Integer representing the maximum number of iterations that the minimization algorithm may run.
        cov_mat:                    Optional. Covariance matrix for the experiments. Must be provided if using 'WLS' as cost function.
        p_bounds:                   Optional. A nx2 2D-Numpy array containing the lower and upper bound for each parameter.
                                    The first column contains the lower bounds, the second column contains the upper bounds.
        noise_mean:                 Optional. Float that specifies the mean of the gaussian noise added for bootstrapping.
        noise_std_dev:              Optional. Float that specifies the standard deviation of the gaussian noise added for bootstrapping.
    """

    identification_experiments: list[Experiment]
    model: BaseModel
    cost_function: str
    p_0: np.ndarray
    max_iter: int
    cov_mat: np.ndarray = None
    p_bounds: np.ndarray = None
    noise_mean: float = 0.0
    noise_std_dev: float = 0.0
    cost_functions = {"WLS": cost_WLS, "MLE": cost_MLE}

    def run(self, **kwargs) -> np.ndarray:
        if self.p_0.ndim == 1:
            return self._run_single_identification(self, **kwargs)
        elif self.p_0.ndim == 2:
            return self._run_multiple_identifications(**kwargs)
        else:
            raise ValueError("self.p_0 has an invalid amount of dimensions")

    @staticmethod
    def _run_single_identification(
        optimization_data: GreyBoxIdentification,
    ) -> np.ndarray:
        """
        Runs a single identification.

        Returns the array of optimal parameters it found.
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
                optimization_data.identification_experiments,
                optimization_data.model,
                optimization_data.cov_mat,
            ),
            method="Nelder-Mead",
            bounds=bounds,
            options={"maxiter": optimization_data.max_iter},
        )
        return optimization_result.x

    @staticmethod
    def _run_single_bootstrap_iteration(
        optimization_data: GreyBoxIdentification,
    ) -> np.ndarray:
        """
        Runs a single bootstrap iteration.

        Returns the array of optimal parameters it found.
        """

        # Add some noise to the simulated experiment(s)
        for experiment in optimization_data.identification_experiments:
            experiment.signal_handler.y = experiment.signal_handler.add_gaussian_noise(
                optimization_data.noise_mean,
                optimization_data.noise_std_dev,
            )

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
                optimization_data.identification_experiments,
                optimization_data.model,
                optimization_data.cov_mat,
            ),
            method="Nelder-Mead",
            bounds=bounds,
            options={"maxiter": optimization_data.max_iter},
        )
        return optimization_result.x

    def _run_multiple_identifications(self):
        """Runs multiple identifications in parallel (using multiprocessing)."""
        single_optimization_data = [
            GreyBoxIdentification(
                self.identification_experiments,
                self.model,
                self.cost_function,
                p_0_,
                self.max_iter,
                self.cov_mat,
                self.p_bounds,
            )
            for p_0_ in self.p_0
        ]

        with Pool() as pool:
            results = list(
                tqdm(
                    pool.imap(
                        self._run_single_identification, single_optimization_data
                    ),
                    total=len(single_optimization_data),
                )
            )

        return self._find_optimal_params(np.array(results))

    def _find_optimal_params(self, param_sets: np.ndarray, *args) -> np.ndarray:
        """
        Given a set of identified parameters, this method finds the best one (subject to the respective cost function).
        """
        costs = np.zeros(param_sets.shape[0])
        for idx, params in enumerate(param_sets):
            costs[idx] = self.cost_functions[self.cost_function](
                params, self.identification_experiments, self.model, *args
            )
        idx_min = np.argmin(costs)
        return param_sets[idx_min]
