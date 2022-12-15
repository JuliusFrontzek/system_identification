from sys_ident.utils import Experiment
from sys_ident.cost_functions import cost_MLE, cost_WLS
from sys_ident.identifications import GreyBoxIdentification, ARMAXIdentification
import numpy as np
from sys_ident.models.base_model import BaseModel
from copy import deepcopy
from typing import Union
from multiprocessing import Pool
from tqdm import tqdm
import warnings

# #suppress warnings_run_single_optimization
warnings.filterwarnings("ignore")


class Bootstrap:
    cost_functions = {"WLS": cost_WLS, "MLE": cost_MLE}

    def __init__(
        self,
        identifier: Union[GreyBoxIdentification, ARMAXIdentification],
        params: np.ndarray,
    ) -> None:
        # Simulate y data with the identified parameters
        self.identification_experiments = deepcopy(
            identifier.identification_experiments
        )
        if isinstance(identifier, GreyBoxIdentification):
            for experiment in self.identification_experiments:
                experiment.signal_handler.y = identifier.model.simulate_experiment(
                    experiment, params
                )
        elif isinstance(identifier, ARMAXIdentification):
            raise NotImplementedError
        else:
            raise TypeError(f"The given identifier is not supported")

        self.identifier = identifier

    def run(
        self,
        cost_function: str,
        identification_iterations: int,
        p: np.ndarray,
        bootstrap_iterations: int,
        noise_std_dev: float,
    ):

        self.bootstrap_resulting_params = []
        cost_function = self.cost_functions[cost_function]

        if isinstance(self.identifier, GreyBoxIdentification):
            single_bootstrap_iteration_data = [
                GreyBoxIdentification(
                    self.identification_experiments,
                    self.identifier.model,
                    self.identifier.cost_function,
                    p,
                    identification_iterations,
                    self.identifier.cov_mat,
                    self.identifier.p_bounds,
                    noise_mean=0.0,
                    noise_std_dev=noise_std_dev,
                )
                for _ in range(bootstrap_iterations)
            ]
        elif isinstance(self.identifier, ARMAXIdentification):
            raise NotImplementedError
        else:
            raise TypeError(f"The given identifier is not supported")

        with Pool() as pool:
            results = list(
                tqdm(
                    pool.imap(
                        GreyBoxIdentification._run_single_bootstrap_iteration,
                        single_bootstrap_iteration_data,
                    ),
                    total=bootstrap_iterations,
                )
            )

        self.bootstrap_resulting_params = np.array(results)

    def evaluate(self):
        """
        Returns:
            means:      1D-Numpy array listing the mean of the parameter vectors that have been found through bootstrapping
            std_devs:   1D-Numpy array listing the standard deviations of the bootstrap parameter vectors.
        """
        assert hasattr(self, "bootstrap_resulting_params")
        means = self.bootstrap_resulting_params.mean(axis=0)
        std_devs = self.bootstrap_resulting_params.std(axis=0)
        return means, std_devs
