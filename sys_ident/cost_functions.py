import numpy as np
from .utils import Experiment
from .models import BaseModel


def cost_MLE(
    params: np.ndarray, experiments: list[Experiment], model: BaseModel, *args
):
    """
    Cost function Maximum likelihood estimation
    """

    C = 0
    N = 0
    for experiment in experiments:
        # Simulation
        y_sim = model.simulate_experiment(experiment, params)

        # Difference between simulation and experiment
        diff = experiment.signal_handler.y - y_sim

        # Estimation of covariance matrix
        # Sum across the individual experiments
        C = C + np.reshape(diff, (1, diff.shape[0])) @ np.reshape(
            diff, (diff.shape[0], 1)
        )
        N = (
            N + diff.shape[0]
        )  # Number of experiment time points (Number of experiments in diff)

    C = C / (N - 3)  # Divide by N - number of degrees of freedom that have been removed

    # cost function
    # log(det(C)) is negative, the determinant of C, det(C), shall be minimized however
    I = np.log(np.linalg.det(C))
    return I


def cost_WLS(
    params: np.ndarray, experiments: list[Experiment], model: BaseModel, C: np.ndarray
):
    invC = np.linalg.inv(C)
    I = 0
    for experiment in experiments:
        # Simulation
        y_sim = model.simulate_experiment(experiment, params)

        # Difference between simulation and experiment
        diff = experiment.signal_handler.y - y_sim

        # Cost function
        I = (
            np.reshape(diff, (1, diff.shape[0]))
            @ invC
            @ np.reshape(diff, (diff.shape[0], 1))
        )

    return I
