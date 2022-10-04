from sys_ident.utils import Experiment
from sys_ident.models import BaseModel
import numpy as np
from scipy.integrate import solve_ivp


def simulate_experiment(experiment: Experiment, model: BaseModel, params: np.ndarray):
    x_0 = experiment.x_0
    y_sim = np.empty(len(experiment.t))
    y_sim[0] = model.experiment_equation(x_0)
    for idx_time in range(1, len(experiment.t)):
        tspan = (
            experiment.t[idx_time - 1],
            experiment.t[idx_time],
        )
        sol = solve_ivp(
            model.ode,
            tspan,
            x_0,
            args=(experiment.u[idx_time], params),
            t_eval=tspan,
            dense_output=False,
        )
        x_0 = sol.y[-1]
        y_sim[idx_time] = model.experiment_equation(x_0)

    return y_sim
