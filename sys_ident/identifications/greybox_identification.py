from sys_ident.utils import (
    SignalHandler,
    auto_corr,
    covariance,
    cross_covariance,
    Experiment,
)
import matplotlib.pyplot as plt
import numpy as np

from sys_ident.identifications import Optimization
from sys_ident.models import Monocopter
from sys_ident.simulations import simulate_experiment
from sys_ident.utils import generate_initial_params_lhs


def main():
    """
    Nomenclature:
        u refers to the input signal into the control loop
        w refers to the reference signal the we want the system to follow
        y refers to the system's output signal

        Suffixes:
            No suffix: Identification/training data
            Suffix v: Refers to validation data
    """

    identification_signal_handler = SignalHandler(
        signal_names={
            "u": "u.csv",
            "w": "w.csv",
            "y": "y.csv",
            "t": "t.csv",
        },
        signals_directory="/home/julius/Projects/system_identification/data",
    )

    validation_signal_handler = SignalHandler(
        signal_names={
            "u": "u_v.csv",
            "w": "w_v.csv",
            "y": "y_v.csv",
            "t": "t_v.csv",
        },
        signals_directory="/home/julius/Projects/system_identification/data",
    )

    identification_signal_handler.filter_signals_by_time(100.0, 110.0)

    identification_signal_handler.u *= 5.0 / 100.0
    identification_signal_handler.y = (
        (identification_signal_handler.y / 11.33 - 15.0) * np.pi / 180.0
    )

    identification_signal_handler.filter_signals_butterworth(6, 0.5)
    identification_signal_handler.down_sample(10)
    validation_signal_handler.down_sample(10)

    phi_0 = identification_signal_handler.y[0]
    phi_dot_0 = (
        identification_signal_handler.y[1] - identification_signal_handler.y[0]
    ) / (identification_signal_handler.t[1] - identification_signal_handler.t[0])

    x_0 = np.array([phi_0, phi_dot_0])

    # p_0 = np.array([60.0, 1100.0, 0.0])

    # plt.plot(covariance(identification_signal_handler.y))
    # plt.plot(
    #     cross_covariance(
    #         identification_signal_handler.u, identification_signal_handler.y
    #     )
    # )
    # identification_signal_handler.show_periodogram("y")
    # plt.show()

    experiments = [
        Experiment(
            identification_signal_handler.t,
            identification_signal_handler.u,
            identification_signal_handler.y,
            x_0,
        )
    ]

    monocopter = Monocopter()
    p_0 = generate_initial_params_lhs(
        num_samples=10, p_bounds=np.array([[5.0, 100.0], [1050.0, 1500.0], [0.0, 10.0]])
    )
    optimization = Optimization(p_0, experiments, monocopter, "MLE")

    optimization_result = optimization.run(max_iter=200)
    print(f"Optimal parameters: {optimization_result}")
    # optimization_result = np.array([60.0, 1100.0, 0.0])

    fig, ax = plt.subplots()
    ax.plot(experiments[0].t, experiments[0].y, label="Ground truth")
    ax.plot(experiments[0].t, experiments[0].u, label="Input")
    for idx, result in enumerate(optimization_result):
        y = simulate_experiment(experiments[0], monocopter, result)
        ax.plot(experiments[0].t, y, label=f"Simulation {idx+1}")

    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
