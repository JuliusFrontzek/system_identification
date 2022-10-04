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

    identification_signal_handler.filter_signals_by_time(0, 10)

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

    p_0 = np.array([60.0, 1100.0, 0.0])

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
    optimization = Optimization(p_0, experiments, monocopter, "MLE", maxiter=10)

    optimization_result = optimization.run()[0]
    print(f"Optimal parameters: {optimization_result}")

    fig, ax = plt.subplots()
    y = simulate_experiment(experiments[0], monocopter, optimization_result)
    ax.plot(experiments[0].t, y, label="Simulation")
    ax.plot(experiments[0].t, experiments[0].y, label="Ground truth")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
