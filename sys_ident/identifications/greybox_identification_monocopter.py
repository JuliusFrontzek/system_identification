from sys_ident.utils import (
    SignalHandler,
    Experiment,
)
import matplotlib.pyplot as plt
import numpy as np
from sys_ident.identifications import GreyBoxIdentification
from sys_ident.models import Monocopter
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
        u_y_w_t_file_names={
            "u": "u.csv",
            "w": "",
            "y": "y.csv",
            "t": "t.csv",
        },
        signals_directory="/home/julius/Projects/system_identification/data",
    )

    validation_signal_handler = SignalHandler(
        u_y_w_t_file_names={
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
    identification_signal_handler.move_signal("y", 0.6, False)

    phi_0 = identification_signal_handler.y_0
    phi_dot_0 = identification_signal_handler.y_dot_0

    x_0 = np.array([phi_0, phi_dot_0])

    p_0 = [60.0, 1100.0, 0.0]

    # plot_cross_covariance(
    #     identification_signal_handler.t,
    #     identification_signal_handler.u,
    #     identification_signal_handler.y,
    # )
    # plt.plot(covariance(identification_signal_handler.y))
    # identification_signal_handler.show_signals(["u", "y"])

    # plt.plot(
    #     cross_covariance(
    #         identification_signal_handler.u, identification_signal_handler.y
    #     )
    # )
    # identification_signal_handler.show_periodogram("y")
    # plt.show()

    experiments = [
        Experiment(
            identification_signal_handler,
            x_0,
        )
    ]

    monocopter = Monocopter()
    p_0 = generate_initial_params_lhs(
        num_samples=10,
        p_bounds=np.array([[1.0, 1000.0], [100.0, 10000.0], [0.0, 100.0]]),
    )
    optimization_data = MultipleOptimizationsData(
        experiments, monocopter, "MLE", p_0, max_iter=10
    )
    optimization_result = run_multiple_optimizations(optimization_data)

    fig, ax = plt.subplots()
    ax.plot(
        experiments[0].signal_handler.t,
        experiments[0].signal_handler.y,
        label="Ground truth",
    )
    ax.plot(
        experiments[0].signal_handler.t, experiments[0].signal_handler.u, label="Input"
    )
    for idx, result in enumerate(optimization_result):
        y = monocopter.simulate_experiment(experiments[0], result)
        ax.plot(experiments[0].signal_handler.t, y, label=f"Simulation {idx+1}")

    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
