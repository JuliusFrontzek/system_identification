from sys_ident.utils import SignalHandler, Experiment
import matplotlib.pyplot as plt
import numpy as np
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function._basis_function import Polynomial
from dataclasses import dataclass


@dataclass
class ARMAXIdentification:
    max_u_leg_steps: int
    max_y_leg_steps: int
    identification_experiments: list[Experiment]
    # validation_experiments: list[Experiment]
    max_num_parameters: int = 5
    max_polynomial_degree: int = 1

    def run(self):
        u_ident = np.array(
            [
                experiment.signal_handler.u
                for experiment in self.identification_experiments
            ]
        ).T
        y_ident = np.array(
            [
                experiment.signal_handler.y
                for experiment in self.identification_experiments
            ]
        ).T
        # u_val = np.array(
        #     [experiment.signal_handler.u for experiment in self.validation_experiments]
        # ).T
        # y_val = np.array(
        #     [experiment.signal_handler.y for experiment in self.validation_experiments]
        # ).T

        basis_function = Polynomial(degree=self.max_polynomial_degree)

        for _ in range(1, self.max_num_parameters):
            model = FROLS(
                order_selection=True,
                n_info_values=7,
                extended_least_squares=False,
                ylag=self.max_y_leg_steps,
                xlag=self.max_u_leg_steps,
                info_criteria="aic",
                estimator="least_squares",
                basis_function=basis_function,
            )

            model.fit(X=u_ident, y=y_ident)


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
            "w": "w.csv",
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

    identification_signal_handler.filter_signals_by_time(0, 100)

    identification_signal_handler.u *= 5.0 / 100.0
    identification_signal_handler.y = (
        (identification_signal_handler.y / 11.33 - 15.0) * np.pi / 180.0
    )

    identification_signal_handler.filter_signals_butterworth(6, 0.5)
    identification_signal_handler.down_sample(10)
    validation_signal_handler.down_sample(10)


if __name__ == "__main__":
    main()
