from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sys_ident.utils import Experiment
from sys_ident.models import BaseModel
import matplotlib.pyplot as plt


@dataclass
class Validation:
    validation_experiments: list[Experiment]
    model: BaseModel

    def __post_init__(self):
        self.criterion_map = {
            "fit": self._estimate_fit,
            "AIC": self._aic,
            "AICc": self._aic_c,
            "AIC_norm": self._aic_norm,
            "BIC": self._bic,
        }

    def run(
        self, val_criterions: list[str], theta: np.ndarray, plot_experiment: bool = True
    ) -> dict[str, float]:
        y_sim = [
            self.model.simulate_experiment(experiment, theta)
            for experiment in self.validation_experiments
        ]

        results = {}

        for crit in val_criterions:
            results[crit] = self.criterion_map[crit](theta, y_sim)

        if plot_experiment:
            self._plot_experiment(theta, y_sim)

        return results

    def _estimate_fit(self, theta: np.ndarray, y_sim: list[np.ndarray]) -> float:
        rel_error = 0.0

        for y_sim, experiment in zip(y_sim, self.validation_experiments):
            amplitude = np.max(experiment.signal_handler.y) - np.min(
                experiment.signal_handler.y
            )

            rel_error += np.abs(y_sim - experiment.signal_handler.y).mean() / amplitude

        return 1.0 - rel_error

    def _aic(self, theta: np.ndarray, y_sim: list[np.ndarray]) -> float:
        """
        Raw Akaike criterion.

        Reference: https://de.mathworks.com/help/ident/ref/idmodel.aic.html#buy66l9-2
        """
        aic = 0.0

        for y_sim, experiment in zip(y_sim, self.validation_experiments):
            n = len(experiment.signal_handler.y)
            aic += (
                n * np.log(np.linalg.norm(y_sim - experiment.signal_handler.y) / n)
                + 2.0 * len(theta)
                + n * (np.log(2 * np.pi) + 1.0)
            )

        return aic

    def _aic_c(self, theta: np.ndarray, y_sim: list[np.ndarray]) -> float:
        """
        Small sample-size corrected Akaike criterion.

        Reference: https://de.mathworks.com/help/ident/ref/idmodel.aic.html#buy66l9-2
        """
        aic_c = 0.0
        for experiment in self.validation_experiments:
            aic_c += (
                self._aic(theta, y_sim)
                + 2.0 * len(theta)
                + (len(theta) + 1.0)
                / (len(experiment.signal_handler.y) - len(theta) - 1.0)
            )

        return aic_c

    def _aic_norm(self, theta: np.ndarray, y_sim: list[np.ndarray]) -> float:
        """
        Normalized Akaike criterion.

        Reference: https://de.mathworks.com/help/ident/ref/idmodel.aic.html#buy66l9-2
        """
        aic_norm = 0.0

        for y_sim, experiment in zip(y_sim, self.validation_experiments):
            n = len(experiment.signal_handler.y)
            aic_norm += (
                np.log(np.linalg.norm(y_sim - experiment.signal_handler.y) / n)
                + 2.0 * len(theta) / n
            )

        return aic_norm

    def _bic(self, theta: np.ndarray, y_sim: list[np.ndarray]) -> float:
        """
        Bayesian information criterion.

        Reference: https://de.mathworks.com/help/ident/ref/idmodel.aic.html#buy66l9-2
        """
        bic = 0.0

        for y_sim, experiment in zip(y_sim, self.validation_experiments):
            n = len(experiment.signal_handler.y)
            bic += (
                n * np.log(np.linalg.norm(y_sim - experiment.signal_handler.y) / n)
                + n * (np.log(2 * np.pi) + 1.0)
                + len(theta) * np.log(n)
            )

        return bic

    def _plot_experiment(self, theta: np.ndarray, y_sim: list[np.ndarray]):
        fig, ax = plt.subplots(len(self.validation_experiments))
        side_length = 8
        fig.set_size_inches(side_length, side_length * len(self.validation_experiments))
        if not isinstance(ax, np.ndarray):
            ax = np.array([ax])

        for ax_, y_sim, experiment in zip(ax, y_sim, self.validation_experiments):
            ax_.plot(
                experiment.signal_handler.t,
                experiment.signal_handler.y,
                label="Ground truth",
            )
            ax_.plot(
                experiment.signal_handler.t,
                y_sim,
                label=f"Validation",
            )

            ax_.legend()
        plt.show()
