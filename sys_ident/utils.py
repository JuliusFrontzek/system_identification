from dataclasses import dataclass
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import signal
from pyDOE import lhs


def auto_corr(x):
    result = np.correlate(x, x, mode="full")
    return result[result.size // 2 :]


def covariance_fixed_tau(X: np.ndarray, tau: int) -> float:
    N = len(X) - tau
    m_x = X.mean()
    X_offset = X[tau:]

    if tau > 0:
        X = X[:-tau]

    cov = np.sum((X - m_x) * (X_offset - m_x)) / (N - 1.0)
    return cov


def covariance(X: np.ndarray) -> np.ndarray:
    N = len(X)
    cov_fun = np.empty(N)
    for i in range(N):
        cov_fun[i] = covariance_fixed_tau(X, i)
    return cov_fun


def cross_covariance_fixed_tau(X: np.ndarray, Y: np.ndarray, tau: int) -> float:
    N = len(X) - tau
    m_x = X.mean()
    m_y = Y.mean()
    if tau > 0:
        X = X[:-tau]
    Y = Y[tau:]
    cross_cov = np.sum((X - m_x) * (Y - m_y)) / (N - 1.0)
    return cross_cov


def cross_covariance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    N_x = len(X)
    N_y = len(Y)
    assert N_x == N_y, "Dimensions of X and Y must match"
    cross_cov_fun = np.empty(N_x)
    for i in range(N_x):
        cross_cov_fun[i] = cross_covariance_fixed_tau(X, Y, i)
    return cross_cov_fun


class SignalHandler:
    def __init__(
        self,
        *,
        signal_names: dict[str],
        signals_directory: str,
    ):
        for signal_name, signal_file_name in signal_names.items():
            signal_full_file_name = os.path.join(signals_directory, signal_file_name)
            setattr(
                self, signal_name, np.genfromtxt(signal_full_file_name, delimiter=",")
            )

        self.signal_names = list(signal_names.keys())
        assert hasattr(self, "t"), "A time signal must be provided"

    @property
    def sampling_frequency(self):
        return 1.0 / (self.t[1] - self.t[0])

    def down_sample(self, n_th_element: int):
        for signal_name in self.signal_names:
            signal = getattr(self, signal_name)
            signal = signal[::n_th_element]
            setattr(self, signal_name, signal)

    def filter_signals_by_time(self, start: float, end: float):
        start_idx = np.abs(self.t - start).argmin()
        end_idx = np.abs(self.t - end).argmin()

        for signal_name in self.signal_names:
            signal = getattr(self, signal_name)
            signal = signal[start_idx:end_idx]
            setattr(self, signal_name, signal)

    def show_signals(self, specific_signal=None):
        fig, ax = plt.subplots()
        if specific_signal is not None:
            signal_names = [specific_signal]
        else:
            signal_names = self.signal_names.copy()
            signal_names.remove("t")

        for signal_name in signal_names:
            ax.plot(self.t, getattr(self, signal_name), label=signal_name)

        ax.legend()

        return fig

    def filter_signals_butterworth(self, order: int, cut_off_freq_mult_nyquist: float):
        """
        Params:
            order: Order of the butterworth filter
            cut_off_freq_mult_nyquist: Cut off frequency of the filter is set as a multiple of the Nyquist frequency
        """
        assert (
            cut_off_freq_mult_nyquist < 1.0
        ), "Cut off frequency must be smaller than or euqal to the Nyquist frequency"

        sos = signal.butter(
            order,
            cut_off_freq_mult_nyquist * self.sampling_frequency * 0.5,
            fs=self.sampling_frequency,
            analog=False,
            output="sos",
        )

        first_n_data_points_to_remove = 10

        for signal_name in self.signal_names:
            if not signal_name == "t":
                signal_ = getattr(self, signal_name)
                filtered_signal = signal.sosfilt(sos, signal_)
                filtered_signal = filtered_signal[first_n_data_points_to_remove:]
                setattr(self, signal_name, filtered_signal)

        self.t = self.t[first_n_data_points_to_remove:]

    def get_steady_component(self, signal_name: str):
        return np.mean(getattr(self, signal_name))

    def show_periodogram(self, signal_name: str):
        signal_ = getattr(self, signal_name)
        f, Pxx_den = signal.periodogram(signal_, self.sampling_frequency)
        plt.semilogy(f, Pxx_den)


@dataclass
class Experiment:
    t: np.ndarray
    u: np.ndarray
    y: np.ndarray
    x_0: np.ndarray


def generate_initial_params_lhs(num_samples: int, p_bounds: np.ndarray) -> np.ndarray:
    """
    Params:
        num_samples:    An integer that specifies the number of samples that shall be created.
        p_bounds:       A 2D-Numpy array that specifies the parameter bounds. Column 0 contains
                        the lower bounds, column 1 the upper bounds.

        Returns:
            A 2D-Numpy array. Each row represents a sample and contains n columns for the n parameters provided.
    """
    num_params = p_bounds.shape[0]
    return p_bounds[:, 0] + lhs(
        num_params, samples=num_samples, criterion="maximin"
    ) * (p_bounds[:, 1] - p_bounds[:, 0])
