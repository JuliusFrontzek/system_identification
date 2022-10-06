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


def plot_covariance(t: np.ndarray, X: np.ndarray) -> None:
    cov_fun = covariance(X)
    fig, ax = plt.subplots()
    ax.plot(t, cov_fun)
    ax.set_xlabel("t")
    ax.set_ylabel("Covariance")
    plt.show()


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


def plot_cross_covariance(t: np.ndarray, X: np.ndarray, Y: np.ndarray) -> None:
    cross_cov_fun = cross_covariance(X, Y)
    fig, ax = plt.subplots()
    ax.plot(t, cross_cov_fun)
    ax.set_xlabel("t")
    ax.set_ylabel("Covariance")
    plt.show()


def read_csv_as_array(
    directory: str, file_name: str, delimiter: str = ","
) -> np.ndarray:
    """
    Reads a csv file that contains a time series and returns a numpy array of the time series.
    """
    if file_name == "":
        return
    full_file_name = os.path.join(directory, file_name)
    return np.genfromtxt(full_file_name, delimiter=delimiter)


class SignalHandler:
    signal_names = ["u", "y", "w"]

    def __init__(
        self,
        *,
        u_y_w_t_file_names: dict[str] = None,
        signals_directory: str = None,
        u_y_w_t: list[np.ndarray] = None,
    ):
        if isinstance(u_y_w_t, list):
            self.u = u_y_w_t[0]
            self.y = u_y_w_t[1]
            self.w = u_y_w_t[2]
            self.t = u_y_w_t[3]
        else:
            self.u = read_csv_as_array(signals_directory, u_y_w_t_file_names["u"])
            self.y = read_csv_as_array(signals_directory, u_y_w_t_file_names["y"])
            self.w = read_csv_as_array(signals_directory, u_y_w_t_file_names["w"])
            self.t = read_csv_as_array(signals_directory, u_y_w_t_file_names["t"])

    @property
    def sampling_step(self) -> float:
        return self.t[1] - self.t[0]

    @property
    def sampling_frequency(self) -> float:
        return 1.0 / self.sampling_step

    @property
    def y_0(self) -> float:
        return self.y[0]

    @property
    def y_dot_0(self) -> float:
        return (self.y[1] - self.y[0]) / (self.t[1] - self.t[0])

    def down_sample(self, n_th_element: int):
        for signal_name in self.signal_names:
            signal = getattr(self, signal_name)
            if signal is None:
                continue
            signal = signal[::n_th_element]
            setattr(self, signal_name, signal)
        self.t = self.t[::n_th_element]

    def filter_signals_by_time(self, start: float, end: float):
        start_idx = np.abs(self.t - start).argmin()
        end_idx = np.abs(self.t - end).argmin()

        for signal_name in self.signal_names:
            signal = getattr(self, signal_name)
            if signal is None:
                continue
            signal = signal[start_idx:end_idx]
            setattr(self, signal_name, signal)

        self.t = self.t[start_idx:end_idx]

    def show_signals(self, signal_names: list[str] = None):
        fig, ax = plt.subplots()
        if signal_names is None:
            signal_names = self.signal_names

        for signal_name in signal_names:
            ax.plot(self.t, getattr(self, signal_name), label=signal_name)

        ax.legend()
        plt.show()

    def move_signal(self, signal_name: str, delta_time: float, delay: bool):
        """
        Move a signal relative to the others by a given time delta.

        Params:
            signal_name: str
                Name of the signal
            delta_time: float
                Time delta by which the signal will be moved approximately.
                The maximum error that is introduced here is half the sampling frequency.
            delay: bool
                If True, the signal will be delayed, i.e. moved to the right on the time axis.
                If False, the signal will be advanced, i.e. moved to the left on the time axis.
        """

        num_indices_to_move = round(delta_time / self.sampling_step)
        if delay:
            for signal_name_ in self.signal_names:
                signal_ = getattr(self, signal_name_)
                if signal_ is None:
                    continue
                if signal_name_ == signal_name:
                    signal_ = signal_[:-num_indices_to_move]
                else:
                    signal_ = signal_[num_indices_to_move:]
                setattr(self, signal_name_, signal_)

            self.t = self.t[num_indices_to_move:]
        else:
            for signal_name_ in self.signal_names:
                signal_ = getattr(self, signal_name_)
                if signal_ is None:
                    continue
                if signal_name_ == signal_name:
                    signal_ = signal_[num_indices_to_move:]
                else:
                    signal_ = signal_[:-num_indices_to_move]
                setattr(self, signal_name_, signal_)
            self.t = self.t[:-num_indices_to_move]

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
            signal_ = getattr(self, signal_name)
            if signal_ is None:
                continue
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

    def add_noise(
        self, mean: float, std_dev: float, in_place: bool = False
    ) -> np.ndarray:
        """
        Adds noise to signal y.

        Params:
            mean:       Float that represents the mean of the noise.
            std_dev:    Non-negative float that defines the noise's standard deviation.
            in_place:   Boolean that specifies whether or not to change the experiment's y-value
                        in place or not. Default is False.

        Returns:
            A 1D-Numpy array representing the noisy signal.
        """
        noise = np.random.normal(mean, std_dev, self.y.shape[0])
        new_y = self.y + noise
        if in_place:
            self.y = new_y
        return new_y


@dataclass
class Experiment:
    signal_handler: SignalHandler
    x_0: np.ndarray

    def __str__(self):
        return f"Experiment with {self.signal_handler.t.shape[0]} datapoints."

    def __repr__(self):
        return f"Experiment with object id: {id(self)}."


def generate_initial_params_lhs(num_samples: int, p_bounds: np.ndarray) -> np.ndarray:
    """
    Params:
        num_samples:    An integer that specifies the number of samples that shall be created.
        p_bounds:       A 2D-Numpy array that specifies the parameter bounds. Column 0 contains
                        the lower bounds, column 1 the upper bounds.

        Returns:
            A nested list of parameter 'vectors'. Each row represents a sample and contains n columns for the n parameters provided.
    """
    num_params = p_bounds.shape[0]
    params = p_bounds[:, 0] + lhs(
        num_params, samples=num_samples, criterion="maximin"
    ) * (p_bounds[:, 1] - p_bounds[:, 0])
    return params.tolist()
