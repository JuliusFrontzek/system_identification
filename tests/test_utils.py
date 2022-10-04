import unittest
import scipy.io
from src.utils import auto_corr, covariance, cross_covariance
from itertools import combinations


class TestCorrCov(unittest.TestCase):
    def setUp(self) -> None:
        signals = scipy.io.loadmat("./tests/signale.mat")
        self.signals = [signals[f"sig{i}"][0, :] for i in range(1, 6)]

    def test_auto_corr(self):
        for signal in self.signals:
            auto_corr(signal)

    def test_covariance(self):
        for signal in self.signals:
            covariance(signal)

    def test_cross_covariance(self):
        for signal_combination in combinations(self.signals, 2):
            max_len = min(len(signal_combination[0]), len(signal_combination[1]))
            signals = [signal[:max_len] for signal in signal_combination]
            cross_covariance(signals[0], signals[1])


if __name__ == "__main__":
    unittest.main()
