import unittest
from unittest import TestCase

from sarnn.metrics import *

# Threshold (epsilon) for assuming numerical equality
EPS = 1e-5

# Use in normal tests
ACC_SNN = np.array([0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.75, 0.75])
ACC_ANN = 0.75
POWER = np.array([100, 0, 50, 50, 0, 60, 70, 80])

# Use in out_of_range tests
OOR_ACC_SNN = np.array([-1.3, 0.7, 0.8, 1e4, -5e5])
OOR_ACC_ANN = -2.5
OOR_POWER = np.array([-100, 0, 50, -50, 0])

# Use in relative tests
REL_ACC_ANN = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.75, 0.75])

# Use in size_mismatch tests
SM_ACC_ANN = np.array([0.7, 0.8])
SM_POWER = np.array([100, 0])


class TestPeakAccuracy(TestCase):
    def test_out_of_range(self):
        self.assertTrue(np.abs(1e4 - peak_accuracy(OOR_ACC_SNN)) < EPS)

    def test_peak_accuracy(self):
        self.assertTrue(np.abs(0.8 - peak_accuracy(ACC_SNN)) < EPS)


class TestPowerAboveCurve(TestCase):
    def test_out_of_range(self):
        expect = (1.2 * 100 - 3.3 * 50 + (1e4 + 2.5) * 50) / OOR_ACC_ANN
        actual = power_above_curve(OOR_ACC_SNN, OOR_ACC_ANN, OOR_POWER)
        self.assertTrue(np.abs(expect - actual) < EPS)

    def test_power_above_curve(self):
        expect = (0.65 * 100 + 0.25 * 50 + 0.15 * 50 - 0.05 * 60) / ACC_ANN
        actual = power_above_curve(ACC_SNN, ACC_ANN, POWER)
        self.assertTrue(np.abs(expect - actual) < EPS)

    def test_relative_power_above_curve(self):
        expect = (0.4 * 100 + 0.1 * 50 + 0.05 * 50 - 0.05 * 60) / ACC_ANN
        actual = power_above_curve(ACC_SNN, REL_ACC_ANN, POWER)
        self.assertTrue(np.abs(expect - actual) < EPS)

    def test_size_mismatch_1(self):
        with self.assertRaises(ValueError):
            power_above_curve(ACC_SNN, SM_ACC_ANN, POWER)

    def test_size_mismatch_2(self):
        with self.assertRaises(ValueError):
            power_above_curve(ACC_SNN, ACC_ANN, SM_POWER)


class TestPowerToAccuracy(TestCase):
    def test_first_above(self):
        expect = 100
        actual = power_to_accuracy(ACC_SNN, 0.05, POWER)
        self.assertEqual(expect, actual)

    def test_none_above(self):
        expect = 100 + 50 + 50 + 60 + 70 + 80
        actual = power_to_accuracy(ACC_SNN, 0.85, POWER)
        self.assertEqual(expect, actual)

    def test_out_of_range(self):
        expect = -100
        actual = power_to_accuracy(OOR_ACC_SNN, -0.5, OOR_POWER)
        self.assertEqual(expect, actual)

    def test_power_to_accuracy(self):
        expect = 100 + 50 + 50
        actual = power_to_accuracy(ACC_SNN, 0.7, POWER)
        self.assertEqual(expect, actual)

    def test_size_mismatch(self):
        with self.assertRaises(ValueError):
            power_to_accuracy(ACC_SNN, 0.7, SM_POWER)


class TestTimeAboveCurve(TestCase):
    def test_out_of_range(self):
        expect = (-1.2 - 3.2 - 3.3 - (1e4 + 2.5) + (5e5 - 2.5)) / OOR_ACC_ANN
        actual = time_above_curve(OOR_ACC_SNN, OOR_ACC_ANN)
        self.assertTrue(np.abs(expect - actual) < EPS)

    def test_relative_time_above_curve(self):
        expect = (0.4 + 0.25 + 0.1 + 0.05 - 0.05) / ACC_ANN
        actual = time_above_curve(ACC_SNN, REL_ACC_ANN)
        self.assertTrue(np.abs(expect - actual) < EPS)

    def test_size_mismatch(self):
        with self.assertRaises(ValueError):
            time_above_curve(ACC_SNN, SM_ACC_ANN)

    def test_time_above_curve(self):
        expect = (0.65 + 0.45 + 0.25 + 0.15 + 0.05 - 0.05) / ACC_ANN
        actual = time_above_curve(ACC_SNN, ACC_ANN)
        self.assertTrue(np.abs(expect - actual) < EPS)


class TestTimeToAccuracy(TestCase):
    def test_first_above(self):
        self.assertEqual(1, time_to_accuracy(ACC_SNN, 0.05))

    def test_none_above(self):
        self.assertEqual(9, time_to_accuracy(ACC_SNN, 0.85))

    def test_out_of_range(self):
        self.assertEqual(2, time_to_accuracy(OOR_ACC_SNN, -0.5))

    def test_time_to_accuracy(self):
        self.assertEqual(5, time_to_accuracy(ACC_SNN, 0.7))


if __name__ == "__main__":
    unittest.main()
