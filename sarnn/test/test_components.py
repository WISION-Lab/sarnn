import unittest
from unittest import TestCase

import numpy as np
from scipy.stats import norm

from sarnn.components import *

# Threshold (epsilon) for assuming numerical equality
EPS = 1e-5

# Number of random repeats for each test
N_REPEATS = 100


class TestBatchNormSparsityRegularizer(TestCase):
    def test_call(self):
        reg = BatchNormSparsityRegularizer(
            0.1, layer_neurons=3, model_neurons=12)
        for x in np.linspace(-3.0, 3.0, 61):
            self.assertTrue(np.abs(reg(x) - 0.1 * 3 / 12 * norm.cdf(x)) < EPS)

    def test_negative_penalty(self):
        reg = BatchNormSparsityRegularizer(
            -0.5, layer_neurons=3, model_neurons=12)
        for x in np.linspace(-3.0, 3.0, 61):
            self.assertTrue(np.abs(reg(x) + 0.5 * 3 / 12 * norm.cdf(x)) < EPS)

    def test_neurons_none(self):
        reg = BatchNormSparsityRegularizer(0.1)
        with self.assertRaises(TypeError):
            reg(1.0)

    def test_zero_penalty(self):
        reg = BatchNormSparsityRegularizer(
            0.0, layer_neurons=3, model_neurons=12)
        for x in np.linspace(-3.0, 3.0, 61):
            self.assertEqual(0.0, reg(x))


class TestL1WeightRegularizer(TestCase):
    def test_call(self):
        reg = L1WeightRegularizer(0.1, layer_synapses=3, model_synapses=12)
        for x in np.logspace(-3.0, 3.0, 61):
            self.assertTrue(np.abs(reg(x) - 0.1 * 3 / 12 * np.abs(x)) < EPS)

    def test_l2_decay(self):
        reg = L1WeightRegularizer(
            0.1, l2_decay=0.05, layer_synapses=3, model_synapses=12)
        for x in np.logspace(-3.0, 3.0, 61):
            expect = 0.1 * 3 / 12 * np.abs(x) + 0.05 * x ** 2
            self.assertTrue(np.abs(reg(x) - expect) < EPS)

    def test_negative_penalty(self):
        reg = L1WeightRegularizer(-0.5, layer_synapses=3, model_synapses=12)
        for x in np.logspace(-3.0, 3.0, 61):
            self.assertTrue(np.abs(reg(x) + 0.5 * 3 / 12 * np.abs(x)) < EPS)

    def test_synapses_none(self):
        reg = L1WeightRegularizer(0.1)
        with self.assertRaises(TypeError):
            reg(1.0)

    def test_zero_penalty(self):
        reg = L1WeightRegularizer(0.0, layer_synapses=3, model_synapses=12)
        for x in np.logspace(-3.0, 3.0, 61):
            self.assertEqual(0.0, reg(x))


class TestPostBatchNormOffset(TestCase):
    def test_build(self):
        layer = PostBatchNormOffset()
        self.assertTrue(layer.offset is None)
        layer.build((1, 2, 3, 4))
        self.assertTrue(np.array_equal(np.zeros(4), layer.offset.numpy()))

    def test_call(self):
        layer = PostBatchNormOffset()
        shape = (1, 2, 3, 4)
        layer.build(shape)
        inputs = np.ones(shape)
        self.assertTrue(np.array_equal(np.ones(shape), layer(inputs)))
        layer.offset.assign_add(tf.fill(layer.offset.shape, 0.5))
        self.assertTrue(np.array_equal(np.full(shape, 1.5), layer(inputs)))


# An abstract base class used to mirror the component hierarchy
# noinspection PyPep8Naming, PyUnresolvedReferences
class TestSpikingCell:
    @classmethod
    def setUpClass(cls):
        np.random.seed(0)

    def test_accumulate_only(self):
        cell, input_shape = self._build_cell(accumulate_only=True)
        for _ in range(N_REPEATS):
            inputs, v_i = self._random_init(cell, input_shape)
            o_actual, [v_o_actual] = cell.call(inputs, [v_i])
            v_o_expect = v_i + self._compute_dv(cell, inputs)
            self.assertTrue(np.all(np.abs(v_o_expect - v_o_actual) < EPS))
            self.assertTrue(np.all(np.abs(v_o_expect - o_actual) < EPS))

    def test_call(self):
        cell, input_shape = self._build_cell()
        for _ in range(N_REPEATS):
            inputs, v_i = self._random_init(cell, input_shape)
            s_o_actual, [v_o_actual] = cell.call(inputs, [v_i])
            self._check_standard(cell, inputs, v_i, s_o_actual, v_o_actual)

    def test_clamp(self):
        cell, input_shape = self._build_cell(enable_clamp=True)
        for _ in range(N_REPEATS):
            inputs, v_i = self._random_init(cell, input_shape)
            c_i = np.random.randint(low=-10, high=10)
            s_o_actual, [v_o_actual, c_o_actual] = cell.call(inputs, [
                v_i, np.full(cell.state_size[-1], c_i).astype(np.float32)])
            self.assertTrue(np.all(c_o_actual == c_i - 1))
            if c_i > 0:
                self.assertTrue(np.all(s_o_actual == 0))
                self.assertTrue(np.all(np.abs(v_o_actual - v_i) < EPS))
            else:
                self._check_standard(cell, inputs, v_i, s_o_actual, v_o_actual)

    def test_reset_mechanism_zero(self):
        cell, input_shape = self._build_cell(reset_mechanism="zero")
        for _ in range(N_REPEATS):
            inputs, v_i = self._random_init(cell, input_shape)
            s_o_actual, [v_o_actual] = cell.call(inputs, [v_i])
            v_o_expect = v_i + self._compute_dv(cell, inputs)
            s_o_expect = v_o_expect >= 1.0
            v_o_expect = v_o_expect * tf.cast(~s_o_expect, tf.float32)
            self.assertTrue(np.array_equal(s_o_expect, s_o_actual))
            self.assertTrue(np.all(np.abs(v_o_expect - v_o_actual) < EPS))

    def test_spike_tracking(self):
        cell, input_shape = self._build_cell(track_spikes=True)
        for _ in range(N_REPEATS):
            inputs, v_i = self._random_init(cell, input_shape)
            s_o_actual, [_, n_actual] = cell.call(
                inputs, [v_i, np.zeros(cell.state_size[1])])
            n_expect = self._count_spikes(cell, inputs)
            self.assertEqual(n_expect, n_actual)

    def test_buffer_dv(self):
        cell, input_shape = self._build_cell(buffer_dv=True)
        last_inputs, _ = self._random_init(cell, input_shape)
        for _ in range(N_REPEATS):
            inputs, v_i = self._random_init(cell, input_shape)
            s_o_actual, [v_o_actual, dv_buffer] = cell.call(
                inputs, [v_i, self._compute_dv(cell, last_inputs)])
            self._check_standard(
                cell, last_inputs, v_i, s_o_actual, v_o_actual)
            self.assertTrue(
                np.array_equal(dv_buffer, self._compute_dv(cell, inputs)))
            last_inputs = inputs

    def test_t_refrac(self):
        cell, input_shape = self._build_cell(t_refrac=1)
        for _ in range(N_REPEATS):
            inputs, v_i = self._random_init(cell, input_shape)
            r_i = np.random.randint(
                low=0, high=3, size=cell.state_size[-1]).astype(np.float32)
            s_o_actual, [v_o_actual, r_o_actual] = cell.call(
                inputs, [v_i, r_i])
            self.assertTrue(np.array_equal(
                r_o_actual, (r_i + 1.0) * (np.array(s_o_actual) == 0.0)))
            v_o_expect = v_i + self._compute_dv(cell, inputs) * (r_i >= 1.0)
            s_o_expect = v_o_expect >= 1.0
            v_o_expect -= s_o_expect
            self.assertTrue(np.array_equal(s_o_expect, s_o_actual))
            self.assertTrue(np.all(np.abs(v_o_expect - v_o_actual) < EPS))

    def _check_standard(self, cell, inputs, v_i, s_o_actual, v_o_actual):
        v_o_expect = v_i + self._compute_dv(cell, inputs)
        s_o_expect = v_o_expect >= 1.0
        v_o_expect -= s_o_expect
        self.assertTrue(np.array_equal(s_o_expect, s_o_actual))
        self.assertTrue(np.all(np.abs(v_o_expect - v_o_actual) < EPS))

    @staticmethod
    def _random_init(cell, input_shape):
        inputs = np.random.randint(
            low=0, high=10, size=input_shape).astype(np.float32)
        v_i = np.random.uniform(size=cell.state_size[0]).astype(np.float32)
        return inputs, v_i

    @staticmethod
    @abstractmethod
    def _build_cell(**kwargs):
        pass

    @staticmethod
    @abstractmethod
    def _compute_dv(cell, inputs):
        pass

    @staticmethod
    @abstractmethod
    def _count_spikes(cell, inputs):
        pass


# noinspection PyAbstractClass,PyUnresolvedReferences
class TestWeightedSpikingCell(TestSpikingCell):
    def test_no_bias(self):
        cell, input_shape = self._build_cell(use_bias=False)
        for _ in range(N_REPEATS):
            inputs, v_i = self._random_init(cell, input_shape)
            s_o_actual, [v_o_actual] = cell.call(inputs, [v_i])
            self._check_standard(cell, inputs, v_i, s_o_actual, v_o_actual)

    def test_sparse_tracking(self):
        cell, input_shape = self._build_cell(
            track_spikes=True, sparse_tracking=True)
        for _ in range(N_REPEATS):
            inputs, v_i = self._random_init(cell, input_shape)
            mask = np.random.randint(low=0, high=2, size=cell.w.shape)
            cell.w.assign(cell.w * mask)
            s_o_actual, [_, n_actual] = cell.call(
                inputs, [v_i, np.zeros(cell.state_size[1])])
            n_expect = self._count_spikes(cell, inputs)
            self.assertEqual(n_expect, n_actual)

    @classmethod
    def _random_init(cls, cell, input_shape):
        w = np.random.uniform(size=cell.w.shape).astype(np.float32)
        if cell.use_bias:
            b = np.random.uniform(size=cell.b.shape).astype(np.float32)
            cell.set_weights([w, b])
        else:
            cell.set_weights([w])
        return super()._random_init(cell, input_shape)


class TestSpikingConv2DCell(TestWeightedSpikingCell, TestCase):
    def test_build(self):
        cell = SpikingConv2DCell(2, (3, 3), strides=(1, 1), padding="valid")
        self.assertTrue(cell.w is None)
        self.assertTrue(cell.b is None)
        self.assertTrue(cell.state_size is None)
        cell.build((1, 4, 4, 3))
        self.assertEqual((3, 3, 3, 2), cell.w.shape)
        self.assertEqual((2,), cell.b.shape)
        self.assertEqual(1, len(cell.state_size))
        self.assertEqual((2, 2, 2), cell.state_size[0])

    @staticmethod
    def _build_cell(**kwargs):
        cell = SpikingConv2DCell(
            2, (3, 3), strides=(1, 1), padding="valid", **kwargs)
        input_shape = (1, 4, 4, 3)
        cell.build(input_shape)
        return cell, input_shape

    @staticmethod
    def _compute_dv(cell, inputs):
        input_dv = tf.nn.conv2d(inputs, cell.w, (1, 1), "VALID")
        if cell.use_bias:
            return (input_dv + cell.b).numpy()
        else:
            return input_dv.numpy()

    @staticmethod
    def _count_spikes(cell, inputs):
        if cell.sparse_tracking:
            kernel = tf.cast(cell.w != 0.0, inputs.dtype)
        else:
            kernel = tf.ones_like(cell.w)
        return np.sum(tf.nn.conv2d(inputs, kernel, (1, 1), "VALID"))


class TestSpikingDenseCell(TestWeightedSpikingCell, TestCase):
    def test_build(self):
        cell = SpikingDenseCell(2)
        self.assertTrue(cell.w is None)
        self.assertTrue(cell.b is None)
        cell.build((1, 3))
        self.assertEqual(1, len(cell.state_size))
        self.assertEqual((2,), cell.state_size[0])
        self.assertEqual((3, 2), cell.w.shape)
        self.assertEqual((2,), cell.b.shape)

    @staticmethod
    def _build_cell(**kwargs):
        cell = SpikingDenseCell(2, **kwargs)
        input_shape = (1, 3)
        cell.build(input_shape)
        return cell, input_shape

    @staticmethod
    def _compute_dv(cell, inputs):
        input_dv = inputs @ cell.w
        if cell.use_bias:
            return (input_dv + cell.b).numpy()
        else:
            return input_dv.numpy()

    @staticmethod
    def _count_spikes(cell, inputs):
        if cell.sparse_tracking:
            kernel = tf.cast(cell.w != 0.0, inputs.dtype)
        else:
            kernel = tf.ones_like(cell.w)
        return np.sum(inputs @ kernel)


class TestSpikingDepthwiseConv2DCell(TestWeightedSpikingCell, TestCase):
    def test_build(self):
        cell = SpikingDepthwiseConv2DCell(
            (2, 2), strides=(1, 1), padding="valid")
        self.assertTrue(cell.w is None)
        self.assertTrue(cell.b is None)
        self.assertTrue(cell.state_size is None)
        cell.build((1, 5, 5, 3))
        self.assertEqual((2, 2, 3, 1), cell.w.shape)
        self.assertEqual((3,), cell.b.shape)
        self.assertEqual(1, len(cell.state_size))
        self.assertEqual((4, 4, 3), cell.state_size[0])

    def test_depth_multiplier(self):
        cell, input_shape = self._build_cell(depth_multiplier=5)
        for _ in range(N_REPEATS):
            inputs, v_i = self._random_init(cell, (1, 5, 5, 3))
            self.assertEqual((2, 2, 3, 5), cell.w.shape)
            self.assertEqual((15,), cell.b.shape)
            self.assertEqual(1, len(cell.state_size))
            self.assertEqual((4, 4, 15), cell.state_size[0])
            s_o_actual, [v_o_actual] = cell.call(inputs, [v_i])
            self._check_standard(cell, inputs, v_i, s_o_actual, v_o_actual)

    @staticmethod
    def _build_cell(**kwargs):
        cell = SpikingDepthwiseConv2DCell(
            (2, 2), strides=(1, 1), padding="valid", **kwargs)
        input_shape = (1, 5, 5, 3)
        cell.build(input_shape)
        return cell, input_shape

    @staticmethod
    def _compute_dv(cell, inputs):
        input_dv = tf.nn.depthwise_conv2d(
            inputs, cell.w, (1, 1, 1, 1), "VALID")
        if cell.use_bias:
            return (input_dv + cell.b).numpy()
        else:
            return input_dv.numpy()

    @staticmethod
    def _count_spikes(cell, inputs):
        if cell.sparse_tracking:
            kernel = tf.cast(cell.w != 0.0, inputs.dtype)
        else:
            kernel = tf.ones_like(cell.w)
        return np.sum(
            tf.nn.depthwise_conv2d(inputs, kernel, (1, 1, 1, 1), "VALID"))


class TestSpikingInputCell(TestSpikingCell, TestCase):
    @staticmethod
    def _build_cell(**kwargs):
        cell = SpikingInputCell(**kwargs)
        input_shape = (3, 4)
        cell.build(input_shape)
        return cell, input_shape

    @staticmethod
    def _compute_dv(_, inputs):
        return inputs

    @staticmethod
    def _count_spikes(cell, inputs):
        return np.sum(inputs)


if __name__ == "__main__":
    unittest.main()
