import os
import unittest
from unittest import TestCase

from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential

from sarnn.conversion import preprocess
from sarnn.utils import *

# Threshold (epsilon) for assuming numerical equality
EPS = 1e-5


class TestCopyLayerConfig(TestCase):
    def setUp(self):
        self.layer = Conv2D(16, (3, 3), input_shape=(32, 32, 1))
        self.copy = copy_layer_config(self.layer)

    def test_copy_layer_config(self):
        layer_config = self.layer.get_config()
        copy_config = self.copy.get_config()
        for key in layer_config:
            if key != "name":
                self.assertEqual(layer_config[key], copy_config[key])

    def test_has_input_shape(self):
        self.assertTrue("batch_input_shape" in self.copy.get_config())

    def test_name_unique(self):
        self.assertNotEqual(self.layer.name, self.copy.name)

    def test_type_matches(self):
        self.assertEqual(type(self.layer), type(self.copy))


class TestCountLayerSynapses(TestCase):
    def test_conv2d_average_pooling_2d(self):
        model = Sequential()
        model.add(Conv2D(16, (3, 3), padding="same", input_shape=(32, 32, 3)))
        model.add(AveragePooling2D((2, 2)))
        expect = 32 * 32 * 16
        self.assertEqual(expect, count_layer_synapses(model.layers[1]))

    def test_conv2d_conv2d(self):
        model = Sequential()
        model.add(Conv2D(16, (3, 3), padding="same", input_shape=(32, 32, 3)))
        model.add(Conv2D(16, (3, 3), padding="same"))
        expect = (30 * 30 * 9 + 30 * 4 * 6 + 4 * 4) * 16 * 16
        self.assertEqual(expect, count_layer_synapses(model.layers[1]))

    def test_conv2d_dense(self):
        model = Sequential()
        model.add(Conv2D(16, (3, 3), padding="same", input_shape=(32, 32, 3)))
        model.add(Flatten())
        model.add(Dense(10))
        expect = 32 * 32 * 16 * 10
        self.assertEqual(expect, count_layer_synapses(model.layers[2]))

    def test_conv2d_depthwise_conv2d(self):
        model = Sequential()
        model.add(Conv2D(16, (3, 3), padding="same", input_shape=(32, 32, 3)))
        model.add(DepthwiseConv2D((3, 3), padding="same", depth_multiplier=5))
        expect = (30 * 30 * 9 + 30 * 4 * 6 + 4 * 4) * 16 * 5
        self.assertEqual(expect, count_layer_synapses(model.layers[1]))

    def test_dense_average_pooling_2d(self):
        model = Sequential()
        model.add(Dense(300, input_shape=(100,)))
        model.add(Reshape((10, 10, 3)))
        model.add(AveragePooling2D((2, 2)))
        expect = 10 * 10 * 3
        self.assertEqual(expect, count_layer_synapses(model.layers[2]))

    def test_dense_conv2d(self):
        model = Sequential()
        model.add(Dense(300, input_shape=(100,)))
        model.add(Reshape((10, 10, 3)))
        model.add(Conv2D(16, (3, 3), padding="same"))
        expect = (8 * 8 * 9 + 8 * 4 * 6 + 4 * 4) * 3 * 16
        self.assertEqual(expect, count_layer_synapses(model.layers[2]))

    def test_dense_dense(self):
        model = Sequential()
        model.add(Dense(300, input_shape=(100,)))
        model.add(Dense(10))
        expect = 300 * 10
        self.assertEqual(expect, count_layer_synapses(model.layers[1]))

    def test_dense_depthwise_conv2d(self):
        model = Sequential()
        model.add(Dense(300, input_shape=(100,)))
        model.add(Reshape((10, 10, 3)))
        model.add(DepthwiseConv2D((3, 3), padding="same", depth_multiplier=5))
        expect = (8 * 8 * 9 + 8 * 4 * 6 + 4 * 4) * 3 * 5
        self.assertEqual(expect, count_layer_synapses(model.layers[2]))

    def test_multilayer(self):
        model = _build_multilayer_ann()
        expect_1 = (30 * 30 * 9 + 30 * 4 * 6 + 4 * 4) * 16 * 16
        self.assertEqual(expect_1, count_layer_synapses(model.layers[3]))
        expect_2 = 32 * 32 * 16
        self.assertEqual(expect_2, count_layer_synapses(model.layers[6]))
        expect_3 = 16 * 16 * 16 * 10
        self.assertEqual(expect_3, count_layer_synapses(model.layers[8]))

    def test_spiking_conv2d_conv2d(self):
        model = Sequential()
        model.add(RNN(
            SpikingConv2DCell(16, (3, 3), padding="same"),
            return_sequences=True,
            stateful=True,
            batch_input_shape=(1, 10, 32, 32, 3)))
        model.add(RNN(
            SpikingConv2DCell(16, (3, 3), padding="same"),
            return_sequences=True,
            stateful=True))
        expect = (30 * 30 * 9 + 30 * 4 * 6 + 4 * 4) * 16 * 16
        self.assertEqual(expect, count_layer_synapses(model.layers[1]))

    def test_spiking_dense_dense(self):
        model = Sequential()
        model.add(RNN(
            SpikingDenseCell(10),
            return_sequences=True,
            stateful=True,
            batch_input_shape=(1, 10, 256)))
        model.add(RNN(
            SpikingDenseCell(10), return_sequences=True, stateful=True))
        expect = 10 * 10
        self.assertEqual(expect, count_layer_synapses(model.layers[1]))

    def test_sparse_conv2d_conv2d(self):
        model = Sequential()
        model.add(Conv2D(16, (3, 3), padding="same", input_shape=(32, 32, 3)))
        model.add(Conv2D(16, (3, 3), padding="same"))
        weights = model.layers[-1].get_weights()
        weights[0][:] = 0.0
        weights[0][..., 0] = 0.1
        model.layers[-1].set_weights(weights)
        expect = (30 * 30 * 9 + 30 * 4 * 6 + 4 * 4) * 16 * 16 / 16
        self.assertEqual(expect, count_layer_synapses(
            model.layers[1], sparse_counting=True))

    def test_sparse_conv2d_dense(self):
        model = Sequential()
        model.add(Conv2D(16, (3, 3), padding="same", input_shape=(32, 32, 3)))
        model.add(Flatten())
        model.add(Dense(10))
        weights = model.layers[-1].get_weights()
        weights[0][:] = 0.0
        weights[0][3200, 3] = 0.2
        weights[0][1887, 0] = -1.0
        weights[0][0, 9] = 1e-10
        model.layers[-1].set_weights(weights)
        expect = 3
        self.assertEqual(expect, count_layer_synapses(
            model.layers[2], sparse_counting=True))

    def test_sparse_conv2d_depthwise_conv2d(self):
        model = Sequential()
        model.add(Conv2D(16, (3, 3), padding="same", input_shape=(32, 32, 3)))
        model.add(DepthwiseConv2D((3, 3), padding="same", depth_multiplier=5))
        weights = model.layers[-1].get_weights()
        weights[0][:] = 0.0
        weights[0][..., 15, :] = -0.5
        weights[0][..., 2, :] = 10.0
        model.layers[-1].set_weights(weights)
        expect = (30 * 30 * 9 + 30 * 4 * 6 + 4 * 4) * 16 * 5 / 8
        self.assertEqual(expect, count_layer_synapses(
            model.layers[1], sparse_counting=True))

    def test_spiking_multilayer(self):
        model = _build_multilayer_snn()
        expect_1 = (30 * 30 * 9 + 30 * 4 * 6 + 4 * 4) * 16 * 16
        self.assertEqual(expect_1, count_layer_synapses(model.layers[1]))
        expect_2 = 32 * 32 * 16
        self.assertEqual(expect_2, count_layer_synapses(model.layers[2]))
        expect_3 = 16 * 16 * 16 * 10
        self.assertEqual(expect_3, count_layer_synapses(model.layers[4]))


class TestCountModelSynapses(TestCase):
    def test_multilayer(self):
        model = _build_multilayer_ann()
        expect = ((30 * 30 * 9 + 30 * 4 * 6 + 4 * 4) * 16 * 3
                  + (30 * 30 * 9 + 30 * 4 * 6 + 4 * 4) * 16 * 16
                  + 32 * 32 * 16
                  + 16 * 16 * 16 * 10)
        self.assertEqual(expect, count_model_synapses(model, skip_first=False))

    def test_spiking_multilayer(self):
        model = _build_multilayer_snn()
        expect = ((30 * 30 * 9 + 30 * 4 * 6 + 4 * 4) * 16 * 3
                  + (30 * 30 * 9 + 30 * 4 * 6 + 4 * 4) * 16 * 16
                  + 32 * 32 * 16
                  + 16 * 16 * 16 * 10)
        self.assertEqual(expect, count_model_synapses(model, skip_first=False))


class TestLoadFilename(TestCase):
    def test_load_custom(self):
        model = _build_multilayer_snn()
        filename = "custom.h5"
        model.save(filename)
        load_model(filename)
        os.remove(filename)

    def test_load_vanilla(self):
        model = _build_multilayer_ann()
        filename = "vanilla.h5"
        model.save(filename)
        load_model(filename)
        os.remove(filename)


class TestScale(TestCase):
    def setUp(self):
        np.random.seed(0)
        model = preprocess(_build_multilayer_ann())
        outputs = []
        for layer in model.layers:
            if is_scalable(layer):
                outputs.append(layer.output)
        self.n = len(outputs)
        self.model = Model(inputs=model.inputs, outputs=outputs)

    def randomize_weights(self):
        for layer in self.model.layers:
            weights = layer.get_weights()
            for i in range(len(weights)):
                weights[i] = np.random.uniform(
                    low=0.1, high=10.0, size=weights[i].shape)
            layer.set_weights(weights)

    def test_invalid_scales(self):
        for i in range(2, self.n):
            with self.assertRaises(ValueError):
                scale(self.model, [1.0] * i)
        with self.assertRaises(ValueError):
            scale(self.model, [1.0] * (self.n + 1))
        with self.assertRaises(ValueError):
            scales = scaling_template(self.model, 3)
            scales[0] = np.append(scales[0], 1.0)
            scale(self.model, scales)

    def test_layer_wise_scale(self):
        for _ in range(10):
            self.randomize_weights()
            inputs = np.random.uniform(size=(1,) + self.model.input_shape[1:])
            outputs_1 = self.model.predict(inputs)
            scales = list(np.random.uniform(low=0.5, high=2.0, size=self.n))
            scale(self.model, scales)
            outputs_2 = self.model.predict(inputs)
            for i in range(self.n):
                frac = outputs_2[i] / outputs_1[i]
                self.assertTrue(np.all(np.abs(frac - scales[i]) < EPS))

    def test_network_wise_scale(self):
        for _ in range(10):
            self.randomize_weights()
            inputs = np.random.uniform(size=(1,) + self.model.input_shape[1:])
            outputs_1 = self.model.predict(inputs)
            scales = [np.random.uniform(low=0.5, high=2.0)]
            scale(self.model, scales)
            outputs_2 = self.model.predict(inputs)
            for i in range(self.n):
                frac = outputs_2[i] / outputs_1[i]
                self.assertTrue(np.all(np.abs(frac - scales[0]) < EPS))

    def test_neuron_wise_scale(self):
        for _ in range(10):
            self.randomize_weights()
            inputs = np.random.uniform(size=(1,) + self.model.input_shape[1:])
            outputs_1 = self.model.predict(inputs)
            scales = scaling_template(self.model, 3)
            for i in range(self.n):
                if isinstance(scales[i], np.ndarray):
                    scales[i] = np.random.uniform(
                        low=0.5, high=2.0, size=len(scales[i]))
                else:
                    scales[i] = np.random.uniform(low=0.5, high=2.0)
            scale(self.model, scales)
            outputs_2 = self.model.predict(inputs)
            for i in range(self.n):
                if isinstance(scales[i], float):
                    frac = outputs_2[i] / outputs_1[i]
                    self.assertTrue(np.all(np.abs(frac - scales[i]) < EPS))
                else:
                    for j in range(len(scales[i])):
                        frac = outputs_2[i][..., j] / outputs_1[i][..., j]
                        self.assertTrue(
                            np.all(np.abs(frac - scales[i][j]) < EPS))


class TestScalingTemplate(TestCase):
    def assertArrayListsEqual(self, array_list_1, array_list_2):
        for x, y in zip(array_list_1, array_list_2):
            if isinstance(x, np.ndarray):
                self.assertTrue(np.array_equal(x, y))
            else:
                self.assertEqual(x, y)

    def test_scaling_template_1(self):
        model = preprocess(_build_multilayer_ann())
        expect = [1.0]
        self.assertArrayListsEqual(
            expect, scaling_template(model, granularity=1))

    def test_scaling_template_2(self):
        model = preprocess(_build_multilayer_ann())
        expect = [1.0] * 4
        self.assertArrayListsEqual(
            expect, scaling_template(model, granularity=2))

    def test_scaling_template_3(self):
        model = preprocess(_build_multilayer_ann())
        expect = [np.ones(16), np.ones(16), np.ones(16), 1.0]
        self.assertArrayListsEqual(
            expect, scaling_template(model, granularity=3))

    def test_scaling_template_spiking(self):
        snn = _build_multilayer_snn()
        ann = preprocess(_build_multilayer_ann())
        for g in 1, 2, 3:
            self.assertArrayListsEqual(
                scaling_template(snn, granularity=g),
                scaling_template(ann, granularity=g))


def _build_multilayer_ann():
    model = Sequential()
    model.add(Conv2D(16, (3, 3), padding="same", input_shape=(32, 32, 3)))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Conv2D(16, (3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(AveragePooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(10))
    model.add(Softmax())
    return model


def _build_multilayer_snn():
    model = Sequential()
    model.add(RNN(
        SpikingConv2DCell(16, (3, 3), padding="same"),
        return_sequences=True,
        stateful=True,
        batch_input_shape=(1, 10, 32, 32, 3)))
    model.add(RNN(
        SpikingConv2DCell(16, (3, 3), padding="same"),
        return_sequences=True,
        stateful=True))
    model.add(RNN(
        SpikingDepthwiseConv2DCell((2, 2), strides=(2, 2)),
        return_sequences=True,
        stateful=True))
    model.add(Reshape((10, 16 * 16 * 16)))
    model.add(RNN(SpikingDenseCell(10), return_sequences=True, stateful=True))
    return model


if __name__ == "__main__":
    unittest.main()
