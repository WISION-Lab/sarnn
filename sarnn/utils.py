import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.utils import CustomObjectScope

from sarnn.components import *

CUSTOM_OBJECTS = {
    "BatchNormSparsityRegularizer": BatchNormSparsityRegularizer,
    "L1WeightRegularizer":          L1WeightRegularizer,
    "PostBatchNormOffset":          PostBatchNormOffset,
    "SpikingConv2DCell":            SpikingConv2DCell,
    "SpikingDenseCell":             SpikingDenseCell,
    "SpikingDepthwiseConv2DCell":   SpikingDepthwiseConv2DCell,
    "SpikingInputCell":             SpikingInputCell,
}


def compute_sparsity(ann, x, layer_types=("ReLU",), skip_final=True):
    """
    Computes the fraction of ANN activations which are zero.

    :param tensorflow.keras.Model ann: The model to examine
    :param numpy.ndarray x: Data to pass through the model
    :param tuple layer_types: A tuple of valid layer type names
    :param bool skip_final: Whether to skip the last valid layer (as
        this is often the classifier)
    :return float: The fraction of activations which are nonzero
    """

    # Count of all activations
    a = tf.zeros((), dtype=tf.int64)

    # Count of nonzero activations
    n = tf.zeros((), dtype=tf.int64)

    # Construct a temporary model whose only job is counting activations
    skip = skip_final
    for i, layer in reversed(list(enumerate(ann.layers))):
        if not layer_typename(layer) in layer_types:
            continue
        if skip:
            skip = False
            continue
        a += tf.size(layer.output, out_type=tf.int64)
        n += tf.math.count_nonzero(layer.output)
    count_model = Model(inputs=ann.inputs, outputs=[a, n])

    a_out, n_out = count_model.predict(x)
    return 1.0 - np.sum(n_out) / np.sum(a_out)


def copy_layer(layer, sequential):
    new_layer = copy_layer_config(layer)
    sequential.add(new_layer)
    new_layer.set_weights(layer.get_weights())


def copy_layer_config(layer):
    """
    Copies a layer's configuration to a new layer.

    The name of the new layer is set to None; this allows Keras to
    automatically determine a unique name.

    The layer weights are not copied. Do this manually if this is the
    desired behavior.

    :param tensorflow.keras.Layer layer: The layer to copy
    :return: A new layer with the same configuration as the input layer
    """

    config = layer.get_config()
    config["name"] = None
    if "batch_input_shape" not in config and hasattr(layer, "input_shape"):
        config["batch_input_shape"] = layer.input_shape
    with CustomObjectScope(CUSTOM_OBJECTS):
        return type(layer).from_config(config)


def count_layer_synapses(layer, sparse_counting=False, epsilon=None):
    """
    Counts the number of synapses incoming to a layer.

    :param tensorflow.keras.Layer layer: The layer whose synapses should
        be counted
    :param bool sparse_counting: Whether synapses should only be counted
        if they have a nonzero weight
    :param float epsilon: When sparse_counting is enabled, weights whose
        absolute values are less than epsilon are considered to be zero
    :return int: The number of synapses incoming to the layer, or zero
        if the layer is not one of the recognized types
    """

    if layer_typename(layer) == "RNN":
        input_shape = layer.input_shape[2:]
        layer = layer.cell
    else:
        input_shape = layer.input_shape[1:]

    if len(layer.get_weights()) > 0:
        kernel = layer.get_weights()[0]
        if sparse_counting:
            if epsilon is None:
                synapses = kernel != 0.0
            else:
                synapses = np.abs(kernel) > epsilon
        else:
            synapses = np.ones_like(kernel)

        if layer_typename(layer) in ["Dense", "SpikingDenseCell"]:
            return int(np.sum(synapses))

        if layer_typename(layer) in ["Conv2D", "SpikingConv2DCell"]:
            density_map = tf.nn.conv2d(
                tf.ones((1,) + input_shape),
                synapses.astype(np.float32),
                layer.strides,
                layer.padding.upper())
            return int(np.sum(density_map))

        if layer_typename(layer) in [
            "DepthwiseConv2D", "SpikingDepthwiseConv2DCell"]:
            # depthwise_conv2d expects exactly 4 stride values
            strides = layer.strides
            if len(strides) == 2:
                strides = (1,) + strides + (1,)

            density_map = tf.nn.depthwise_conv2d(
                tf.ones((1,) + input_shape),
                synapses.astype(np.float32),
                strides, layer.padding.upper())
            return int(np.sum(density_map))

    if layer_typename(layer) == "AveragePooling2D":
        density_map = tf.nn.avg_pool2d(
            tf.ones((1,) + input_shape),
            layer.pool_size,
            layer.strides,
            layer.padding.upper())
        return int(np.prod(layer.pool_size) * np.sum(density_map))

    return 0


def count_model_synapses(model, sparse_counting=False, epsilon=None, skip_first=False):
    """
    Counts the total number of synapses in a model.

    :param tensorflow.keras.Model model: The model whose synapses should
        be counted
    :param bool sparse_counting: Whether synapses should only be counted
        if they have a nonzero weight
    :param float epsilon: When sparse_counting is enabled, weights whose
        absolute values are less than epsilon are considered to be zero
    :param bool skip_first: Whether the model input synapses should be
        ignored
    :return: The number of synapses in the model
    """

    model_synapses = 0
    first = True
    for layer in model.layers:
        if is_synaptic(layer):
            if first:
                first = False
                if skip_first:
                    continue
            model_synapses += count_layer_synapses(
                layer, sparse_counting=sparse_counting, epsilon=epsilon)
    return model_synapses


def is_scalable(layer):
    return _is_conv2d(layer) or _is_dense(layer) or _is_depthwise_conv2d(layer)


def is_synaptic(layer):
    return (layer_typename(layer) in ["AveragePooling2D", "Conv2D", "Dense", "DepthwiseConv2D"]
            or (layer_typename(layer) == "RNN"
                and layer_typename(layer.cell)
                in ["SpikingConv2DCell", "SpikingDenseCell", "SpikingDepthwiseConv2DCell"]))


def layer_typename(layer):
    """
    isinstance(layer, class) sometimes behaves unexpectedly, so we use
    this instead.

    :param tensorflow.keras.Layer layer: The layer whose typename should
        be returned
    :return: The layer's typename (a string).
    """

    return layer.__class__.__name__


def load_model(filename):
    """
    Loads a model which includes custom sarnn components (see
    CUSTOM_OBJECTS for all such components).

    :param string filename: The filename of the model to load
    :return: The loaded model
    """

    with CustomObjectScope(CUSTOM_OBJECTS):
        return tf.keras.models.load_model(filename)


def prune(model, epsilon):
    """
    Zeros out kernel weights which are below some epsilon.

    This is performed in-place on the provided model.

    :param tensorflow.keras.Model model: The model whose weights should be
        modified
    :param float epsilon: Kernel weights whose absolute values are less than
        epsilon are zeroed out
    """

    for i, layer in enumerate(model.layers):
        weights = layer.get_weights()
        if len(weights) == 0:
            continue
        weights[0] *= np.abs(weights[0]) >= epsilon
        layer.set_weights(weights)


def scale(model, scales):
    """
    Tunes the activations of each model layer by the specified scales.

    If the scales list contains only one floating-point item, the
    activations of the entire model are scaled by that amount.
    Otherwise, the scales list should contain one item for each scalable
    layer in the model. If the item is a single floating-point value,
    then all activations in the layer are scaled by that value. If the
    item is a NumPy array (with a number of elements equal to the number
    of neurons/channels in the layer), then neurons are individually
    tuned.

    Scaling is performed in-place.

    :param tensorflow.keras.Model model: The model whose activations
        should be tuned
    :param list scales: The list of tuning scales
    """

    n_scalable = sum(is_scalable(layer) for layer in model.layers)
    if len(scales) not in (1, n_scalable):
        s = "expected scales to have length 1 or {}, found length {}"
        raise ValueError(s.format(n_scalable, len(scales)))
    if len(scales) != n_scalable and not isinstance(scales[0], float):
        s = "non-float value of type {} found for global scaling factor"
        raise ValueError(s.format(type(scales[0])))

    k = 0
    for i, layer in enumerate(model.layers):
        if not is_scalable(layer):
            continue
        layer_scales = scales[k] if len(scales) > 1 else scales[0]
        k += 1
        if isinstance(layer_scales, float):
            layer_scales = np.repeat(layer_scales, layer.output_shape[-1])

        if layer_scales.size != layer.output_shape[-1]:
            s = "expected {} layer scales but found {}"
            raise ValueError(s.format(layer_scales.size, layer.output_shape[-1]))

        _scale_output(layer, layer_scales)
        next_scalable = _next_scalable_layer(model, i)
        if next_scalable is not None:
            _compensate_for_input(next_scalable, layer, layer_scales)


def scaling_template(model, granularity, value=1.0):
    """
    Returns a tuning template of the correct shape to be passed to
    scale(model, scales).

    :param tensorflow.keras.Model model: The model for which a tuning
        template should be created
    :param int granularity: The granularity of the tuning template;
        1=network-wise, 2=layer-wise, 3=neuron-wise
    :param float value: The value with which the template should be filled
    :return list template: A list of the correct shape
    """

    if granularity not in (1, 2, 3):
        s = "granularity {} is not one of the allowed values: 1, 2, or 3"
        raise ValueError(s.format(granularity))

    if granularity == 1:
        template = [value]
    else:
        template = []
        for i, layer in enumerate(model.layers):
            if not is_scalable(layer):
                continue
            if granularity == 3 and _next_scalable_layer(model, i) is not None:
                template.append(np.full(layer.output_shape[-1], value))
            else:
                template.append(value)
    return template


def v_initial_template(model, granularity, value=1.0):
    """
    Returns an initialization template of the correct shape to be provided to
    simulation.simulate or simulation.evaluate.

    :param tensorflow.keras.Model model: The model for which an initialization
        template should be created
    :param int granularity: The granularity of the initialization template;
        1=network-wise, 2=layer-wise, 3=neuron-wise
    :param float value: The value with which the template should be filled
    :return list template: A list of the correct shape
    """
    if granularity not in (1, 2, 3):
        s = "granularity {} is not one of the allowed values: 1, 2, or 3"
        raise ValueError(s.format(granularity))

    if granularity == 1:
        template = [value]
    else:
        template = []
        for i, layer in enumerate(model.layers):
            if not layer_typename(layer) == "RNN":
                continue
            if granularity == 2:
                template.append(value)
            else:
                template.append(np.full(layer.output_shape[-1], value))
    return template


def _compensate_for_input(layer, prev_layer, scales):
    weights = layer.get_weights()
    w = weights[0]

    if _is_conv2d(layer) or _is_depthwise_conv2d(layer):
        if _is_conv2d(prev_layer) or _is_depthwise_conv2d(prev_layer):
            for j, s in enumerate(scales):
                w[..., j, :] /= s
        if _is_dense(prev_layer):
            raise NotImplementedError("scaling of 2d layers after dense is not yet supported")

    if _is_dense(layer):
        if _is_conv2d(prev_layer) or _is_depthwise_conv2d(prev_layer):
            old_shape = w.shape
            w = np.reshape(w, prev_layer.output_shape[-3:] + layer.output_shape[-1:])
            for j, s in enumerate(scales):
                w[..., j, :] /= s
            w = np.reshape(w, old_shape)
        if _is_dense(prev_layer):
            for j, s in enumerate(scales):
                w[j, :] /= s

    layer.set_weights(weights)


def _is_conv2d(layer):
    return (layer_typename(layer) == "Conv2D"
            or (layer_typename(layer) == "RNN"
                and layer_typename(layer.cell) == "SpikingConv2DCell"))


def _is_dense(layer):
    return (layer_typename(layer) == "Dense"
            or (layer_typename(layer) == "RNN"
                and layer_typename(layer.cell) == "SpikingDenseCell"))


def _is_depthwise_conv2d(layer):
    return (layer_typename(layer) == "DepthwiseConv2D"
            or (layer_typename(layer) == "RNN"
                and layer_typename(layer.cell) == "SpikingDepthwiseConv2DCell"))


def _next_scalable_layer(model, i):
    for j in range(i + 1, len(model.layers)):
        if is_scalable(model.layers[j]):
            return model.layers[j]
    return None


def _scale_output(layer, scales):
    weights = layer.get_weights()
    w = weights[0]
    b = weights[1] if len(weights) > 1 else None

    for j, s in enumerate(scales):
        if _is_conv2d(layer) or _is_dense(layer):
            w[..., j] *= s

        if _is_depthwise_conv2d(layer):
            if layer_typename(layer) == "RNN":
                m = layer.cell.depth_multiplier
            else:
                m = layer.depth_multiplier
            w[..., j // m, j % m] *= s

        if b is not None:
            b[j] *= s

    layer.set_weights(weights)
