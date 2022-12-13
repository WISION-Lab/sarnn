import numpy as np
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import *

from sarnn.components import *
from sarnn.utils import _next_scalable_layer
from sarnn.utils import copy_layer, copy_layer_config, is_scalable, layer_typename, scale

# Layers which will be absorbed during preprocessing and conversion
ABSORBED_LAYERS = [
    "BatchNormalization",
    "PostBatchNormOffset",
    "ReLU",
    "Softmax",
]

ALLOWED_LAYERS = [
    "AveragePooling2D",
    "BatchNormalization",
    "Conv2D",
    "Dense",
    "DepthwiseConv2D",
    "Flatten",
    "InputLayer",
    "PostBatchNormOffset",
    "ReLU",
]

# Layers into which batch normalization can be absorbed
BATCH_NORM_DST_LAYERS = [
    "Conv2D",
    "Dense",
    "DepthwiseConv2D",
]

# Layers which, given a non-negative input, are not guaranteed to
# produce a non-negative output
NON_NONNEGATIVE_PRESERVING_LAYERS = [
    "BatchNormalization",
    "Conv2D",
    "Dense",
    "DepthwiseConv2D",
    "PostBatchNormalizationOffset",
]

# Layers for which a spiking equivalent exists
SPIKING_LAYER_TYPES = [
    "Conv2D",
    "Dense",
    "DepthwiseConv2D",
]


def absorb_batch_norm(ann):
    """
    Absorbs inference-time BatchNormalization into the model weights.

    Also absorbs any PostBatchNormOffset layers.

    The following assumptions must hold:
      1. The layer preceding any BatchNormalization is scalable (see
         BATCH_NORM_DST_LAYERS).
      2. The layer preceding any PostBatchNormOffset is
         BatchNormalization.
    Both of these assumptions are checked by assert_convertible.

    :param tensorflow.keras.Model ann: The model to which the
        absorption should be applied
    :return: A new model without BatchNormalization or
        PostBatchNormOffset layers
    """

    new_ann = Sequential()
    for i, layer in enumerate(ann.layers):
        if layer_typename(layer) == "BatchNormalization":
            _absorb_batch_norm(ann, i, new_ann)
        elif layer_typename(layer) != "PostBatchNormOffset":
            copy_layer(layer, new_ann)

    _recompile(ann, new_ann)
    return new_ann


def assert_convertible(ann):
    """
    Throws an AssertionError if the ANN cannot be converted to an SNN.

    Specifically, checks the following:
      1. The model is an instance of tensorflow.keras.models.Sequential.
      2. The model input shape is fixed and known.
      3. Layers other than the last layer belong to the allowed types
         (see ALLOWED_LAYERS).
      4. All layers have the "channels_last" layout (where this
         attribute exists).
      5. All BatchNormalization layers operate on axis -1.
      6. Softmax activations come only at the last layer.
      7. The inputs to all non-absorbed layers (see ABSORBED_LAYERS) are
         non-negative (this requires ReLUs in the right places).
      8. The converted model will not have any negative activations
         before the final layer.
      9. All BatchNormalization layers are preceded by a scalable
         layer (see BATCH_NORM_DST_LAYERS) or average pooling. In the
         former case, the preceding layer must have a bias. In the
         latter case, average_pooling_to_conv must be called before
         absorb_batch_norm.
      10. All PostBatchNormalizationOffset layers are preceded by a
         a BatchNormalization layer.

    It's a good idea to run this on the ANN as a first step. The rest
    of the library assumes the above are true and may not check them
    explicitly.

    :param tensorflow.keras.Model ann: The model to check for
        convertibility
    """

    assert isinstance(ann, Sequential)
    assert None not in ann.input_shape[1:]

    non_negative = True
    for i, layer in enumerate(ann.layers):
        is_last = i == len(ann.layers) - 1

        if not is_last:
            assert layer_typename(layer) in ALLOWED_LAYERS

        if hasattr(layer, "data_format"):
            assert layer.data_format == "channels_last"

        if layer_typename(layer) == "BatchNormalization":
            assert layer.axis[0] == len(layer.input_shape) - 1

        if layer_typename(layer) == "Softmax":
            assert layer == ann.layers[-1]

        if not is_last and layer_typename(layer) not in ABSORBED_LAYERS:
            assert non_negative
        if layer_typename(layer) == "ReLU":
            non_negative = True
        if layer_typename(layer) in NON_NONNEGATIVE_PRESERVING_LAYERS:
            non_negative = False

        if layer_typename(layer) == "BatchNormalization":
            assert i > 0
            last = ann.layers[i - 1]
            if layer_typename(last) in BATCH_NORM_DST_LAYERS:
                assert last.bias is not None
            else:
                assert layer_typename(last) == "AveragePooling2D"

        if layer_typename(layer) == "PostBatchNormalizationOffset":
            assert i > 0 and layer_typename(ann.layers[i - 1]) == "BatchNormalization"


def average_pooling_to_conv(ann):
    """
    Converts all AveragePooling2D layers to DepthwiseConv2D.

    The weights of each DepthwiseConv2D layer are set such that it is
    equivalent to AveragePooling2D.

    :param tensorflow.keras.Model ann: The model whose AveragePooling2D
        layers should be converted
    :return: A new model with all AveragePooling2D layers replaced by
        DepthwiseConv2D
    """

    new_ann = Sequential()
    for layer in ann.layers:
        if layer_typename(layer) == "AveragePooling2D":
            _average_pooling_to_conv(layer, new_ann)
        else:
            copy_layer(layer, new_ann)

    _recompile(ann, new_ann)
    return new_ann


def build_snn(
        ann,
        batch_size=128,
        time_chunk_size=100,
        spiking_input=False,
        sparse_tracking=False,
        track_spikes=False,
        enable_clamp=False,
        reset_mechanism="subtract",
        t_refrac=0,
        buffer_dv=False):
    """
    Builds an SNN model by replacing ANN layers with SNN layers.

    This function does not modify the input ANN. A new Keras model is
    constructed for the SNN and weights are copied from the ANN.

    Preprocessing and normalization should be performed on the ANN
    before calling this function. For an all-in-one function which also
    performs preprocessing and normalization, see conversion.convert.

    The following layers replacements are performed:
      - Conv2D to RNN(SpikingConv2DCell)
      - Dense to RNN(SpikingDenseCell)
      - DepthwiseConv2D to RNN(SpikingDepthwiseConv2DCell)
      - Other layers of class Type to TimeDistributed(Type)
    ReLU layers are ignored.

    Due to certain restrictions on stateful Keras RNN layers, the batch
    size and number of time steps per computation chunk must be fixed
    here.

    :param tensorflow.keras.Model ann: The ANN model to convert
    :param int batch_size: The SNN batch size
    :param int time_chunk_size: The number of time steps in each SNN
        computation chunk
    :param bool spiking_input: Whether a SpikingInputCell should be
        placed as the first SNN layer
    :param bool sparse_tracking: Whether spikes should only be counted
        if they correspond to a nonzero synapse weight; has no effect if
        track_spikes is False
    :param bool track_spikes: Whether the SNN should count the number of
        incoming spikes at each spiking layer
    :param bool enable_clamp: Whether to enable membrane potential
        clamping; this is useful to allow stabilization of input before
        firing
    :param str reset_mechanism: The post-spike membrane reset mechanism;
        can be either "subtract" for reset by subtraction or "zero" for
        reset to zero
    :param int t_refrac: The duration of the post-spiking refractory
        period; during this period the membrane potential is frozen;
        this is similar to clamping, but the refractory period is
        neuron-specific and not layer-global
    :param bool buffer_dv: Whether spikes should take one timestep to
        propagate to downstream neurons
    :return: The resulting SNN
    """

    # kwargs which are common to all spiking cells
    cell_kwargs = {
        "track_spikes":    track_spikes,
        "enable_clamp":    enable_clamp,
        "accumulate_only": False,
    }

    # kwargs which optionally specify the batch_input_shape
    shape_kwargs = {"batch_input_shape": (batch_size, time_chunk_size) + ann.input_shape[1:]}

    # kwargs which are common to all RNN layer instances
    rnn_kwargs = {
        "return_sequences": True,
        "stateful":         True,
        "trainable":        False,
    }

    snn = Sequential()
    if spiking_input:
        snn.add(RNN(
            SpikingInputCell(
                reset_mechanism="subtract",
                t_refrac=0,
                buffer_dv=False,
                **cell_kwargs),
            **rnn_kwargs, **shape_kwargs))
    cell_kwargs["buffer_dv"] = buffer_dv
    cell_kwargs["reset_mechanism"] = reset_mechanism

    # Used later in accumulate_only logic
    last_spiking = -1
    final_softmax = False
    for i in reversed(range(len(ann.layers))):
        typename = layer_typename(ann.layers[i])
        if typename in SPIKING_LAYER_TYPES:
            last_spiking = i
            break
        if typename == "Softmax":
            final_softmax = True

    for i, layer in enumerate(ann.layers):
        if len(snn.layers) > 0:
            shape_kwargs = {}

        if layer_typename(layer) in SPIKING_LAYER_TYPES:
            # If this is the last spiking layer and there is a final
            # softmax, output raw membrane potentials instead of spikes
            accumulate_only = (i == last_spiking and final_softmax)
            cell_kwargs["accumulate_only"] = accumulate_only

            # An output accumulator doesn't really "spike", so it
            # doesn't make sense to give it a refractory period
            cell_kwargs["t_refrac"] = 0 if accumulate_only else t_refrac

            # kwargs which are common to all weighted spiking cells
            weighted_cell_kwargs = {
                "use_bias":        layer.use_bias,
                "sparse_tracking": sparse_tracking,
                "dtype":           layer.dtype,
            }

            if layer_typename(layer) == "Conv2D":
                cell = SpikingConv2DCell(
                    layer.filters, layer.kernel_size,
                    strides=layer.strides,
                    padding=layer.padding,
                    **weighted_cell_kwargs,
                    **cell_kwargs)
            elif layer_typename(layer) == "Dense":
                cell = SpikingDenseCell(
                    layer.units, **weighted_cell_kwargs, **cell_kwargs)
            else:
                cell = SpikingDepthwiseConv2DCell(
                    layer.kernel_size,
                    strides=layer.strides,
                    padding=layer.padding,
                    depth_multiplier=layer.depth_multiplier,
                    **weighted_cell_kwargs,
                    **cell_kwargs)
            snn.add(RNN(cell, **rnn_kwargs, **shape_kwargs))
        elif layer_typename(layer) in ["ReLU", "Softmax"]:
            continue
        else:
            snn.add(TimeDistributed(copy_layer_config(layer), **shape_kwargs))

    snn.set_weights(ann.get_weights())
    return snn


def convert(
        ann, x_norm,
        percentile=99.0,
        layer_wise=True,
        recompute=False,
        batch_size=128,
        time_chunk_size=10,
        spiking_input=False,
        sparse_tracking=False,
        track_spikes=False,
        enable_clamp=False,
        reset_mechanism="subtract",
        t_refrac=0,
        buffer_dv=False):
    """
    An all-in-one function which creates an SNN from an ANN.

    The equivalent of calling::

        ann = preprocess(ann)
        normalize(ann, ...)
        snn = build_snn(ann, ...)

    The input ANN is not modified.

    :param tensorflow.keras.Model ann: The ANN which should be converted
    :param numpy.ndarray x_norm: The data to pass through the model
        during normalization
    :param float percentile: The normalization percentile
    :param bool layer_wise: Whether to apply the same normalization
        scale to all neurons in each layer
    :param bool recompute: Whether to recompute each layer's activations
        from scratch during normalization; this should be used if
        normalization exhausts the system memory
    :param int batch_size: The SNN batch size
    :param int time_chunk_size: The number of time steps in each SNN
        computation chunk
    :param bool spiking_input: Whether a SpikingInputCell should be
        placed as the first SNN layer
    :param bool sparse_tracking: Whether spikes should only be counted
        if they correspond to a nonzero synapse weight; has no effect if
        track_spikes is False
    :param bool track_spikes: Whether the SNN should count the number of
        incoming spikes at each spiking layer
    :param bool enable_clamp: Whether to enable membrane potential
        clamping; this is useful to allow stabilization of input before
        firing
    :param str reset_mechanism: The post-spike membrane reset mechanism;
        can be either "subtract" for reset by subtraction or "zero" for
        reset to zero
    :param int t_refrac: The duration of the post-spiking refractory
        period; during this period the membrane potential is frozen;
        this is similar to clamping, but the refractory period is
        neuron-specific and not layer-global
    :param bool buffer_dv: Whether spikes should take one timestep to
        propagate to downstream neurons
    :return: The resulting SNN
    """

    ann = preprocess(ann)
    normalize(
        ann, x_norm,
        percentile=percentile,
        layer_wise=layer_wise,
        recompute=recompute)
    snn = build_snn(
        ann,
        batch_size=batch_size,
        time_chunk_size=time_chunk_size,
        spiking_input=spiking_input,
        sparse_tracking=sparse_tracking,
        track_spikes=track_spikes,
        enable_clamp=enable_clamp,
        reset_mechanism=reset_mechanism,
        t_refrac=t_refrac,
        buffer_dv=buffer_dv)
    return snn


def normalize(ann, x_norm, percentile=99.0, layer_wise=True, recompute=False):
    """
    Performs data-based activation normalization on an ANN.

    Activations are scaled such that the percentile'th activation is
    1.0.

    If layer_wise=False, layers before the last are scaled at the neuron
    level and the last layer is scaled at the layer level. If
    layer_wise=True, all scaling is performed at the layer level.

    Warning: This function has a very high memory footprint. If
    system memory is exhausted, set recompute=True or provide a subset
    of the dataset for x_norm (e.g. x_train[:1000]).

    :param tensorflow.keras.Model ann: The model whose activations
        should be normalized
    :param numpy.ndarray x_norm: The data to pass through the model
        during normalization
    :param float percentile: The percentile of activations which should
        be scaled to 1.0
    :param bool layer_wise: Whether to apply the same normalization
        scale to all neurons in each layer
    :param bool recompute: Whether to recompute each layer's activations
        from scratch during normalization; this should be used if
        normalization exhausts the system memory
    :return: The scaling factors applied to the model
    """

    i_start = 0
    data = x_norm
    scales = []

    for i, layer in enumerate(ann.layers):
        if not is_scalable(layer):
            continue

        if recompute:
            norm_model = Model(inputs=ann.inputs, outputs=ann.layers[i].output)
            data = norm_model.predict(x_norm)
        else:
            norm_model = Sequential()
            for j in range(i_start, i + 1):
                copy_layer(ann.layers[j], norm_model)
            i_start = i + 1
            data = norm_model.predict(data)

        if not layer_wise and _next_scalable_layer(ann, i) is not None:
            scales.append(_neuron_scales(data, percentile, recompute))
        else:
            scales.append(_single_scale(data, percentile, recompute))

    scale(ann, scales)
    return scales


def preprocess(ann):
    """
    Prepares the ANN for normalization and SNN conversion.

    The equivalent of calling::

        assert_convertible(ann)
        ann = average_pooling_to_conv(ann)
        ann = absorb_batch_norm(ann)

    This is more efficient than calling these functions individually
    since two separate models are not materialized.

    :param tensorflow.keras.Model ann: The model to preprocess
    :return: The preprocessed model
    """

    assert_convertible(ann)

    new_ann = Sequential()
    for i, layer in enumerate(ann.layers):
        if layer_typename(layer) == "AveragePooling2D":
            _average_pooling_to_conv(layer, new_ann)
        elif layer_typename(layer) == "BatchNormalization":
            _absorb_batch_norm(ann, i, new_ann)
        elif layer_typename(layer) != "PostBatchNormOffset":
            copy_layer(layer, new_ann)

    _recompile(ann, new_ann)
    return new_ann


def _absorb_batch_norm(ann, i, new_ann):
    batch_norm = ann.layers[i]
    dst = new_ann.layers[-1]

    mean = batch_norm.moving_mean
    std = np.sqrt(batch_norm.moving_variance + batch_norm.epsilon)
    gamma = batch_norm.gamma if batch_norm.gamma is not None else 1.0
    beta = batch_norm.beta if batch_norm.beta is not None else 0.0

    if i < len(ann.layers) - 1:
        next_layer = ann.layers[i + 1]
        if layer_typename(next_layer) == "PostBatchNormOffset":
            beta += next_layer.offset

    # Adjust weights of the previous layer
    w, b = dst.get_weights()
    w_scale = gamma / std
    if layer_typename(dst) == "DepthwiseConv2D":
        w_scale = np.reshape(w_scale, w.shape[-2:])
    w = w_scale * w
    b = gamma / std * (b - mean) + beta
    dst.set_weights([w, b])


def _average_pooling_to_conv(layer, new_ann):
    if len(new_ann.layers) == 0:
        shape_kwargs = {"batch_input_shape": layer.input_shape}
    else:
        shape_kwargs = {}

    # Use a bias (set to zero) so batch norm can be correctly absorbed
    new_layer = DepthwiseConv2D(
        layer.pool_size,
        strides=layer.strides,
        padding=layer.padding,
        use_bias=True,
        **shape_kwargs)
    new_ann.add(new_layer)

    # Set weights so the convolution is equivalent to pooling
    kernel = np.ones(layer.pool_size + (1, 1)) / np.prod(layer.pool_size)
    w = np.repeat(kernel, layer.input_shape[-1], axis=-2)
    b = np.zeros(new_layer.get_weights()[1].shape)
    new_layer.set_weights([w, b])


def _neuron_scales(data, percentile, overwrite):
    scales = np.empty(data.shape[-1])
    for j in range(data.shape[-1]):
        scales[j] = _single_scale(data[..., j], percentile, overwrite)
    return scales


def _single_scale(data, percentile, overwrite):
    p = np.percentile(data, percentile, overwrite_input=overwrite)
    return 1.0 / p if p > 0.0 else 1.0


def _recompile(ann, new_ann):
    if hasattr(ann, "loss"):
        new_ann.compile(
            optimizer=ann.optimizer,
            loss=ann.loss,
            metrics=ann.metrics)
