from abc import abstractmethod

import tensorflow as tf
from tensorflow.keras.layers import Layer


class BatchNormSparsityRegularizer:
    """
    A BatchNormalization beta_regularizer which penalizes non-sparsity.
    """

    def __init__(self, beta_penalty, layer_neurons=None, model_neurons=None):
        """
        BatchNormSparsityRegularizer constructor.

        :param float beta_penalty: The amount by which the penalty
            should be scaled
        :param int layer_neurons: The number of outputs neurons in the
            parent BatchNormalization layer; used to scale the penalty;
            if None, it is assumed this will be provided later
        :param int model_neurons: The total number of neurons in the
            model; used to normalize the penalty; if None, it is assumed
            this will be provided later
        """

        self.beta_penalty = beta_penalty
        self.layer_neurons = layer_neurons
        self.model_neurons = model_neurons

    def __call__(self, beta):
        """
        Computes the non-sparsity penalty.

        :param tensorflow.Tensor beta: A tensor of BatchNormalization
            betas
        :return float: The value of the penalty
        """

        # Since we're using reduce_mean, multiply by the number of
        # neurons in the layer and not the number of neurons
        # corresponding to one beta
        return (self.beta_penalty
                * self.layer_neurons
                / self.model_neurons
                * tf.reduce_mean(
                    0.5 * (tf.math.erf(beta / tf.sqrt(2.0)) + 1.0)))

    def get_config(self):
        """
        Returns the configuration of the regularizer as Python
        dictionary.

        :return: The serialized configuration
        """

        return {
            "beta_penalty":  self.beta_penalty,
            "layer_neurons": self.layer_neurons,
            "model_neurons": self.model_neurons
        }


class L1WeightRegularizer:
    """
    A kernel_regularizer which employs an L1 penalty scaled by the
    number of incoming layer synapses as a fraction of the number of
    model synapses.

    Optionally may include a traditional L2 weight decay term.
    """

    def __init__(
            self, l1_synapse_penalty,
            l2_decay=None,
            layer_synapses=None,
            model_synapses=None):
        """
        L1WeightRegularizer constructor.

        :param float l1_synapse_penalty: The amount by which the L1
            penalty should be scaled
        :param float l2_decay: The amount of L2 weight decay to add;
            if None, weight decay is not added
        :param int layer_synapses: The number of synapses incoming to
            the parent layer; used to scale the penalty; if None, it is
            assumed this will be provided later
        :param int model_synapses: The total number of synapses in the
            model; used to normalize the penalty; if None, it is assumed
            this will be provided later
        """

        self.l1_synapse_penalty = l1_synapse_penalty
        self.l2_decay = l2_decay
        self.layer_synapses = layer_synapses
        self.model_synapses = model_synapses

    def __call__(self, weights):
        """
        Computes the penalty.

        :param tensorflow.Tensor weights: A tensor of weights
        :return float: The value of the penalty
        """

        # Since we're using reduce_mean, multiply by the total number of
        # synapses incoming to the layer
        penalty = (self.l1_synapse_penalty
                   * self.layer_synapses
                   / self.model_synapses
                   * tf.reduce_mean(tf.abs(weights)))

        # Match the Keras L2 regularizer definition
        if self.l2_decay is not None:
            penalty += self.l2_decay * tf.reduce_sum(weights ** 2)

        return penalty

    def get_config(self):
        """
        Returns the configuration of the regularizer as Python
        dictionary.

        :return: The serialized configuration
        """

        return {
            "l1_synapse_penalty": self.l1_synapse_penalty,
            "l2_decay":           self.l2_decay,
            "layer_synapses":     self.layer_synapses,
            "model_synapses":     self.model_synapses
        }


class PostBatchNormOffset(Layer):
    """
    Layer which adds a fixed offset to the output of BatchNormalization.

    Technically, this can be used anywhere, but it makes the most sense
    after BatchNormalization where the activations have a known
    distribution.
    """

    def __init__(self, **kwargs):
        """
        PostBatchNormOffset constructor.

        The offset is initialized to zero.

        :param float offset: A scalar offset which this layer will add
            to its input
        """

        self.offset = None
        super().__init__(**kwargs)

    def build(self, input_shape):
        """
        Initializes the offset tensor.

        This is typically called behind the scenes when the layer is
        added to a Keras model.

        :param input_shape: The layer input shape
        """

        self.offset = self.add_weight(
            "offset", (input_shape[-1],), initializer="zeros", trainable=False)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        """
        Adds an offset to the input.

        :param tensorflow.Tensor inputs: The layer input
        """

        return inputs + self.offset


class SpikingCell(Layer):
    """
    Abstract class which implements common logic for spiking cells.
    """

    def __init__(
            self,
            track_spikes=False,
            enable_clamp=False,
            accumulate_only=False,
            reset_mechanism="subtract",
            t_refrac=0,
            buffer_dv=False,
            **kwargs):
        """
        SpikingCell constructor.

        :param bool track_spikes: Whether to track the number of spikes
            received by this layer
        :param bool enable_clamp: Whether to enable membrane potential
            clamping; this is useful to allow stabilization of input
            before firing
        :param bool accumulate_only: Whether, instead of simulating
            spiking logic, this layer should simply accumulate its input
            and return its current membrane potential at each timestep.
            This is intended to be used on the last layer.
        :param str reset_mechanism: The post-spike membrane reset
            mechanism; can be either "subtract" for reset by subtraction
            or "zero" for reset to zero
        :param int t_refrac: The duration of the post-spiking refractory
            period; during this period the membrane potential is frozen;
            this is similar to clamping, but the refractory period is
            neuron-specific and not layer-global
        :param bool buffer_dv: Whether input should be buffered for one
            timestep before being integrated by the cell
        """

        if reset_mechanism not in ("subtract", "zero"):
            raise ValueError('reset_mechanism must be "subtract" or "zero"')

        # The "fp_spikes" option is no longer available, but some saved
        # models may still use it
        # TODO: Remove this logic (update models by loading and
        #   re-saving)
        if "fp_spikes" in kwargs:
            del kwargs["fp_spikes"]

        self.track_spikes = track_spikes
        self.enable_clamp = enable_clamp
        self.accumulate_only = accumulate_only
        self.state_size = None
        self.reset_mechanism = reset_mechanism
        self.t_refrac = t_refrac
        self.buffer_dv = buffer_dv

        super().__init__(**kwargs)

    def build(self, input_shape):
        """
        Initializes weight tensors and computes the state size.

        This is typically called behind the scenes when the layer is
        added to a Keras model.

        :param input_shape: The layer input shape
        """

        self.state_size = [self._get_neurons_shape(input_shape)]
        if self.track_spikes:
            self.state_size.append(tf.TensorShape(()))
        if self.enable_clamp:
            self.state_size.append(tf.TensorShape(()))
        if self.t_refrac > 0:
            self.state_size.append(self._get_neurons_shape(input_shape))
        if self.buffer_dv:
            self.state_size.append(self._get_neurons_shape(input_shape))

        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            "track_spikes":    self.track_spikes,
            "enable_clamp":    self.enable_clamp,
            "accumulate_only": self.accumulate_only,
            "reset_mechanism": self.reset_mechanism,
            "t_refrac":        self.t_refrac,
            "buffer_dv":       self.buffer_dv,
        })
        return config

    # noinspection PyMethodOverriding
    def call(self, inputs, states, **kwargs):
        """
        Defines a single timestep of spiking cell logic.

        :param tensorflow.Tensor inputs: The cell inputs
        :param states: A list/tuple of cell states
        :return: The cell outputs and a list/tuple of updated cell
            states
        """

        dv = self._compute_dv(inputs)

        # The dv currently being integrated by the cell
        dv_integrate = self.get_dv_buffer(states) if self.buffer_dv else dv

        # Integration of input
        v = states[0] + self._filter_update(states, dv_integrate)

        # Spiking logic
        if self.accumulate_only:
            outputs = v
        else:
            outputs = tf.cast(v >= 1.0, self.dtype)
            if self.reset_mechanism == "subtract":
                v = v - outputs
            else:
                v = v * tf.cast(outputs == 0, v.dtype)

        new_states = [v]

        # Update optional state variables
        if self.track_spikes:
            # This counts all spikes sent by the upstream layer in this
            # timestep, not those received by this layer (these are
            # different if buffer_dv is True)
            new_states.append(
                self.get_spike_count(states) + self._count_spikes(inputs))
        if self.enable_clamp:
            new_states.append(self.get_clamp(states) - 1.0)
        if self.t_refrac > 0:
            refrac = self.get_refrac(states)
            new_states.append(
                (refrac + 1.0) * tf.cast(outputs == 0.0, refrac.dtype))
        if self.buffer_dv:
            # Buffer the dv for use on the next timestep
            new_states.append(dv)

        return outputs, new_states

    def get_spike_count(self, states):
        """
        Given a states array, returns the spike counter element.

        This is necessary because the length and contents of the states
        list vary based on the cell options.

        :param list states: A list of RNN states
        :return tensorflow.Tensor: The spike count state, or None if
            track_spikes is False
        """

        if self.track_spikes:
            return states[1]
        else:
            return None

    def get_clamp(self, states):
        """
        Given a states array, returns the clamp counter element.

        This is necessary because the length and contents of the states
        list vary based on the cell options.

        :param list states: A list of RNN states
        :return tensorflow.Tensor: The clamp counter state, or None if
            enable_clamp is False
        """

        if self.enable_clamp:
            return states[1 + int(self.track_spikes)]
        else:
            return None

    def get_refrac(self, states):
        """
        Given a states array, returns the refractory period element.

        This is necessary because the length and contents of the states
        list vary based on the cell options.

        :param list states: A list of RNN states
        :return tensorflow.Tensor: The refractory period element, or
            None if t_refrac is not positive
        """

        if self.t_refrac > 0:
            return states[1 + int(self.track_spikes) + int(self.enable_clamp)]
        else:
            return None

    def get_dv_buffer(self, states):
        """
        Given a states array, returns the input buffer element (used
        when buffer_dv is True).

        This is necessary because the length and contents of the states
        list vary based on the cell options.

        :param list states: A list of RNN states
        :return tensorflow.Tensor: The input buffer element, or None if
            buffer_dv is False
        """

        if self.buffer_dv:
            return states[
                1
                + int(self.track_spikes)
                + int(self.enable_clamp)
                + int(self.t_refrac > 0)]
        else:
            return None

    def _filter_update(self, states, dv):
        # Prevent input integration during clamp period
        if self.enable_clamp:
            clamp = self.get_clamp(states)
            shape = clamp.shape + (1,) * len(dv.shape[1:])
            dv = dv * tf.cast(tf.reshape(clamp, shape) <= 0.0, dv.dtype)

        # Prevent input integration during refractory period
        if self.t_refrac > 0:
            dv = dv * tf.cast(
                self.get_refrac(states) >= self.t_refrac, dv.dtype)

        return dv

    @abstractmethod
    def _compute_dv(self, inputs):
        pass

    @abstractmethod
    def _count_spikes(self, inputs):
        pass

    @abstractmethod
    def _get_neurons_shape(self, input_shape):
        pass


class SpikingConv2DCell(SpikingCell):
    """
    Defines a single timestep of logic for a 2D convolutional spiking
    layer.

    This is meant to be used in combination with a Keras RNN layer.
    """

    def __init__(
            self, filters, kernel_size,
            strides=(1, 1),
            padding="valid",
            use_bias=True,
            sparse_tracking=False,
            track_spikes=False,
            enable_clamp=False,
            accumulate_only=False,
            reset_mechanism="subtract",
            t_refrac=0,
            buffer_dv=False,
            **kwargs):
        """
        SpikingConv2DCell constructor.

        :param int filters: The number of convolutional channels
        :param kernel_size: The size of the convolutional kernel. Should
            be an int, 2-list, or 2-tuple. Follows the Keras Conv2D
            conventions.
        :param strides: The convolutional stride. Should be an int,
            2-list, or 2-tuple. Follows the Keras Conv2D conventions.
        :param string padding: Either "valid" or "same". Follows the
            Keras Conv2D conventions.
        :param bool use_bias: Whether a constant bias should be added
            to each channel
        :param bool sparse_tracking: Whether spikes should only be
            counted if they correspond to a nonzero synapse weight; has
            no effect if track_spikes is False
        :param bool track_spikes: Whether to track the number of spikes
            received by this layer
        :param bool enable_clamp: Whether to enable membrane potential
            clamping; this is useful to allow stabilization of input
            before firing
        :param bool accumulate_only: Whether, instead of simulating
            spiking logic, this layer should simply accumulate its input
            and return its current membrane potential at each timestep.
            This is intended to be used on the last layer.
        :param str reset_mechanism: The post-spike membrane reset
            mechanism; can be either "subtract" for reset by subtraction
            or "zero" for reset to zero
        :param int t_refrac: The duration of the post-spiking refractory
            period; during this period the membrane potential is frozen;
            this is similar to clamping, but the refractory period is
            neuron-specific and not layer-global
        :param bool buffer_dv: Whether input should be buffered for one
            timestep before being integrated by the cell
        """

        self.filters = filters
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = tuple(kernel_size)
        self.strides = strides
        self.padding = padding.upper()
        self.use_bias = use_bias
        self.sparse_tracking = sparse_tracking
        self.w = None
        self.b = None

        super().__init__(
            track_spikes=track_spikes,
            enable_clamp=enable_clamp,
            accumulate_only=accumulate_only,
            reset_mechanism=reset_mechanism,
            t_refrac=t_refrac,
            buffer_dv=buffer_dv,
            **kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(
            name="weight",
            shape=self.kernel_size + (input_shape[-1], self.filters),
            trainable=False,
            dtype=self.dtype)
        if self.use_bias:
            self.b = self.add_weight(
                name="bias",
                shape=(self.filters,),
                trainable=False,
                dtype=self.dtype)

        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters":         self.filters,
            "kernel_size":     self.kernel_size,
            "strides":         self.strides,
            "padding":         self.padding,
            "use_bias":        self.use_bias,
            "sparse_tracking": self.sparse_tracking,
        })
        return config

    def _compute_dv(self, inputs):
        input_dv = tf.nn.conv2d(inputs, self.w, self.strides, self.padding)
        if self.use_bias:
            return input_dv + self.b
        else:
            return input_dv

    def _count_spikes(self, inputs):
        if self.sparse_tracking:
            synapses = tf.cast(self.w != 0.0, self.dtype)
            spikes_in = tf.nn.conv2d(
                inputs, synapses, self.strides, self.padding)
        else:
            w_slim = tf.ones(self.w.shape[:-1] + (1,))
            spikes_in = self.filters * tf.nn.conv2d(
                inputs, w_slim, self.strides, self.padding)
        return tf.reduce_sum(spikes_in, axis=(1, 2, 3))

    def _get_neurons_shape(self, input_shape):
        if self.padding == "SAME":
            size_x = input_shape[1] // self.strides[0]
            size_y = input_shape[2] // self.strides[1]
        else:
            size_x = (input_shape[1] - self.kernel_size[0]) // self.strides[0] + 1
            size_y = (input_shape[2] - self.kernel_size[1]) // self.strides[1] + 1
        return tf.TensorShape((size_x, size_y, self.filters))


class SpikingDenseCell(SpikingCell):
    """
    Defines a single timestep of logic for a fully-connected spiking
    layer.

    This is meant to be used in combination with a Keras RNN layer.
    """

    def __init__(
            self, units,
            use_bias=True,
            sparse_tracking=False,
            track_spikes=False,
            enable_clamp=False,
            accumulate_only=False,
            reset_mechanism="subtract",
            t_refrac=0,
            buffer_dv=False,
            **kwargs):
        """
        SpikingDenseCell constructor.

        :param units: The number of fully-connected neurons in this
            layer
        :param use_bias: Whether to add a constant bias to each neuron
        :param bool sparse_tracking: Whether spikes should only be
            counted if they correspond to a nonzero synapse weight; has
            no effect if track_spikes is False
        :param bool track_spikes: Whether to track the number of spikes
            received by this layer
        :param bool enable_clamp: Whether to enable membrane potential
            clamping; this is useful to allow stabilization of input
            before firing
        :param bool accumulate_only: Whether, instead of simulating
            spiking logic, this layer should simply accumulate its input
            and return its current membrane potential at each timestep.
            This is intended to be used on the last layer.
        :param str reset_mechanism: The post-spike membrane reset
            mechanism; can be either "subtract" for reset by subtraction
            or "zero" for reset to zero
        :param int t_refrac: The duration of the post-spiking refractory
            period; during this period the membrane potential is frozen;
            this is similar to clamping, but the refractory period is
            neuron-specific and not layer-global
        :param bool buffer_dv: Whether input should be buffered for one
            timestep before being integrated by the cell
        """

        self.units = units
        self.use_bias = use_bias
        self.sparse_tracking = sparse_tracking
        self.w = None
        self.b = None

        super().__init__(
            track_spikes=track_spikes,
            enable_clamp=enable_clamp,
            accumulate_only=accumulate_only,
            reset_mechanism=reset_mechanism,
            t_refrac=t_refrac,
            buffer_dv=buffer_dv,
            **kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(
            name="weight",
            shape=(input_shape[1], self.units),
            trainable=False,
            dtype=self.dtype)
        if self.use_bias:
            self.b = self.add_weight(
                name="bias",
                shape=(self.units,),
                trainable=False,
                dtype=self.dtype)

        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            "units":           self.units,
            "use_bias":        self.use_bias,
            "sparse_tracking": self.sparse_tracking,
        })
        return config

    def _compute_dv(self, inputs):
        input_dv = inputs @ self.w
        if self.use_bias:
            return input_dv + self.b
        else:
            return input_dv

    def _count_spikes(self, inputs):
        if self.sparse_tracking:
            synapses = tf.cast(self.w != 0.0, self.dtype)
            spikes_in = inputs @ synapses
        else:
            spikes_in = self.units * inputs
        return tf.reduce_sum(spikes_in)

    def _get_neurons_shape(self, input_shape):
        return tf.TensorShape(self.units)


class SpikingDepthwiseConv2DCell(SpikingCell):
    """
    Defines a single timestep of logic for a 2D depthwise convolutional
    spiking layer.

    This is meant to be used in combination with a Keras RNN layer.
    """

    def __init__(
            self, kernel_size,
            strides=(1, 1),
            padding="valid",
            depth_multiplier=1,
            use_bias=True,
            sparse_tracking=False,
            track_spikes=False,
            enable_clamp=False,
            accumulate_only=False,
            reset_mechanism="subtract",
            t_refrac=0,
            buffer_dv=False,
            **kwargs):
        """
        SpikingDepthwiseConv2DCell constructor.

        :param kernel_size: The size of the convolutional kernel. Should
            be an int, 2-list, or 2-tuple. Follows the Keras Conv2D
            conventions.
        :param strides: The convolutional stride. Should be an int,
            2-list, or 2-tuple. Follows the Keras Conv2D conventions.
        :param string padding: Either "valid" or "same". Follows the
            Keras Conv2D conventions.
        :param int depth_multiplier: The number of output channels to
            be created for each input channel
        :param bool use_bias: Whether a constant bias should be added
            to each channel
        :param bool sparse_tracking: Whether spikes should only be
            counted if they correspond to a nonzero synapse weight; has
            no effect if track_spikes is False
        :param bool track_spikes: Whether to track the number of spikes
            received by this layer
        :param bool enable_clamp: Whether to enable membrane potential
            clamping; this is useful to allow stabilization of input
            before firing
        :param bool accumulate_only: Whether, instead of simulating
            spiking logic, this layer should simply accumulate its input
            and return its current membrane potential at each timestep.
            This is intended to be used on the last layer.
        :param str reset_mechanism: The post-spike membrane reset
            mechanism; can be either "subtract" for reset by subtraction
            or "zero" for reset to zero
        :param int t_refrac: The duration of the post-spiking refractory
            period; during this period the membrane potential is frozen;
            this is similar to clamping, but the refractory period is
            neuron-specific and not layer-global
        :param bool buffer_dv: Whether input should be buffered for one
            timestep before being integrated by the cell
        """

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = tuple(kernel_size)
        if len(strides) == 2:
            strides = (1,) + strides + (1,)
        self.strides = strides
        self.depth_multiplier = depth_multiplier
        self.padding = padding.upper()
        self.use_bias = use_bias
        self.sparse_tracking = sparse_tracking
        self.w = None
        self.b = None

        super().__init__(
            track_spikes=track_spikes,
            enable_clamp=enable_clamp,
            accumulate_only=accumulate_only,
            reset_mechanism=reset_mechanism,
            t_refrac=t_refrac,
            buffer_dv=buffer_dv,
            **kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(
            name="weight",
            shape=self.kernel_size + (input_shape[-1], self.depth_multiplier),
            trainable=False,
            dtype=self.dtype)
        if self.use_bias:
            self.b = self.add_weight(
                name="bias",
                shape=(input_shape[-1] * self.depth_multiplier,),
                trainable=False,
                dtype=self.dtype)

        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            "kernel_size":      self.kernel_size,
            "strides":          self.strides,
            "padding":          self.padding,
            "depth_multiplier": self.depth_multiplier,
            "use_bias":         self.use_bias,
            "sparse_tracking":  self.sparse_tracking,
        })
        return config

    def _compute_dv(self, inputs):
        input_dv = tf.nn.depthwise_conv2d(
            inputs, self.w, self.strides, self.padding)
        if self.use_bias:
            return input_dv + self.b
        else:
            return input_dv

    def _count_spikes(self, inputs):
        if self.sparse_tracking:
            synapses = tf.cast(self.w != 0.0, self.dtype)
            spikes_in = tf.nn.depthwise_conv2d(
                inputs, synapses, self.strides, self.padding)
        else:
            w_slim = tf.ones(self.w.shape[:-1] + (1,))
            spikes_in = self.depth_multiplier * tf.nn.depthwise_conv2d(
                inputs, w_slim, self.strides, self.padding)
        return tf.reduce_sum(spikes_in, axis=(1, 2, 3))

    def _get_neurons_shape(self, input_shape):
        if self.padding == "SAME":
            size_x = input_shape[1] // self.strides[1]
            size_y = input_shape[2] // self.strides[2]
        else:
            size_x = (input_shape[1] - self.kernel_size[0]) // self.strides[1] + 1
            size_y = (input_shape[2] - self.kernel_size[1]) // self.strides[2] + 1
        return tf.TensorShape((size_x, size_y, input_shape[-1] * self.depth_multiplier))


class SpikingInputCell(SpikingCell):
    """
    Defines a single timestep for a spiking input layer.

    This is meant to be used in combination with a Keras RNN layer.
    """

    def __init__(
            self,
            track_spikes=False,
            enable_clamp=False,
            accumulate_only=False,
            reset_mechanism="subtract",
            t_refrac=0,
            buffer_dv=False,
            **kwargs):
        """
        SpikingInputCell constructor.

        :param bool track_spikes: Whether to track the number of spikes
            received by this layer
        :param bool enable_clamp: Whether to enable membrane potential
            clamping; this is useful to allow stabilization of input
            before firing
        :param bool accumulate_only: Whether, instead of simulating
            spiking logic, this layer should simply accumulate its input
            and return its current membrane potential at each timestep.
            This is intended to be used on the last layer.
        :param str reset_mechanism: The post-spike membrane reset
            mechanism; can be either "subtract" for reset by subtraction
            or "zero" for reset to zero
        :param int t_refrac: The duration of the post-spiking refractory
            period; during this period the membrane potential is frozen;
            this is similar to clamping, but the refractory period is
            neuron-specific and not layer-global
        :param bool buffer_dv: Whether input should be buffered for one
            timestep before being integrated by the cell
        """

        super().__init__(
            track_spikes=track_spikes,
            enable_clamp=enable_clamp,
            accumulate_only=accumulate_only,
            reset_mechanism=reset_mechanism,
            t_refrac=t_refrac,
            buffer_dv=buffer_dv,
            **kwargs)

    def _compute_dv(self, inputs):
        return inputs

    def _count_spikes(self, inputs):
        return tf.reduce_sum(inputs)

    def _get_neurons_shape(self, input_shape):
        return tf.TensorShape(input_shape[1:])
