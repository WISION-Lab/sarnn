import pickle

from sarnn.metrics import *

DEFAULT_CELL_PARAMS = {
    "v_rest":     0.0,
    "cm":         1.0,
    "tau_m":      np.inf,
    "tau_refrac": 1.0,
    "tau_syn_E":  0.0,
    "tau_syn_I":  0.0,
    "v_thresh":   1.0,
    "v_reset":    0.0,
}

# Beyond an exponent of about +/-4 we start seeing overflow/underflow
SPINNAKER_CELL_PARAMS = {
    "v_rest":     0.0,
    "cm":         1.0,
    "tau_m":      1e4,
    "tau_refrac": 1.0,
    "tau_syn_E":  1e-4,
    "tau_syn_I":  1e-4,
    "v_thresh":   1.0,
    "v_reset":    0.0,
}

# These exact values were determined empirically (i.e. magic numbers);
# the weight scale varies inversely with the synaptic time constant
SPINNAKER_WEIGHT_SCALE = 1.09227e4
SPINNAKER_BIAS_SCALE = 1.08864

# Used to warn the user if the fan-in/fan-out is high on SpiNNaker; the
# precise value is arbitrary
SPINNAKER_FAN_THRESH = 100


# Have observed some weird numerical issues when using the NEURON
# backend (membrane potentials accumulate tiny numerical errors in the
# absence of input)
class PyNNModel:
    def __init__(self, population_sizes, neuron_attributes, syn_excitatory, syn_inhibitory):
        self.population_sizes = population_sizes
        self.neuron_attributes = neuron_attributes
        self.syn_excitatory = syn_excitatory
        self.syn_inhibitory = syn_inhibitory
        self.sim = None

    @staticmethod
    def from_keras(model, sparse_synapses=True):
        population_sizes = []
        neuron_attributes = []
        syn_excitatory = []
        syn_inhibitory = []

        for layer in model.layers:
            # We don't use sarnn.utils.layer_typename because we want
            # this module to be independent of sarnn.utils
            layer_type = layer.__class__.__name__

            if layer_type == "RNN":
                population_sizes.append(np.prod(layer.output_shape[2:]))
                cell_type = layer.cell.__class__.__name__

                if cell_type == "SpikingInputCell":
                    # Continue so we don't try to set i_offset and
                    # v_thresh for the input SpikeSourceArray layer
                    continue
                elif cell_type == "SpikingConv2DCell":
                    synapses = _conv2d_synapses(layer, sparse_synapses)
                elif cell_type == "SpikingDenseCell":
                    synapses = _dense_synapses(layer, sparse_synapses)
                elif cell_type == "SpikingDepthwiseConv2DCell":
                    synapses = _depthwise_conv2d_synapses(layer, sparse_synapses)
                else:
                    raise ValueError('invalid RNN cell type "{}"'.format(cell_type))
                syn_excitatory.append(synapses[0])
                syn_inhibitory.append(synapses[1])

                attributes = {
                    "v_thresh_inf": layer.cell.accumulate_only
                }
                if layer.cell.use_bias:
                    attributes["i_offset"] = _biases(layer)
                neuron_attributes.append(attributes)

            elif layer_type == "TimeDistributed":
                inner_type = layer.layer.__class__.__name__
                if inner_type != "Flatten":
                    raise ValueError('invalid TimeDistributed layer "{}"'.format(inner_type))
            else:
                raise ValueError('invalid layer type "{}"'.format(layer_type))

        return PyNNModel(population_sizes, neuron_attributes, syn_excitatory, syn_inhibitory)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as pickle_file:
            return PyNNModel(**pickle.load(pickle_file))

    def build(self, sim, cell_typename="IF_curr_exp", cell_params=None):
        self.sim = sim
        self.sim.setup(timestep=1.0, min_delay=1.0, max_delay=1.0)
        is_spinnaker = sim.__name__ == "pyNN.spiNNaker"

        if cell_params is None:
            cell_params = SPINNAKER_CELL_PARAMS if is_spinnaker else DEFAULT_CELL_PARAMS
        cell_type = getattr(sim, cell_typename)(**cell_params)

        if is_spinnaker:
            weight_scale, bias_scale = SPINNAKER_WEIGHT_SCALE, SPINNAKER_BIAS_SCALE
        else:
            weight_scale, bias_scale = 1.0, 1.0

        self.populations = [
            self.sim.Population(self.population_sizes[0], self.sim.SpikeSourceArray())]
        self.connections = []

        for i in range(1, len(self.population_sizes)):
            # Build a population of neurons
            self.populations.append(self.sim.Population(self.population_sizes[i], cell_type))

            if self.neuron_attributes[i - 1]["v_thresh_inf"]:
                self.populations[-1].set(v_thresh=1e4 if is_spinnaker else np.inf)
            if "i_offset" in self.neuron_attributes[i - 1]:
                self.populations[-1].set(
                    i_offset=self.neuron_attributes[i - 1]["i_offset"] * bias_scale)

            # PyCharm messes up the formatting on the continuation line
            # @formatter:off
            for synapses, receptor_type in [
                    (self.syn_excitatory[i - 1], "excitatory"),
                    (self.syn_inhibitory[i - 1], "inhibitory")]:
                # @formatter:on

                if is_spinnaker:
                    if len(synapses) / self.population_sizes[i - 1] > SPINNAKER_FAN_THRESH:
                        print("Warning: high fan-out ({} synapses) "
                              "detected from layer {}. This may reduce "
                              "simulation accuracy on SpiNNaker."
                              .format(receptor_type, i - 1))
                    if len(synapses) / self.population_sizes[i] > SPINNAKER_FAN_THRESH:
                        print("Warning: high fan-in ({} synapses) "
                              "detected into layer {}. This may reduce "
                              "simulation accuracy on SpiNNaker."
                              .format(receptor_type, i))

                # There's got to be a faster way to do this...
                scaled = []
                for synapse in synapses:
                    scaled.append((synapse[0], synapse[1], synapse[2] * weight_scale))
                if len(scaled) == 0:
                    continue
                self.connections.append(self.sim.Projection(
                    self.populations[-2],
                    self.populations[-1],
                    self.sim.FromListConnector(scaled, column_names=["weight"]),
                    synapse_type=self.sim.StaticSynapse(delay=1.0),
                    receptor_type=receptor_type))

    def evaluate(
            self, x, y,
            acc_ann=None,
            threshold=None,
            v_initial=0.5,
            n_time_steps=100):

        results = self.simulate(
            x, y,
            v_initial=v_initial,
            n_time_steps=n_time_steps,
            track_spikes=True,
            transparent=False,
            return_outputs=False,
            return_predictions=False)

        acc_snn = results["accuracy"]
        power = np.mean(results["spike_rates"], axis=0)
        output_dict = {
            "peak_accuracy": peak_accuracy(results["accuracy"])
        }
        if acc_ann is not None:
            output_dict["time_above_curve"] = time_above_curve(acc_snn, acc_ann)
            output_dict["power_above_curve"] = power_above_curve(acc_snn, acc_ann, power)
        if threshold is not None:
            output_dict["time_to_accuracy"] = time_to_accuracy(acc_snn, threshold)
            output_dict["power_to_accuracy"] = power_to_accuracy(acc_snn, threshold, power)
        return output_dict

    def save(self, filename):
        with open(filename, "wb") as pickle_file:
            write_dict = {
                "population_sizes":  self.population_sizes,
                "neuron_attributes": self.neuron_attributes,
                "syn_excitatory":    self.syn_excitatory,
                "syn_inhibitory":    self.syn_inhibitory,
            }

            # HBP systems don't support protocol > 2
            pickle.dump(write_dict, pickle_file, protocol=2)

    def simulate(
            self, x, y,
            v_initial=0.5,
            n_time_steps=100,
            track_spikes=True,
            transparent=False,
            return_outputs=False,
            return_predictions=False):
        # Set up neuron populations
        self.populations[-1].record("v")
        if transparent or track_spikes:
            for p in self.populations:
                p.record("spikes")
        for p in self.populations[1:]:
            p.initialize(v=v_initial)

        # Run the simulation for each item in the dataset
        for i in range(x.shape[0]):
            self.populations[0].set(
                spike_times=self._spike_times(x[i].flatten(), v_initial, n_time_steps))

            # This may result in one extra time step being simulated
            # with some backends; this is addressed by indexing to
            # :n_time_steps below
            self.sim.run(float(n_time_steps))
            self.sim.reset()

        # Allocate data structures for results
        n_correct = np.zeros(n_time_steps, dtype=np.int64)
        spike_rates = None
        if track_spikes:
            spike_rates = np.zeros((x.shape[0], n_time_steps))
        outputs = None
        if return_outputs or transparent:
            outputs = []
            if transparent:
                for p in self.populations[:-1]:
                    outputs.append(np.zeros((x.shape[0], n_time_steps, p.size), dtype=np.bool))
            outputs.append(np.empty((x.shape[0], n_time_steps, self.populations[-1].size)))
        predictions = None
        if return_predictions:
            predictions = np.empty((x.shape[0], n_time_steps), dtype=np.int64)

        # Extract data and save results
        data = [p.get_data() for p in self.populations]
        for i in range(x.shape[0]):
            predictions_i = np.argmax(data[-1].segments[i].analogsignals[0], axis=-1)
            n_correct += predictions_i[:n_time_steps] == np.argmax(y[i])
            for j, data_j in enumerate(data[:-1]):
                for k, spikes in enumerate(data_j.segments[i].spiketrains):
                    # NumPy indexing blows my mind...
                    spikes_np = np.array(spikes).astype(np.int64)
                    spikes_np = spikes_np[spikes_np < n_time_steps]
                    if track_spikes:
                        spike_rates[i][spikes_np] += 1
                    if transparent:
                        outputs[j][i, :, k][spikes_np] = True
            if return_outputs or transparent:
                outputs[-1][i] = data[-1].segments[i].analogsignals[0][:n_time_steps]
            if return_predictions:
                predictions[i] = predictions_i[:n_time_steps]

        # Reset tracking variables
        for p in self.populations[1:]:
            p.record([])

        # Create and return the output dictionary
        output_dict = {
            "accuracy": n_correct / float(x.shape[0])
        }
        if track_spikes:
            output_dict["spike_rates"] = spike_rates
        if return_outputs or transparent:
            output_dict["outputs"] = outputs
        if return_predictions:
            output_dict["predictions"] = predictions
        return output_dict

    @staticmethod
    def _spike_times(x_i, v_initial, n_time_steps):
        spike_times = []
        for analog_in in x_i:
            spikes = []
            v = v_initial
            for t in range(n_time_steps):
                v += analog_in
                if v >= 1.0:
                    spikes.append(float(t))
                    v = v - 1.0
            spike_times.append(spikes)
        return spike_times


# Abbreviations used below:
#   k = kernel
#   s = shape
#   n = input
#   o = output
#   p = padding
#   t = strides
#   i = index along spatial dimension 0 (y)
#   j = index along spatial dimension 1 (x)
#   c = index along channel dimension
#   m = index along channel multiplier
#   w = weight
#   d = delta


def _biases(layer):
    return np.broadcast_to(layer.get_weights()[1], layer.output_shape[2:]).flatten()


def _conv2d_spatial_repeat(s_n, s_o, p, t, i_k, j_k, c_n, c_o, w):
    synapses = []

    # Loop over the output positions
    for i_o in range(s_o[0]):
        i_n = -p[0] + i_o * t[0] + i_k

        # These offsets are fixed over the inner for loop
        d_n = i_n * s_n[1] * s_n[2] + c_n
        d_o = i_o * s_o[1] * s_o[2] + c_o

        for j_o in range(s_o[1]):
            j_n = -p[1] + j_o * t[1] + j_k
            if i_n < 0 or j_n < 0 or i_n >= s_n[0] or j_n >= s_n[1]:
                continue
            synapses.append((d_n + j_n * s_n[2], d_o + j_o * s_o[2], w))
    return synapses


def _conv2d_synapses(layer, sparse_synapses):
    k, s_k, s_n, s_o, p, t = _conv2d_vars(layer)
    syn_excitatory = []
    syn_inhibitory = []

    # Loop over the kernel positions
    for i_k in range(s_k[0]):
        for j_k in range(s_k[1]):

            # Loop over the kernel channels
            for c_n in range(s_k[2]):
                for c_o in range(s_k[3]):
                    w = k[i_k, j_k, c_n, c_o]
                    if sparse_synapses and w == 0.0:
                        continue
                    synapses = _conv2d_spatial_repeat(s_n, s_o, p, t, i_k, j_k, c_n, c_o, w)
                    if w >= 0.0:
                        syn_excitatory += synapses
                    else:
                        syn_inhibitory += synapses

    return syn_excitatory, syn_inhibitory


def _dense_synapses(layer, sparse_synapses):
    k = layer.get_weights()[0]
    syn_excitatory = []
    syn_inhibitory = []

    for n in range(np.prod(layer.input_shape[2:])):
        for o in range(layer.output_shape[2]):
            w = k[n, o]
            if sparse_synapses and w == 0.0:
                continue
            synapse = (n, o, w)
            if w >= 0.0:
                syn_excitatory.append(synapse)
            else:
                syn_inhibitory.append(synapse)

    return syn_excitatory, syn_inhibitory


def _depthwise_conv2d_synapses(layer, sparse_synapses):
    k, s_k, s_n, s_o, p, t = _conv2d_vars(layer)
    syn_excitatory = []
    syn_inhibitory = []

    # Loop over the kernel positions
    for i_k in range(s_k[0]):
        for j_k in range(s_k[1]):

            # Loop over the input channels
            for c_n in range(s_k[2]):

                # Loop over the depth multiplier
                for m in range(s_k[3]):
                    w = k[i_k, j_k, c_n, m]
                    if sparse_synapses and w == 0.0:
                        continue
                    c_o = c_n * s_k[3] + m
                    synapses = _conv2d_spatial_repeat(s_n, s_o, p, t, i_k, j_k, c_n, c_o, w)
                    if w >= 0.0:
                        syn_excitatory += synapses
                    else:
                        syn_inhibitory += synapses

    return syn_excitatory, syn_inhibitory


def _conv2d_vars(layer):
    k = layer.get_weights()[0]
    s_k = k.shape
    s_n = layer.input_shape[2:]
    s_o = layer.output_shape[2:]
    if layer.cell.padding.lower() == "same":
        p = (int(np.floor(s_k[0] / 2)), int(np.floor(s_k[1] / 2)))
    else:
        p = (0, 0)
    t = tuple(layer.cell.strides)

    # Throw out stride along the batch/channel dimensions
    if len(t) == 4:
        t = t[1:3]

    return k, s_k, s_n, s_o, p, t
