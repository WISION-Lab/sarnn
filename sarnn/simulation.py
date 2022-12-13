import tensorflow as tf
from mpi4py import MPI
from tensorflow.keras import Model

from sarnn.metrics import *
from sarnn.utils import layer_typename


# Abbreviations used below:
#   b = batch
#   c = time chunk
#   i = inputs
#   o = outputs
#   p = predictions
#   r = spike rates
#   s = start
#   e = end
#   f = first
#   l = last
#   j = index


def evaluate(
        snn, x, y,
        acc_ann=None,
        threshold=None,
        v_initial=0.5,
        n_time_chunks=1,
        poisson=False,
        decay=None,
        clamp_rate=0,
        n_items=None,
        mask_class=None,
        input_repeat=1,
        input_squash=1):
    """
    Evaluates the performance of the SNN on some data.

    If this is called in an MPI context, only rank 0 will return a
    results dictionary. Other ranks will return None, acting as workers.

    :param tensorflow.keras.Model snn: The SNN to evaluate
    :param x: Input data for the evaluation; either a numpy.ndarray or a
        tf.data.Dataset with no batch dimension
    :param y: Correct one-hot encoded output labels; either a
        numpy.ndarray or a tf.data.Dataset with no batch dimension
    :param acc_ann: If not None, this will be used to compute the TAC
        and PAC or R-TAC and R-PAC metrics (see metrics module for
        details)
    :param float threshold: If not None, this will be used to compute
        the TTA and PTA metrics (see metrics module for details)
    :param v_initial: Either a single float value or a list of initial
        neuron membrane potentials; see utils.v_initial_template for
        details on the list shape
    :param int n_time_chunks: The number of time chunks to simulate the
        model (the time chunk size is specified in conversion.build_snn)
    :param bool poisson: Whether to convert input to binary Poisson
        spike trains
    :param float decay: Exponential decay rate to apply to model outputs
        when determining predictions (helps minimize the impact of bad
        initial outputs)
    :param int clamp_rate: This value, when multiplied by the depth of a
        layer, gives the number of time steps each layer waits for its
        input to stabilize before beginning membrane potential updates
    :param int n_items: The number of data items over which the
        simulation should be performed; this is required if x and y are
        tf.data.Dataset
    :param int mask_class: If not None, items with this ground truth
        label are ignored when computing accuracy
    :param int input_repeat: The number of SNN time steps for which each
        input frame should be repeated (this only makes sense with
        poisson input)
    :param int input_squash: The number of SNN time steps over which
        each input frame should be averaged (this only makes sense with
        poisson input)
    :return dict: A dictionary containing summarized results
        ("accuracy", "latency", "power_consumption", "relative_latency")
    """

    results = simulate(
        snn, x, y,
        v_initial=v_initial,
        n_time_chunks=n_time_chunks,
        poisson=poisson,
        decay=decay,
        clamp_rate=clamp_rate,
        n_items=n_items,
        mask_class=mask_class,
        input_repeat=input_repeat,
        input_squash=input_squash,
        transparent=False,
        return_outputs=False,
        return_inputs=False,
        return_predictions=False)

    if results is None:
        return None

    acc_snn = results["accuracy"]
    output_dict = {
        "peak_accuracy": peak_accuracy(acc_snn)
    }
    if acc_ann is not None:
        output_dict["time_above_curve"] = time_above_curve(acc_snn, acc_ann)
    if threshold is not None:
        output_dict["time_to_accuracy"] = time_to_accuracy(acc_snn, threshold)
    if "spike_rates" in results:
        c_size = snn.input_shape[1]
        power = np.repeat(
            np.mean(results["spike_rates"], axis=0), c_size) / c_size
        if acc_ann is not None:
            output_dict["power_above_curve"] = power_above_curve(
                acc_snn, acc_ann, power)
        if threshold is not None:
            output_dict["power_to_accuracy"] = power_to_accuracy(
                acc_snn, threshold, power)

    return output_dict


def simulate(
        snn, x, y,
        v_initial=0.5,
        n_time_chunks=1,
        poisson=False,
        decay=None,
        clamp_rate=0,
        n_items=None,
        mask_class=None,
        input_repeat=1,
        input_squash=1,
        transparent=False,
        return_outputs=False,
        return_inputs=False,
        return_predictions=False):
    """
    Simulates SNN inference on the provided data.

    Takes care of chunking data into manageable sizes, counting spikes
    between neurons, computing accuracy, etc.

    In order to count spikes, track_spikes=True should have been
    specified at conversion.

    If this is called in an MPI context, only rank 0 will return a
    results dictionary. Other ranks will return None, acting as workers.

    :param tensorflow.keras.Model snn: The SNN to simulate
    :param x: Input data for the simulation; either a numpy.ndarray or a
        tf.data.Dataset with no batch dimension
    :param y: Correct one-hot encoded outputs labels; either a
        numpy.ndarray or a tf.data.Dataset with no batch dimension
    :param list v_initial: The list of initial neuron membrane
        potentials; see utils.v_initial_template for details on the
        shape
    :param int n_time_chunks: The number of time chunks to simulate the
        model (the time chunk size is specified in conversion.build_snn)
    :param bool poisson: Whether to convert input to binary Poisson
        spike trains
    :param float decay: Exponential decay rate to apply to model outputs
        when determining predictions (helps minimize the impact of bad
        initial outputs)
    :param int clamp_rate: This value, when multiplied by the depth of a
        layer, gives the number of time steps each layer waits for its
        input to stabilize before beginning membrane potential updates
    :param int n_items: The number of data items over which the
        simulation should be performed; this is required if x and y are
        tf.data.Dataset
    :param int mask_class: If not None, items with this ground truth
        label are ignored when computing accuracy
    :param int input_repeat: The number of SNN time steps for which each
        input frame should be repeated (this only makes sense with
        poisson input)
    :param int input_squash: The number of SNN time steps over which
        each input frame should be averaged (this only makes sense with
        poisson input)
    :param bool transparent: Whether internal model activations should
        be stored and returned (this is only advisable for small
        models); implies return_outputs=True
    :param bool return_outputs: Whether to include raw model outputs
        (i.e. spike trains) in the output dictionary as "outputs"
    :param bool return_inputs: Whether to include model inputs in the
        output dictionary as "inputs" (this really only makes sense when
        poisson=True)
    :param bool return_predictions: Whether to include prediction time
        series in the output dictionary as "predictions"
    :return dict: A dictionary containing summarized results
        ("accuracy", "outputs", "inputs", "predictions", and
        "spike_rates")
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Decide how to split work over batches, time chunks, and MPI ranks
    if n_items is None:
        n_items = x.shape[0]
    b_size = snn.input_shape[0]
    n_b = _n_tiles(n_items, b_size)
    b_f, b_l, _ = _tile_bounds(rank, n_b, _n_tiles(n_b, comm.Get_size()))
    offset = b_f * b_size
    n_local = max(min(n_items, b_l * b_size) - offset, 0)
    n_c = n_time_chunks
    c_size = snn.input_shape[1]
    n_t = n_c * c_size

    # Initialize an iterator over tf.data.Dataset
    if isinstance(x, tf.data.Dataset):
        assert isinstance(y, tf.data.Dataset)
        x = x.skip(offset).as_numpy_iterator()
        y = y.skip(offset).as_numpy_iterator()

    # Allocate data structures for results
    if transparent:
        snn = _make_transparent(snn)
    o = None
    if transparent or return_outputs:
        o = _alloc_outputs(snn, n_local, n_t)
    i = None
    if return_inputs:
        i = np.empty((n_local, n_t) + snn.input_shape[2:], dtype=snn.dtype)
    p = None
    o_shape = snn.output_shape
    if isinstance(o_shape, list):
        o_shape = o_shape[-1]
    if return_predictions:
        p = np.zeros((n_local, n_t) + o_shape[2:-1], dtype=np.int64)
    n_correct = np.zeros(n_t, dtype=np.int64)
    n_no_mask = np.zeros(1, dtype=np.int64)
    track_spikes = _spike_tracking_enabled(snn)
    r = None
    if track_spikes:
        r = np.zeros((n_local, n_c))
    b_o = np.empty((b_size, n_t) + o_shape[2:], dtype=snn.dtype)

    # Iteration over this worker's data batches
    for b_j in range(b_f, b_l):
        b_s, b_e, b_size_j = _tile_bounds(b_j, n_items, b_size)

        # Prepare elements of the dataset corresponding to this batch
        b_x = _extract_batch(x, b_s, b_e, b_size)
        b_y = _extract_batch(y, b_s, b_e, b_size)
        b_i = _expand_temporally(
            b_x, b_size_j, n_t, poisson, input_repeat, input_squash, snn.dtype)

        # Each rank only allocates room for local results
        b_s -= offset
        b_e -= offset

        if return_inputs:
            i[b_s:b_e] = b_i[:b_size_j]

        # Reset cell states between batches
        _reset_v_membranes(snn, v_initial)
        _reset_clamps(snn, clamp_rate)
        _reset_refracs(snn)
        _reset_dv_buffers(snn)

        # Iteration over time chunks in this batch
        for c_j in range(n_c):
            c_s, c_e, _ = _tile_bounds(c_j, n_t, c_size)

            # Reset spike counts between time chunks
            _reset_spike_counts(snn)

            # This is where the real work happens
            c_o = snn.predict(b_i[:, c_s:c_e], batch_size=b_size)

            # Save results
            c_o = c_o if isinstance(c_o, list) else [c_o]
            b_o[:, c_s:c_e] = c_o[-1]
            if transparent or return_outputs:
                for j, output in enumerate(c_o):
                    o[j][b_s:b_e, c_s:c_e] = output[:b_size_j]
            if track_spikes:
                r[b_s:b_e, c_j] = _spike_rates(
                    snn, c_o, b_size_j, not snn.layers[-1].cell.accumulate_only)

        # Compute predictions on the batch
        b_p = _b_predictions(b_o, b_size_j, decay)
        if return_predictions:
            p[b_s:b_e] = b_p

        # Update values which will be used to compute accuracy
        y_sparse = np.argmax(b_y[:b_size_j], axis=-1)
        n_correct += _count_correct(b_p, y_sparse)
        n_no_mask += np.count_nonzero(y_sparse != mask_class)

    # Gather results from MPI workers (if MPI is being used)
    n_correct = _mpi_gather_arrays(n_correct)
    n_no_mask = _mpi_gather_arrays(n_no_mask)
    if rank == 0:
        n_correct = np.sum(np.reshape(n_correct, (-1, n_t)), axis=0)
        n_no_mask = np.sum(np.reshape(n_no_mask, (-1, 1)), axis=0)
    if return_outputs or transparent:
        o = _mpi_gather_outputs(o)
    if return_inputs:
        i = _mpi_gather_arrays(i)
    if return_predictions:
        p = _mpi_gather_arrays(p)
    if track_spikes:
        r = _mpi_gather_arrays(r)
    if rank != 0:
        return None

    # Create and return the output dictionary (master rank only)
    output_dict = {
        "accuracy": n_correct / n_no_mask
    }
    if return_outputs or transparent:
        output_dict["outputs"] = o
    if return_inputs:
        output_dict["inputs"] = i
    if return_predictions:
        output_dict["predictions"] = p
    if track_spikes:
        output_dict["spike_rates"] = r
    return output_dict


def _alloc_outputs(snn, n_local, n_t):
    o = []
    for output in snn.outputs:
        o.append(np.zeros((n_local, n_t) + output.shape[2:]))
    return o


def _b_predictions(b_o, b_size_j, decay):
    if decay is None:
        smooth = np.cumsum(b_o[:b_size_j], axis=1)
    else:
        smooth = np.empty_like(b_o[:b_size_j])
        smooth[:, 0] = (1.0 - decay) * b_o[:b_size_j, 0]
        for t in range(1, smooth.shape[1]):
            smooth[:, t] = decay * smooth[:, t - 1] + (1.0 - decay) * b_o[:b_size_j, t]
    return np.argmax(smooth, axis=-1)


def _binary_poisson(x):
    return np.random.random(x.shape) > np.exp(-x)


def _count_correct(p, y_sparse):
    y_sparse = np.expand_dims(y_sparse, axis=1)
    y_sparse = np.repeat(y_sparse, p.shape[1], axis=1)
    y_sparse = np.reshape(y_sparse, y_sparse.shape[:2] + (-1,))
    p = np.reshape(p, p.shape[:2] + (-1,))
    return np.count_nonzero(p == y_sparse, axis=(0, -1))


def _expand_temporally(b_x, b_size_j, n_t, poisson, input_repeat, input_squash, dtype):
    tmp = np.expand_dims(b_x, axis=1)
    tmp = np.repeat(tmp, int(np.ceil(n_t * input_squash / input_repeat)), axis=1)
    if poisson:
        tmp[:b_size_j] = _binary_poisson(tmp[:b_size_j])
    if input_repeat != 1:
        tmp = np.repeat(tmp, input_repeat, axis=1)
    if input_squash != 1:
        b_i = np.empty((tmp.shape[0], n_t) + tmp.shape[2:], dtype=dtype)
        for t in range(n_t):
            b_i[:, t] = np.mean(tmp[:, input_squash * t:input_squash * (t + 1)], axis=1)
    else:
        b_i = tmp
    return b_i


def _extract_batch(data, b_s, b_e, b_size):
    if isinstance(data, np.ndarray):
        b_i = np.empty((b_size,) + data.shape[1:], dtype=data.dtype)
        b_i[:b_e - b_s] = data[b_s:b_e]
    else:
        b_i = None
        for j in range(b_e - b_s):
            item = next(data)
            if b_i is None:
                b_i = np.empty((b_size,) + item.shape, dtype=item.dtype)
            b_i[j] = item
    return b_i


def _make_transparent(snn):
    o = []
    for layer in snn.layers:
        if layer_typename(layer) == "RNN":
            o.append(layer.output)
    return Model(inputs=snn.inputs, outputs=o)


def _mpi_gather_arrays(array):
    comm = MPI.COMM_WORLD
    if comm.Get_size() == 1:
        return array
    counts = comm.gather(array.shape[0], root=0)
    if comm.Get_rank() == 0:
        counts = np.array(counts)
        shape = (np.sum(counts),) + array.shape[1:]
        gathered = np.empty(shape, dtype=array.dtype)
        comm.Gatherv(array, (gathered, np.prod(shape[1:]) * counts), root=0)
        return gathered
    else:
        comm.Gatherv(array, None, root=0)
        return None


def _mpi_gather_outputs(o):
    comm = MPI.COMM_WORLD
    if comm.Get_size() == 1:
        return o
    if comm.Get_rank() == 0:
        gathered = []
        for output in o:
            gathered.append(_mpi_gather_arrays(output))
        return gathered
    else:
        for output in o:
            _mpi_gather_arrays(output)
        return None


def _n_tiles(n_items, tile_size):
    return int(np.ceil(n_items / tile_size))


def _reset_clamps(snn, clamp_rate):
    t_clamp = 0
    for layer in snn.layers:
        if layer_typename(layer) == "RNN":
            t_clamp += clamp_rate
            if layer.cell.enable_clamp:
                state = layer.cell.get_clamp(layer.states)
                state.assign(tf.fill(
                    state.shape, tf.constant(t_clamp, dtype=state.dtype)))


def _reset_dv_buffers(snn):
    for layer in snn.layers:
        if layer_typename(layer) == "RNN" and layer.cell.buffer_dv:
            state = layer.cell.get_dv_buffer(layer.states)
            state.assign(tf.zeros_like(state))


def _reset_spike_counts(snn):
    for layer in snn.layers:
        if layer_typename(layer) == "RNN" and layer.cell.track_spikes:
            state = layer.cell.get_spike_count(layer.states)
            state.assign(tf.zeros_like(state))


def _reset_refracs(snn):
    for layer in snn.layers:
        if layer_typename(layer) == "RNN" and layer.cell.t_refrac > 0:
            state = layer.cell.get_refrac(layer.states)
            state.assign(tf.fill(
                state.shape,
                tf.constant(layer.cell.t_refrac, dtype=state.dtype)))


def _reset_v_membranes(snn, v_initial):
    if isinstance(v_initial, float):
        v_initial = [v_initial]
    j = 0
    for layer in snn.layers:
        if layer_typename(layer) == "RNN":
            if len(v_initial) > 1:
                layer_v_initial = v_initial[j]
            else:
                layer_v_initial = v_initial[0]
            state = layer.states[0]
            state.assign(layer_v_initial * tf.ones_like(state))
            j += 1


def _spike_rates(snn, c_o, b_size_j, count_output):
    spike_rates = np.zeros(b_size_j)
    c_size = c_o[0].shape[1]

    # Count spikes due to output
    if count_output:
        for output in c_o:
            flat = np.reshape(output[:b_size_j], (b_size_j, -1))
            spike_rates += np.sum(flat, axis=-1) / c_size

    # Count internal spikes
    first_rnn = True
    for layer in snn.layers:
        if layer_typename(layer) == "RNN":
            if layer.cell.track_spikes and not first_rnn:
                # It is more efficient to slice and divide in the NumPy/CPU
                # space (based on profiling experiments)
                spike_rates[:b_size_j] += layer.states[1].numpy()[:b_size_j] / c_size
            first_rnn = False

    return spike_rates


def _spike_tracking_enabled(snn):
    for layer in snn.layers:
        if layer_typename(layer) == "RNN" and layer.cell.track_spikes:
            return True
    return False


def _tile_bounds(j, n_items, tile_size):
    n_tiles = _n_tiles(n_items, tile_size)
    if j < n_tiles:
        tile_s = j * tile_size
    else:
        tile_s = n_items
    if j < n_tiles - 1:
        tile_e = (j + 1) * tile_size
    else:
        tile_e = n_items
    tile_size_j = tile_e - tile_s
    return tile_s, tile_e, tile_size_j
