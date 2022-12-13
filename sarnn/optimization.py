import os
import os.path as path
import pickle

import mpi4py
import nlopt
import numpy as np

from sarnn.simulation import evaluate
from sarnn.utils import scale, scaling_template, v_initial_template

# Maps string algorithm names to nlopt constants
ALGORITHMS = {
    "COBYLA":      nlopt.LN_COBYLA,
    "BOBYQA":      nlopt.LN_BOBYQA,
    "NEWUOA":      nlopt.LN_NEWUOA,
    "PRAXIS":      nlopt.LN_PRAXIS,
    "SUBPLEX":     nlopt.LN_SBPLX,
    "NELDER_MEAD": nlopt.LN_NELDERMEAD,
}


def optimize(
        snn, x_opt, y_opt, acc_ann_opt,
        x_val=None,
        y_val=None,
        acc_ann_val=None,
        algorithm="SUBPLEX",
        granularities=(3,),
        max_iterations=(10000,),
        global_v_initial=0.5,
        optimize_scales=True,
        optimize_v_initial=False,
        n_time_chunks=1,
        poisson=False,
        decay=None,
        clamp_rate=0,
        n_opt=None,
        n_val=None,
        mask_class=None,
        input_repeat=1,
        input_squash=1,
        lambdas=(1e2, 1e1, 1e2),
        auto_scale=(False, False, False),
        verbose=True,
        val_freq=None,
        cache_filename=None,
        cache_freq=10):
    """
    Optimizes the performance of the SNN by scaling neuron firing rates.

    The loss function is a linear combination of 1.0 - PA, TAC, and PAC
    (see metrics module for details).

    This optimization can be performed in multiple phases, each phase
    having its own unit of scaling. See the granularities and
    max_iterations arguments.

    If this is called in an MPI context, only rank 0 will return a
    results list. Other ranks will return None, acting as workers.

    :param tensorflow.keras.Model snn: The SNN to optimize
    :param numpy.ndarray x_opt: Input data for optimization
    :param numpy.ndarray y_opt: Ground-truth labels for optimization
    :param acc_ann_opt: The ANN accuracy used to compute TAC and PAC on
        the optimization set; if a time series, used to compute R-TAC
        and R-PAC instead
    :param numpy.ndarray x_val: Input data for validation
    :param numpy.ndarray y_val: Ground-truth data for validation
    :param acc_ann_val: The ANN accuracy used to compute TAC and PAC on
        the validation set; if a time series, used to compute R-TAC and
        R-PAC instead
    :param string algorithm: The name of the optimization algorithm to
        use (see optimization.ALGORITHMS)
    :param tuple granularities: The level(s) of granularity at which
        the optimization should be performed; 1=network-wise,
        2=layer-wise, 3=neuron-wise
    :param tuple max_iterations: The maximum number(s) of optimizer
        iterations; must have the same length as granularities
    :param float global_v_initial: The initial neuron membrane
        potential; if optimize_v_initial=True, this is used as the
        optimizer initialization
    :param bool optimize_scales: Whether to optimize firing rate scales
    :param bool optimize_v_initial: Whether to optimize initial neuron
        membrane potentials in addition to firing rates
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
    :param int n_opt: The number of optimization-set items to use; this
        is required if x_opt and y_opt are tf.data.Dataset
    :param int n_val: The number of validation-set items to use; this
        is required if x_val and y_val are tf.data.Dataset
    :param int mask_class: If not None, items with this ground truth
        label are ignored when computing accuracy
    :param int input_repeat: The number of SNN time steps for which each
        input frame should be repeated (this only makes sense with
        poisson input)
    :param int input_squash: The number of SNN time steps over which
        each input frame should be averaged (this only makes sense with
        poisson input)
    :param tuple lambdas: A 3-tuple consisting of (lambda_PA,
        lambda_TAC, lambda_PAC); these are constants used to scale each
        term in the loss function
    :param tuple auto_scale: A 3-tuple indicating whether each loss term
        should be automatically scaled such that its initial
        contribution on the optimization set is equal to lambdas[i]
    :param bool verbose: Whether to print updates to standard output
    :param int val_freq: If not None, the frequency with which
        performance on the validation set should be evaluated
    :param string cache_filename: The filename where evaluations should
        be cached on disk (this should have extension .p); if None,
        evaluations are not cached to disk
    :param int cache_freq: The frequency with which cached evaluations
        should be written to disk; only used if cache_filename is not
        None
    :return list: A list of dictionaries, each with keys "scales",
        "final_opt", "history_opt", "final_val", and "history_val" (the
        last two depending on whether x_val, y_val, or val_freq are
        None)
    """

    class _Optimizer:
        def __init__(self, scales, v_initial):
            self.scales = scales
            self.v_initial = v_initial
            self.s_dim = _flat_len(scales) if optimize_scales else 0
            self.v_dim = _flat_len(v_initial) if optimize_v_initial else 0

            self.iteration = 0
            self.best_opt = None
            self.last_val = None
            self.update_val = True
            self.history_opt = []
            self.history_val = []

        def create_output_dict(self):
            final_opt = _compute_metrics(self.scales, self.v_initial, "opt")
            output_dict = {
                "final_opt":   final_opt,
                "history_opt": self.history_opt
            }
            if x_val is not None and y_val is not None:
                output_dict["final_val"] = _compute_metrics(
                    self.scales, self.v_initial, "val")
            if val_freq is not None:
                output_dict["history_val"] = self.history_val
            if optimize_scales:
                output_dict["scales"] = self.scales
            if optimize_v_initial:
                output_dict["v_initial"] = self.v_initial
            return output_dict

        def opt_callback(self, flat, _):
            if optimize_scales:
                scales = _un_flatten(flat[:self.s_dim], self.scales)
            else:
                scales = 1
            if optimize_v_initial:
                v_initial = _un_flatten(flat[self.s_dim:], self.v_initial)
            else:
                v_initial = global_v_initial
            metrics_opt = _compute_metrics(scales, v_initial, "opt")
            if self.best_opt is None or metrics_opt["loss"] < self.best_opt["loss"]:
                self.scales = scales
                self.v_initial = v_initial
                self.best_opt = metrics_opt
                self.update_val = True
            self.history_opt.append(self.best_opt["loss"])

            if val_freq is not None:
                if self.update_val and self.iteration % val_freq == 0:
                    self.last_val = _compute_metrics(self.scales, self.v_initial, "val")
                    self.update_val = False
                self.history_val.append(self.last_val["loss"])

            nonlocal cache_dirty
            if cache_dirty and cache_freq is not None and self.iteration % cache_freq == 0:
                _save_cache(cache, cache_filename)
                cache_dirty = False
            self.iteration += 1
            if verbose:
                self.print_iteration_status()
            return metrics_opt["loss"]

        def optimize(self, max_iteration):
            if optimize_scales and optimize_v_initial:
                start = np.concatenate([_flatten(self.scales), _flatten(self.v_initial)])
            elif optimize_v_initial:
                start = _flatten(self.v_initial)
            else:
                start = _flatten(self.scales)
            opt = nlopt.opt(ALGORITHMS[algorithm], self.s_dim + self.v_dim)
            opt.set_lower_bounds(
                np.concatenate([np.full(self.s_dim, 1e-6), np.full(self.v_dim, -np.inf)]))
            opt.set_min_objective(self.opt_callback)
            opt.set_maxeval(max_iteration)
            final = opt.optimize(start)

            if optimize_scales:
                self.scales = _un_flatten(final[:self.s_dim], self.scales)
            if optimize_v_initial:
                self.v_initial = _un_flatten(final[self.s_dim:], self.v_initial)

        def print_iteration_status(self):
            message = "Iteration {} - Opt loss: {:#.4g}".format(
                self.iteration, self.best_opt["loss"])
            if val_freq is not None:
                message += " - Val loss: {:#.4g}".format(self.last_val["loss"])
            print(message, flush=True, end="\r")

    def _compute_metrics(scales, v_initial, split):
        results = _evaluate(scales, v_initial, split)

        term_pa = lambdas[0] * (1.0 - results["peak_accuracy"])
        term_tac = lambdas[1] * results["time_above_curve"]
        term_pac = lambdas[2] * results["power_above_curve"]
        return {
            "term_pa":  term_pa,
            "term_tac": term_tac,
            "term_pac": term_pac,
            "loss":     term_pa + term_tac + term_pac,
        }

    def _evaluate(scales, v_initial, split):
        s_flat = None
        v_flat = None
        key = split.encode()
        if optimize_scales:
            s_flat = _flatten(scales)
            key += s_flat.tobytes()
        if optimize_v_initial:
            v_flat = _flatten(v_initial)
            key += v_flat.tobytes()

        if key not in cache.keys():
            if n_ranks > 0:
                # Tell workers which dataset we're evaluating on
                comm.bcast(0 if split == "opt" else 1, root=0)

                # Tell workers how to scale their weights and initialize
                # their membrane potentials
                if optimize_scales:
                    comm.Bcast(s_flat, root=0)
                    snn.set_weights(start_weights)
                    scale(snn, scales)
                if optimize_v_initial:
                    comm.Bcast(v_flat, root=0)

            if split == "opt":
                x, y, acc_ann, n_items = x_opt, y_opt, acc_ann_opt, n_opt
            else:
                x, y, acc_ann, n_items = x_val, y_val, acc_ann_val, n_val
            cache[key] = _wrapper(x, y, acc_ann, n_items, v_initial)
            nonlocal cache_dirty
            cache_dirty = True

        return cache[key]

    def _print_detailed_status(scales, v_initial):
        print("Opt performance")
        _print_set_breakdown(_compute_metrics(scales, v_initial, "opt"))
        if x_val is not None and y_val is not None:
            print("Val performance")
            _print_set_breakdown(_compute_metrics(scales, v_initial, "val"))
        print(flush=True)

    def _worker_loop(s_template, v_template):
        s_buffer = _flatten(s_template)
        v_buffer = _flatten(v_template)

        while True:
            # Status determines evaluation dataset and whether to break
            status = comm.bcast(None, root=0)
            if status == 0:
                x, y, n_items = x_opt, y_opt, n_opt
            elif status == 1:
                x, y, n_items = x_val, y_val, n_val
            else:
                return

            # evaluate function has its own logic for distributing work
            # Pass None as acc_ann since worker calls to evaluate don't
            # return anything
            if optimize_scales:
                comm.Bcast(s_buffer, root=0)
                snn.set_weights(start_weights)
                scale(snn, _un_flatten(s_buffer, s_template))
            if optimize_v_initial:
                comm.Bcast(v_buffer, root=0)
                _wrapper(x, y, None, n_items, _un_flatten(v_buffer, v_template))
            else:
                _wrapper(x, y, None, n_items, global_v_initial)

    def _wrapper(x, y, acc_ann, n_items, v_initial):
        return evaluate(
            snn, x, y,
            acc_ann=acc_ann,
            v_initial=v_initial,
            n_time_chunks=n_time_chunks,
            poisson=poisson,
            decay=decay,
            clamp_rate=clamp_rate,
            n_items=n_items,
            mask_class=mask_class,
            input_repeat=input_repeat,
            input_squash=input_squash)

    assert optimize_scales or optimize_v_initial
    assert len(granularities) == len(max_iterations)

    comm = mpi4py.MPI.COMM_WORLD
    n_ranks = comm.Get_size()
    rank = comm.Get_rank()

    # Things the functions above will access contextually
    start_weights = snn.get_weights()
    cache = _load_cache(cache_filename)
    cache_dirty = False

    s_cumulative = scaling_template(snn, granularities[0])
    v_cumulative = v_initial_template(snn, granularities[0], value=global_v_initial)
    output = []

    if any(auto_scale):
        initial = _evaluate(s_cumulative, v_cumulative, "opt")
        if rank == 0:
            l_0, l_1, l_2 = lambdas
            if auto_scale[0]:
                l_0 /= 1.0 - initial["peak_accuracy"]
            if auto_scale[1]:
                l_1 /= initial["time_above_curve"]
            if auto_scale[2]:
                l_2 /= initial["power_above_curve"]
            lambdas = l_0, l_1, l_2

    for i, granularity in enumerate(granularities):
        if i > 0:
            if optimize_scales:
                s_cumulative = _multiply(scaling_template(snn, granularity), s_cumulative)
            if optimize_v_initial:
                v_cumulative = _multiply(v_initial_template(snn, granularity), v_cumulative)

        try:
            # Only rank zero engages in optimizer logic
            if rank == 0:
                if verbose:
                    print(" Granularity {} ".format(granularity).center(79, "="))
                    _print_detailed_status(s_cumulative, v_cumulative)

                optimizer = _Optimizer(s_cumulative, v_cumulative)
                optimizer.optimize(max_iterations[i])
                output.append(optimizer.create_output_dict())

                if optimize_scales:
                    s_cumulative = output[-1]["scales"]
                if optimize_v_initial:
                    v_cumulative = output[-1]["v_initial"]
                if verbose:
                    print("\n")
                    _print_detailed_status(s_cumulative, v_cumulative)

                # Cache is also periodically saved during optimization
                _save_cache(cache, cache_filename)

                # Tell workers to break out of their loops
                comm.bcast(-1, root=0)

            # Nonzero ranks just act as evaluation engines
            else:
                _worker_loop(s_cumulative, v_cumulative)

        finally:
            snn.set_weights(start_weights)

    if rank == 0:
        return output
    else:
        return None


def _flat_len(array_list):
    flat_len = 0
    for item in array_list:
        flat_len += _len(item)
    return flat_len


def _flatten(array_list):
    flattened = np.empty(_flat_len(array_list))
    i = 0
    for item in array_list:
        item_len = _len(item)
        flattened[i:i + item_len] = item
        i += item_len
    return flattened


def _len(item):
    if isinstance(item, float):
        return 1
    else:
        return len(item)


def _load_cache(filename):
    if filename is not None and path.exists(filename):
        with open(filename, "rb") as cache_file_r:
            return pickle.load(cache_file_r)
    else:
        return {}


def _multiply(array_list_1, array_list_2):
    if len(array_list_1) <= len(array_list_2):
        short, long = array_list_1, array_list_2
    else:
        short, long = array_list_2, array_list_1
    assert len(short) == len(long) or len(short) == 1

    product = []
    for i in range(len(long)):
        short_i = short[0] if len(short) == 1 else short[i]
        product.append(short_i * long[i])
    return product


def _print_set_breakdown(metrics):
    print(" " * 4 + "Loss: {:#.4g}".format(metrics["loss"]))
    print(" " * 4 + "PA term: {:#.4g}".format(metrics["term_pa"]))
    print(" " * 4 + "TAC term: {:#.4g}".format(metrics["term_tac"]))
    print(" " * 4 + "PAC term: {:#.4g}".format(metrics["term_pac"]))


def _save_cache(cache, filename):
    if filename is not None:
        # Caches can get very large and thus take a long time to write
        # Ensure there is always an uncorrupted copy of the cache by
        # initially writing to a temporary file
        tmp_written = False
        try:
            with open(filename + ".tmp", "wb") as cache_file_w:
                pickle.dump(cache, cache_file_w)
            tmp_written = True
        finally:
            if tmp_written:
                if path.isfile(filename):
                    os.remove(filename)
                os.rename(filename + ".tmp", filename)
            else:
                os.remove(filename + ".tmp")


def _un_flatten(array, template):
    flattened_len = 0
    for item in template:
        flattened_len += _len(item)
    assert flattened_len == len(array)

    un_flattened = []
    i = 0
    for item in template:
        item_len = _len(item)
        if item_len == 1:
            un_flattened.append(array[i])
        else:
            un_flattened.append(np.copy(array[i:i + item_len]))
        i += item_len
    return un_flattened
