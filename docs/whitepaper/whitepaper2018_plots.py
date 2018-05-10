from functools import partial
import itertools
import os
import pickle
import sys
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import click
import matplotlib.pyplot as plt
import nengo
from nengo import spa
import nengo_dl
from nengo_dl import graph_optimizer, benchmarks
import nengo_ocl
import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import mnist

sys.path.append("../../../../spaun")
from _spaun.configurator import cfg
from _spaun.vocabulator import vocab
from _spaun.experimenter import experiment
from _spaun.modules.vision.data import vis_data
from _spaun.modules.motor.data import mtr_data
from _spaun.spaun_main import Spaun


def filter_results(results, **kwargs):
    return [x["relative_time"] if "relative_time" in x else x["times"]
            for x in results if all(x[k] == v for k, v in kwargs.items())]


def bootstrap_ci(data, alpha=0.95, n_samples=1000, func=np.mean):
    samples = sorted(
        func(np.random.choice(data, replace=True, size=len(data)))
        for _ in range(n_samples))
    lower = int(n_samples * (1 - alpha) / 2)
    upper = int(n_samples * (alpha + (1 - alpha) / 2))
    return func(data), samples[lower], samples[upper]


def build_spaun(dimensions):
    vocab.sp_dim = dimensions
    cfg.mtr_arm_type = None

    cfg.set_seed(1)
    experiment.initialize('A', vis_data.get_image_ind,
                          vis_data.get_image_label,
                          cfg.mtr_est_digit_response_time, cfg.rng)
    vocab.initialize(experiment.num_learn_actions, cfg.rng)
    vocab.initialize_mtr_vocab(mtr_data.dimensions, mtr_data.sps)
    vocab.initialize_vis_vocab(vis_data.dimensions, vis_data.sps)

    with Spaun() as net:
        nengo_dl.configure_settings(trainable=False)

    return net


@click.group()
def main():
    pass


@main.command()
@click.option("--load/--no-load", default=False)
@click.option("--batch", default=1)
@click.option("--reps", default=10)
def compare_backends(load, batch, reps):
    bench_names = ["cconv", "integrator", "pes"]
    n_range = [2048, 4096]
    d_range = [64, 128, 256]
    neuron_types = [nengo.RectifiedLinear()]
    backends = ["nengo_dl", "nengo_ocl", "nengo"]
    sim_time = 10.0

    params = list(itertools.product(
        bench_names, n_range, d_range, neuron_types, backends))

    if load:
        with open("compare_backends_%d_data_saved.pkl" % batch, "rb") as f:
            results = pickle.load(f)
    else:
        results = [{"times": [], "benchmark": bench, "n_neurons": n_neurons,
                    "dimensions": dimensions, "neuron_type": neuron_type,
                    "backend": backend}
                   for bench, n_neurons, dimensions, neuron_type, backend
                   in params]

    n_results = len(results[0]["times"])
    for r in range(n_results, n_results + reps):
        print("=" * 30)
        print("REP %d" % r)
        for i, (bench, n_neurons, dimensions, neuron_type,
                backend) in enumerate(params):
            print("%d/%d: %s %s %s %s %s" % (
                i + 1, len(params), bench, n_neurons, dimensions, neuron_type,
                backend))

            net = getattr(benchmarks, bench)(
                dimensions=dimensions, neurons_per_d=n_neurons // dimensions,
                neuron_type=neuron_type)

            with net:
                nengo_dl.configure_settings(trainable=False)

            if "nengo_dl" in backend:
                sim = nengo_dl.Simulator(
                    net, unroll_simulation=25, minibatch_size=batch,
                    device="/gpu:0", progress_bar=False)
            elif backend == "nengo":
                sim = nengo.Simulator(net, progress_bar=False, optimize=True)
            elif backend == "nengo_ocl":
                sim = nengo_ocl.Simulator(net, progress_bar=False)

            with sim:
                # run once to eliminate startup overhead
                sim.run(0.1, progress_bar=False)

                start = time.time()
                for b in range(1 if "nengo_dl" in backend else batch):
                    if b > 0:
                        sim.reset()
                    sim.run(sim_time, progress_bar=False)
                results[i]["times"].append((time.time() - start) / sim_time)

            print("   ", min(results[i]["times"]), max(results[i]["times"]),
                  np.mean(results[i]["times"]))

        with open("compare_backends_%d_data.pkl" % batch, "wb") as f:
            pickle.dump(results, f)

    # plotting
    f, axes = plt.subplots(1, len(benchmarks), sharey=True, sharex=False,
                           figsize=(5 * len(benchmarks), 5))
    n_bars = len(d_range)
    neuron_type = nengo.RectifiedLinear()
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for k, m in enumerate(benchmarks):
        x_pos = np.arange(n_bars)
        for j, b in enumerate(backends):
            bottoms = np.zeros(n_bars)
            c = 0
            for n in n_range:
                data = np.asarray([bootstrap_ci(t) for t in filter_results(
                    results, benchmark=m, neuron_type=neuron_type,
                    n_neurons=n, backend=b)])

                axes[k].bar(x_pos, data[:, 0],
                            yerr=abs(np.transpose(data[:, 1:] - data[:, [0]])),
                            width=0.5, bottom=bottoms, color=colours[c])
                bottoms += data[:, 0]
                c += 1
            x_pos += n_bars + 1

        axes[k].set_title("%s" % m)
        if k == 0:
            axes[k].legend(["N=%d" % n for n in n_range])
        axes[k].set_xticks(np.concatenate(
            [np.arange(i * (n_bars + 1), i * (n_bars + 1) + n_bars)
             for i in range(len(backends))]))
        axes[k].set_xticklabels([t for _ in range(len(backends))
                                 for t in d_range])
        for i, b in enumerate(backends):
            axes[k].annotate(
                b, (((n_bars - 1) / 2 + (n_bars + 1) * i + 1) /
                    ((n_bars + 1) * len(backends)),
                    -0.1),
                xycoords="axes fraction", ha="center")

        axes[k].set_ylim([0, 10 * batch])
        axes[k].set_xlim([-1, (n_bars + 1) * len(benchmarks) - 1])

        if k == 0:
            axes[k].set_ylabel("real time / simulated time")

    plt.tight_layout(rect=(0, 0.05, 1, 1))

    plt.savefig("compare_backends_%d.pdf" % batch)
    plt.show()


@main.command()
@click.option("--load/--no-load", default=False)
def compare_backends_spaun(load):
    backends = [nengo_dl, nengo_ocl, nengo]
    d_range = [64, 128, 256]

    if load:
        with open("compare_backends_spaun_data.pkl", "rb") as f:
            results = pickle.load(f)
    else:
        params = list(itertools.product(d_range, backends))

        results = []
        net = None
        for i, (dimensions, backend) in enumerate(params):
            print("%d/%d: %s %s" % (i + 1, len(params), backend.__name__,
                                    dimensions))

            if net is None or vocab.sp_dim != dimensions:
                net = build_spaun(dimensions)

            if backend == nengo_dl:
                kwargs = {"unroll_simulation": 50,
                          "minibatch_size": None,
                          "device": "/gpu:0",
                          "dtype": tf.float32,
                          "progress_bar": True
                          }
                with net:
                    nengo_dl.configure_settings(session_config={
                        "gpu_options.allow_growth": True})
            elif backend == nengo:
                kwargs = {"progress_bar": True,
                          "optimize": True}
            elif backend == nengo_ocl:
                kwargs = {"progress_bar": True}

            with backend.Simulator(net, **kwargs) as sim:
                sim.run(0.1, progress_bar=False)

                sim_time = 1.0
                start = time.time()
                sim.run(sim_time)
                data = {"relative_time": (time.time() - start) / sim_time,
                        "backend": backend.__name__, "dimensions": dimensions}

            print("  %.2fx realtime" % (1. / data["speed"]))
            results.append(data)

        with open("compare_backends_spaun_data.pkl", "wb") as f:
            pickle.dump(results, f)

    plt.figure()
    for backend in backends:
        plt.plot(d_range, filter_results(results, backend=backend.__name__),
                 label=backend.__name__)
    plt.legend()
    plt.xlabel("dimensions")
    plt.ylabel("real time / simulated time")

    plt.show()


@main.command()
@click.option("--load/--no-load", default=False)
@click.option("--reps", default=10)
def compare_optimizations(load, reps):
    dimensions = 4

    # optimizations to apply (simplifications, merging, sorting, unroll)
    params = [
        (False, False, False, False),
        (False, True, False, False),
        (False, True, True, False),
        (True, True, True, False),
        (True, True, True, True)
    ]
    # params = list(itertools.product((False, True), repeat=4))

    if load:
        with open("compare_optimizations_data.pkl", "rb") as f:
            results = pickle.load(f)
    else:
        results = [{"times": [], "simplifications": simp, "planner": plan,
                    "sorting": sort, "unroll": unro}
                   for simp, plan, sort, unro in params]

    net = build_spaun(dimensions)
    # net = nengo_benchmarks.all_benchmarks["convolution"](
    #     dimensions=dimensions, n_neurons=1024).model()
    model = nengo.builder.Model(
        dt=0.001, builder=nengo_dl.builder.NengoBuilder())
    model.build(net)

    for r in range(reps):
        print("=" * 30)
        print("REP %d" % r)
        for i, (simp, plan, sort, unro) in enumerate(params):
            print("%d/%d: %s %s %s %s" % (i + 1, len(params), simp, plan, sort,
                                          unro))
            with net:
                config = {}
                if simp:
                    config["simplifications"] = [
                        graph_optimizer.remove_constant_copies,
                        graph_optimizer.remove_unmodified_resets,
                        # graph_optimizer.remove_zero_incs,
                        graph_optimizer.remove_identity_muls
                    ]
                else:
                    config["simplifications"] = []
                if not plan:
                    config[
                        "planner"] = graph_optimizer.greedy_planner  # graph_optimizer.noop_planner
                if not sort:
                    config["sorter"] = graph_optimizer.noop_order_signals
                nengo_dl.configure_settings(**config)

            with nengo_dl.Simulator(
                    None, model=model, unroll_simulation=50 if unro else 1,
                    device="/gpu:0") as sim:
                sim.run(0.1)

                sim_time = 1.0
                start = time.time()
                sim.run(sim_time)
                results[i]["times"].append((time.time() - start) / sim_time)

            print("   ", min(results[i]["times"]), max(results[i]["times"]),
                  np.mean(results[i]["times"]))

        with open("compare_optimizations_data.pkl", "wb") as f:
            pickle.dump(results, f)

    plt.figure()
    plt.bar(np.arange(len(results)), [np.mean(t) for t in
                                      filter_results(results)])
    labels = ["none"]
    for r in results[1:]:
        lab = ""
        if r["planner"]:
            lab += "merging\n"
        if r["sorting"]:
            lab += "sorting\n"
        if r["simplifications"]:
            lab += "simplifications\n"
        if r["unroll"]:
            lab += "unrolling\n"
        labels.append(lab)
    plt.xticks(np.arange(len(results)), labels, rotation="vertical")
    plt.ylabel("real time / simulated time")

    plt.show()


@main.command()
@click.option("--load/--no-load", default=False)
@click.option("--reps", default=10)
@click.option("--dimensions", default=4)
def compare_simplifications(load, reps, dimensions):
    simplifications = [
        graph_optimizer.remove_constant_copies,
        graph_optimizer.remove_unmodified_resets,
        graph_optimizer.remove_zero_incs,
        graph_optimizer.remove_identity_muls
    ]

    params = list(
        itertools.product((False, True), repeat=len(simplifications)))

    if load:
        with open("compare_simplifications_data.pkl", "rb") as f:
            results = pickle.load(f)
    else:
        results = [
            dict([("times", [])] + [
                (s.__name__, p[i]) for i, s in enumerate(simplifications)])
            for j, p in enumerate(params)]

    net = build_spaun(dimensions)
    model = nengo.builder.Model(
        dt=0.001, builder=nengo_dl.builder.NengoBuilder())
    model.build(net)

    for r in range(reps):
        print("=" * 30)
        print("REP %d" % r)

        for j, p in enumerate(params):
            simps = []
            for i, s in enumerate(p):
                if s:
                    simps.append(simplifications[i])

            with net:
                nengo_dl.configure_settings(simplifications=simps)

            print("%d/%d" % (j + 1, len(params)), [x.__name__ for x in simps])

            with nengo_dl.Simulator(
                    None, model=model, unroll_simulation=1, device="/gpu:0",
                    progress_bar=False) as sim:
                sim.run(0.1, progress_bar=False)

                sim_time = 1.0
                start = time.time()
                sim.run(sim_time, progress_bar=False)
                results[j]["times"].append((time.time() - start) / sim_time)

            print("   ", min(results[j]["times"]), max(results[j]["times"]),
                  np.mean(results[j]["times"]))

        with open("compare_simplifications_data.pkl", "wb") as f:
            pickle.dump(results, f)


@main.command()
def spiking_mnist():
    data = mnist.read_data_sets("MNIST_data/", one_hot=True)
    minibatch_size = 200

    def build_network(neuron_type, ens_params):
        with nengo.Network() as net:
            nengo_dl.configure_settings(trainable=False)

            inp = nengo.Node([0] * 28 * 28)

            x = nengo_dl.tensor_layer(
                inp, tf.layers.conv2d, shape_in=(28, 28, 1), filters=32,
                kernel_size=3)
            x = nengo_dl.tensor_layer(x, neuron_type, **ens_params)

            x = nengo_dl.tensor_layer(
                x, tf.layers.conv2d, shape_in=(26, 26, 32),
                filters=64, kernel_size=3)
            x = nengo_dl.tensor_layer(x, neuron_type, **ens_params)

            x = nengo_dl.tensor_layer(
                x, tf.layers.average_pooling2d, shape_in=(24, 24, 64),
                pool_size=2, strides=2)

            x = nengo_dl.tensor_layer(
                x, tf.layers.conv2d, shape_in=(12, 12, 64),
                filters=128, kernel_size=3)
            x = nengo_dl.tensor_layer(x, neuron_type, **ens_params)

            x = nengo_dl.tensor_layer(
                x, tf.layers.average_pooling2d, shape_in=(10, 10, 128),
                pool_size=2, strides=2)

            x = nengo_dl.tensor_layer(x, tf.layers.dense, units=10)

        return net, inp, x

    # construct the network
    net, inp, out = build_network(
        # nengo_dl.SoftLIFRate(amplitude=0.01, sigma=0.001),
        nengo.LIFRate(amplitude=0.01),
        dict(max_rates=nengo.dists.Choice([100]),
             intercepts=nengo.dists.Choice([0]))
    )
    with net:
        out_p = nengo.Probe(out)

    # construct the simulator
    with nengo_dl.Simulator(net, minibatch_size=minibatch_size) as sim:
        train_inputs = {inp: data.train.images[:, None, :]}
        train_targets = {out_p: data.train.labels[:, None, :]}
        test_inputs = {inp: data.test.images[:, None, :]}
        test_targets = {out_p: data.test.labels[:, None, :]}

        def objective(x, y):
            return tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=x, labels=y)

        opt = tf.train.RMSPropOptimizer(learning_rate=0.001)

        def classification_error(outputs, targets):
            return 100 * tf.reduce_mean(
                tf.cast(tf.not_equal(tf.argmax(outputs[:, -1], axis=-1),
                                     tf.argmax(targets[:, -1], axis=-1)),
                        tf.float32))

        print("error before training: %.2f%%" % sim.loss(
            test_inputs, test_targets, classification_error))

        do_training = True
        if do_training:
            # run training
            sim.train(train_inputs, train_targets, opt, objective=objective,
                      n_epochs=10)

            # save the parameters to file
            sim.save_params("./mnist_params")
        else:
            # load parameters
            sim.load_params("./mnist_params")

        print("error after training: %.2f%%" % sim.loss(
            test_inputs, test_targets, classification_error))

    # test performance with spiking neurons
    net, inp, out = build_network(
        nengo.LIF(amplitude=0.01),
        dict(max_rates=nengo.dists.Choice([100]),
             intercepts=nengo.dists.Choice([0]))
    )
    with net:
        out_p = nengo.Probe(out, synapse=0.1)

    with nengo_dl.Simulator(net, minibatch_size=minibatch_size,
                            unroll_simulation=10) as sim:
        sim.load_params("./mnist_params")

        n_steps = 50
        test_inputs_time = {
            inp: np.tile(data.test.images[:, None, :], (1, n_steps, 1))}
        test_targets_time = {out_p: np.tile(v, (1, n_steps, 1)) for v in
                             test_targets.values()}

        print("spiking neuron error: %.2f%%" % sim.loss(test_inputs_time,
                                                        test_targets_time,
                                                        classification_error))

        sim.run_steps(n_steps, input_feeds={
            inp: test_inputs_time[inp][:minibatch_size]})

        for i in range(5):
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(np.reshape(data.test.images[i], (28, 28)))
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.plot(sim.trange(), sim.data[out_p][i])
            plt.legend([str(i) for i in range(10)], loc="upper left")
            plt.xlabel("time")

    plt.show()


@main.command()
@click.option("--load/--no-load", default=False)
@click.option("--reps", default=10)
def spa_optimization(load, reps):
    def get_binding_data(n_inputs, n_pairs, dims, seed, t_int, t_mem,
                         dt=0.001):
        int_steps = int(t_int / dt)
        mem_steps = int(t_mem / dt)
        n_steps = int_steps * n_pairs + mem_steps

        rng = np.random.RandomState(seed)
        vocab = spa.Vocabulary(dimensions=dims, rng=rng, max_similarity=1)

        # initialize arrays for input and output trajectories
        roles = np.zeros((n_inputs, n_steps, dims))
        fills = np.zeros((n_inputs, n_steps, dims))
        cues = np.zeros((n_inputs, n_steps, dims))
        binding = np.zeros((n_inputs, n_steps, dims))
        memory = np.zeros((n_inputs, n_steps, dims))
        output = np.zeros((n_inputs, n_steps, dims))

        # iterate through examples to be generated, fill arrays
        for n in range(n_inputs):
            role_names = ["ROLE_%d_%d" % (n, i) for i in range(n_pairs)]
            filler_names = ["FILLER_%d_%d" % (n, i) for i in range(n_pairs)]

            # each role/filler pair is presented for t_int seconds
            for i in range(n_pairs):
                roles[n, i * int_steps:(i + 1) * int_steps] = vocab.parse(
                    role_names[i]).v
                fills[n, i * int_steps:(i + 1) * int_steps] = vocab.parse(
                    filler_names[i]).v
                binding[n, i * int_steps:(i + 1) * int_steps] = vocab.parse(
                    "%s*%s" % (role_names[i], filler_names[i])).v

            # randomly select a cue
            cue_idx = rng.randint(n_pairs)

            # cue is presented during the memorization period
            cues[n, -mem_steps:, :] = vocab[role_names[cue_idx]].v

            # the goal is to output the associated filler during the
            # memorization phase
            # note: we use nan for the target prior to the memorization phase,
            # to indicate that it doesn't matter what the network output is
            output[n, -mem_steps:, :] = vocab[filler_names[cue_idx]].v
            output[n, :-mem_steps, :] = np.nan

        memory[...] = np.cumsum(binding, axis=1) * dt / t_int

        return roles, fills, cues, binding, memory, output, vocab

    def accuracy(outputs, targets, vocab=None):
        vocab_vectors = tf.constant(vocab.vectors, dtype=tf.float32)
        output = outputs[:, -1, :]
        sims = tf.matmul(vocab_vectors, tf.transpose(output))
        idxs = tf.argmax(sims, axis=0)
        match = tf.reduce_all(tf.equal(
            tf.gather(vocab_vectors, idxs), targets[:, -1]),
            axis=1)

        return tf.reduce_mean(tf.cast(match, tf.float32))

    def build_network(neurons_per_d, seed):
        with nengo.Network(seed=seed) as net:
            net.config[nengo.Ensemble].neuron_type = nengo.RectifiedLinear()
            # net.config[nengo.Ensemble].gain = nengo.dists.Choice([1])
            # net.config[nengo.Ensemble].bias = nengo.dists.Choice([0])
            net.config[nengo.Ensemble].gain = nengo.dists.Uniform(0.5, 1)
            net.config[nengo.Ensemble].bias = nengo.dists.Uniform(-0.1, 0.1)
            net.config[nengo.Connection].synapse = None

            net.role_inp = nengo.Node(np.zeros(dims))
            net.fill_inp = nengo.Node(np.zeros(dims))
            net.cue_inp = nengo.Node(np.zeros(dims))

            # circular convolution network to combine roles/fillers
            cconv = nengo.networks.CircularConvolution(neurons_per_d, dims)
            nengo.Connection(net.role_inp, cconv.input_a)
            nengo.Connection(net.fill_inp, cconv.input_b)

            # memory network to store the role/filler pairs
            memory = nengo.Ensemble(neurons_per_d * dims, dims)
            tau = 0.01
            nengo.Connection(cconv.output, memory, transform=tau / t_int,
                             synapse=tau)
            nengo.Connection(memory, memory, transform=1, synapse=tau)

            # another circular convolution network to extract the cued filler
            ccorr = nengo.networks.CircularConvolution(neurons_per_d, dims,
                                                       invert_b=True)
            nengo.Connection(memory, ccorr.input_a)
            nengo.Connection(net.cue_inp, ccorr.input_b)

            net.conv_probe = nengo.Probe(cconv.output, label="conv_probe")
            net.memory_probe = nengo.Probe(memory, label="memory_probe")
            net.output_probe = nengo.Probe(ccorr.output, label="output_probe")

        return net

    # we'll define a slightly modified version of mean squared error that
    # allows us to specify a weighting (so that we can specify a different
    # weight for each probe)
    def weighted_mse(output, target, weight=1):
        target = tf.where(tf.is_nan(target), output, target)
        return weight * tf.reduce_mean(tf.square(target - output))

    t_int = 0.01  # length of time to present each input pair
    t_mem = 0.03  # length of memorization period
    n_pairs = 2  # number of role/filler pairs in each input
    dims = 64  # dimensionality of semantic pointer vectors
    minibatch_size = 64
    optimizer = tf.train.RMSPropOptimizer(1e-4)

    params = [5, 10, 15, 20]

    if load:
        with open("spa_optimization_data_saved.pkl", "rb") as f:
            results = pickle.load(f)
    else:
        results = [{"pre_retrieval": [], "post_retrieval": [], "pre_mse": [],
                    "post_mse": [], "neurons_per_d": n} for n in params]

    n_results = len(results[0]["pre_retrieval"])
    for r in range(n_results, n_results + reps):
        print("=" * 30)
        print("REP %d" % r)

        seed = r

        # generate training data
        (train_roles, train_fills, train_cues, train_binding, train_memory,
         train_targets, _) = get_binding_data(8000, n_pairs, dims, seed, t_int,
                                              t_mem)
        # generate test data
        (test_roles, test_fills, test_cues, _, _, test_targets,
         test_vocab) = get_binding_data(1024, n_pairs, dims, seed + 1, t_int,
                                        t_mem)

        acc = partial(accuracy, vocab=test_vocab)

        for i, n in enumerate(params):
            print("neurons_per_d", n)

            net = build_network(n, seed)
            train_inputs = {net.role_inp: train_roles,
                            net.fill_inp: train_fills,
                            net.cue_inp: train_cues}
            train_outputs = {net.output_probe: train_targets,
                             net.conv_probe: train_binding,
                             net.memory_probe: train_memory}

            test_inputs = {net.role_inp: test_roles, net.fill_inp: test_fills,
                           net.cue_inp: test_cues}
            test_outputs = {net.output_probe: test_targets}

            with nengo_dl.Simulator(
                    net, seed=seed, minibatch_size=minibatch_size,
                    progress_bar=False) as sim:
                results[i]["pre_retrieval"].append(sim.loss(
                    test_inputs, test_outputs, acc))
                print('pre retrieval:', results[i]["pre_retrieval"][-1])

                results[i]["pre_mse"].append(sim.loss(
                    test_inputs, test_outputs, "mse"))
                print('pre mse:', results[i]["pre_mse"][-1])

                sim.train(train_inputs, train_outputs, optimizer, n_epochs=10,
                          objective={net.output_probe: weighted_mse,
                                     net.conv_probe: partial(weighted_mse,
                                                             weight=0.25),
                                     net.memory_probe: partial(weighted_mse,
                                                               weight=0.25)})

                results[i]["post_mse"].append(sim.loss(
                    test_inputs, test_outputs, "mse"))
                print('post mse:', results[i]["post_mse"][-1])

                results[i]["post_retrieval"].append(sim.loss(
                    test_inputs, test_outputs, acc))
                print('post retrieval:', results[i]["post_retrieval"][-1])

        with open("spa_optimization_data.pkl", "wb") as f:
            pickle.dump(results, f)

    plt.figure()
    plt.plot(params, [np.mean(r["pre_retrieval"]) for r in results])
    plt.fill_between(params, *zip(*[bootstrap_ci(r["pre_retrieval"])
                                    for r in results]), alpha=0.5)
    plt.plot(params, [np.mean(r["post_retrieval"]) for r in results])
    plt.fill_between(params, *zip(*[bootstrap_ci(r["post_retrieval"])
                                    for r in results]), alpha=0.5)
    plt.xlabel("neurons per dimension")
    plt.ylabel("retrieval accuracy")
    plt.legend(["before training", "after training"])
    plt.savefig("spa_optimization.pdf")

    plt.show()


@main.command()
def all_figures():
    compare_backends(False, 1)
    compare_backends(False, 10)
    compare_optimizations(False)
    spiking_mnist()
    spa_optimization()


if __name__ == "__main__":
    main()
