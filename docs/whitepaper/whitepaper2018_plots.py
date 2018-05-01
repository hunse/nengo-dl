import itertools
import os
import pickle
import sys
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import click
import matplotlib.pyplot as plt
import nengo
import nengo_benchmarks
import nengo_dl
from nengo_dl import graph_optimizer
import nengo_ocl
import numpy as np
import tensorflow as tf

sys.path.append("../../../../spaun")
from _spaun.configurator import cfg
from _spaun.vocabulator import vocab
from _spaun.experimenter import experiment
from _spaun.modules.vision.data import vis_data
from _spaun.modules.motor.data import mtr_data
from _spaun.spaun_main import Spaun


def filter_results(results, **kwargs):
    return [1. / x["speed"] for x in results if
            all(x[k] == v for k, v in kwargs.items())]


def compare_backends(load_data=False):
    benchmarks = ["comm_channel", "convolution", "learning"]
    n_range = [2048, 4096]
    d_range = [64, 128, 256]
    neuron_types = [nengo.RectifiedLinear(), nengo.LIF()]
    backends = [nengo_dl, nengo_ocl, nengo]
    sim_time = 10.0

    if load_data:
        with open("compare_backends_data.pkl", "rb") as f:
            results = pickle.load(f)
    else:
        params = list(itertools.product(
            benchmarks, n_range, d_range, neuron_types, backends))

        results = []
        for i, (bench, n_neurons, dimensions, neuron_type,
                backend) in enumerate(params):
            print("%d/%d: %s %s %s %s %s" % (
                i + 1, len(params), bench, n_neurons, dimensions, neuron_type,
                backend.__name__))

            benchmark = nengo_benchmarks.all_benchmarks[bench](
                dimensions=dimensions, n_neurons=n_neurons, sim_time=sim_time)

            conf = nengo.Config(nengo.Ensemble)
            conf[nengo.Ensemble].neuron_type = neuron_type
            with conf:
                net = benchmark.model()

            if backend == nengo_dl:
                kwargs = {"unroll_simulation": 25,
                          "minibatch_size": None,
                          "device": "/gpu:0",
                          "dtype": tf.float32,
                          "progress_bar": False
                          }
            elif backend == nengo:
                kwargs = {"progress_bar": False,
                          "optimize": True}
            elif backend == nengo_ocl:
                kwargs = {"progress_bar": None}

            with backend.Simulator(net, **kwargs) as sim:
                # run once to eliminate startup overhead
                sim.run(0.1, progress_bar=False)

                data = benchmark.evaluate(sim, progress_bar=False)
                data.update(
                    {"benchmark": bench, "n_neurons": n_neurons,
                     "dimensions": dimensions, "neuron_type": neuron_type,
                     "backend": backend.__name__})

            print("  %.2fx realtime" % (1. / data["speed"]))

            results.append(data)

        with open("compare_backends_data.pkl", "wb") as f:
            pickle.dump(results, f)

    # plotting
    f, axes = plt.subplots(1, 3, sharey=True, sharex=False, figsize=(15, 5))
    n_bars = len(d_range)
    neuron_type = nengo.RectifiedLinear()
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for k, m in enumerate(benchmarks):
        x_pos = np.arange(n_bars)
        for j, b in enumerate(backends):
            bottoms = np.zeros(n_bars)
            c = 0
            for n in n_range:
                # for d in d_range:
                data = filter_results(
                    results, benchmark=m, neuron_type=neuron_type, n_neurons=n,
                    backend=b.__name__)
                axes[k].bar(x_pos, data, width=0.5, bottom=bottoms,
                            color=colours[c])
                bottoms += data
                c += 1
            x_pos += n_bars + 1

        axes[k].set_title("%s" % m)
        axes[k].legend(["N=%d" % n for n in n_range])
        axes[k].set_xticks(np.concatenate(
            [np.arange(i * (n_bars + 1), i * (n_bars + 1) + n_bars)
             for i in range(len(backends))]))
        axes[k].set_xticklabels([t for _ in range(len(backends))
                                 for t in d_range])
        for i, b in enumerate(backends):
            axes[k].annotate(
                b.__name__, (((n_bars - 1) / 2 + (n_bars + 1) * i + 1) /
                             ((n_bars + 1) * len(backends)),
                             -0.1),
                xycoords="axes fraction", ha="center")

        axes[k].set_ylim([0, 10])
        axes[k].set_xlim([-1, (n_bars + 1) * len(benchmarks) - 1])

        if k == 0:
            axes[k].set_ylabel("real time / simulated time")

    plt.tight_layout(rect=(0, 0.05, 1, 1))

    plt.show()


def compare_backends_batched(load_data):
    benchmarks = ["comm_channel", "convolution", "learning"]
    n_range = [2048, 4096]
    d_range = [64, 128, 256]
    neuron_types = [nengo.RectifiedLinear(), nengo.LIF()]
    backends = [nengo_dl, nengo_ocl, nengo]
    sim_time = 10.0

    if load_data:
        with open("compare_backends_data.pkl", "rb") as f:
            results = pickle.load(f)
    else:
        params = list(itertools.product(
            benchmarks, n_range, d_range, neuron_types, backends))

        results = []
        for i, (bench, n_neurons, dimensions, neuron_type,
                backend) in enumerate(params):
            print("%d/%d: %s %s %s %s %s" % (
                i + 1, len(params), bench, n_neurons, dimensions,
                neuron_type,
                backend.__name__))

            benchmark = nengo_benchmarks.all_benchmarks[bench](
                dimensions=dimensions, n_neurons=n_neurons,
                sim_time=sim_time)

            conf = nengo.Config(nengo.Ensemble)
            conf[nengo.Ensemble].neuron_type = neuron_type
            with conf:
                net = benchmark.model()

            if backend == nengo_dl:
                kwargs = {"unroll_simulation": 25,
                          "minibatch_size": None,
                          "device": "/gpu:0",
                          "dtype": tf.float32,
                          "progress_bar": False
                          }
            elif backend == nengo:
                kwargs = {"progress_bar": False,
                          "optimize": True}
            elif backend == nengo_ocl:
                kwargs = {"progress_bar": None}

            with backend.Simulator(net, **kwargs) as sim:
                # run once to eliminate startup overhead
                sim.run(0.1, progress_bar=False)

                data = benchmark.evaluate(sim, progress_bar=False)
                data.update(
                    {"benchmark": bench, "n_neurons": n_neurons,
                     "dimensions": dimensions, "neuron_type": neuron_type,
                     "backend": backend.__name__})

            print("  %.2fx realtime" % (1. / data["speed"]))

            results.append(data)

        with open("compare_backends_data.pkl", "wb") as f:
            pickle.dump(results, f)

    # plotting
    f, axes = plt.subplots(1, 3, sharey=True, sharex=False,
                           figsize=(15, 5))
    n_bars = len(d_range)
    neuron_type = nengo.RectifiedLinear()
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for k, m in enumerate(benchmarks):
        x_pos = np.arange(n_bars)
        for j, b in enumerate(backends):
            bottoms = np.zeros(n_bars)
            c = 0
            for n in n_range:
                # for d in d_range:
                data = filter_results(
                    results, benchmark=m, neuron_type=neuron_type,
                    n_neurons=n,
                    backend=b.__name__)
                axes[k].bar(x_pos, data, width=0.5, bottom=bottoms,
                            color=colours[c])
                bottoms += data
                c += 1
            x_pos += n_bars + 1

        axes[k].set_title("%s" % m)
        axes[k].legend(["N=%d" % n for n in n_range])
        axes[k].set_xticks(np.concatenate(
            [np.arange(i * (n_bars + 1), i * (n_bars + 1) + n_bars)
             for i in range(len(backends))]))
        axes[k].set_xticklabels([t for _ in range(len(backends))
                                 for t in d_range])
        for i, b in enumerate(backends):
            axes[k].annotate(
                b.__name__, (((n_bars - 1) / 2 + (n_bars + 1) * i + 1) /
                             ((n_bars + 1) * len(backends)),
                             -0.1),
                xycoords="axes fraction", ha="center")

        axes[k].set_ylim([0, 10])
        axes[k].set_xlim([-1, (n_bars + 1) * len(benchmarks) - 1])

        if k == 0:
            axes[k].set_ylabel("real time / simulated time")

    plt.tight_layout(rect=(0, 0.05, 1, 1))

    plt.show()


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

    return Spaun()


def compare_backends_spaun(load_data=False):
    backends = [nengo_dl, nengo_ocl, nengo]
    d_range = [64, 128, 256]

    if load_data:
        with open("compare_backends_spaun_data.pkl", "rb") as f:
            results = pickle.load(f)
    else:
        params = list(itertools.product(d_range, backends))

        results = []
        net = None
        for i, (dimensions, backend) in enumerate(params):
            print("%d/%d: %s %s" % (
                i + 1, len(params), backend.__name__, dimensions))

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
                data = {"speed": sim_time / (time.time() - start),
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


def compare_optimizations(load_data):
    dimensions = 4

    # optimizations to apply (simplifications, merging, sorting, unroll)
    params = [
        (False, False, False, False),
        (False, True, False, False),
        (False, True, True, False),
        (True, True, True, False),
        (True, True, True, True)
    ]
    # params = itertools.product((False, True), repeat=4)

    if load_data:
        with open("compare_optimizations_data.pkl", "rb") as f:
            results = pickle.load(f)
    else:
        net = build_spaun(dimensions)
        # net = nengo_benchmarks.all_benchmarks["convolution"](
        #     dimensions=dimensions, n_neurons=1024).model()
        model = nengo.builder.Model(
            dt=0.001, builder=nengo_dl.builder.NengoBuilder())
        model.build(net)

        results = []
        for i, (simp, plan, sort, unro) in enumerate(params):
            print("%d/%d: %s %s %s %s" % (i + 1, len(params), simp, plan, sort,
                                          unro))
            with net:
                config = {}
                if not simp:
                    config["simplifications"] = []
                if not plan:
                    config["planner"] = graph_optimizer.greedy_planner # graph_optimizer.noop_planner
                if not sort:
                    config["sorter"] = graph_optimizer.noop_order_signals
                nengo_dl.configure_settings(**config)

            with nengo_dl.Simulator(
                    None, model=model, unroll_simulation=50 if unro else 1,
                    device="/gpu:0") as sim:
                sim.run(0.1)

                start = time.time()
                sim.run(1.0)
                data = {"speed": 1.0 / (time.time() - start),
                        "dimensions": dimensions, "simplifications": simp,
                        "planner": plan, "sorting": sort, "unroll": unro}

            print("  %.2fx realtime" % (1. / data["speed"]))
            results.append(data)

        with open("compare_optimizations_data.pkl", "wb") as f:
            pickle.dump(results, f)

    plt.figure()
    plt.bar(np.arange(len(results)), filter_results(results))
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


def compare_simplifications(load_data):
    if load_data:
        with open("compare_simplifications_data.pkl", "rb") as f:
            results = pickle.load(f)
    else:
        net = build_spaun(4)
        model = nengo.builder.Model(
            dt=0.001, builder=nengo_dl.builder.NengoBuilder())
        model.build(net)

        simplifications = [
            graph_optimizer.remove_constant_copies,
            graph_optimizer.remove_unmodified_resets,
            graph_optimizer.remove_zero_incs,
            graph_optimizer.remove_identity_muls
        ]

        params = list(
            itertools.product((False, True), repeat=len(simplifications)))
        times = [[] for _ in params]
        reps = 10

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

                print("%d/%d" % (j + 1, len(params)),
                      [x.__name__ for x in simps])

                with nengo_dl.Simulator(None, model=model, unroll_simulation=1,
                                        device="/gpu:0",
                                        progress_bar=False) as sim:
                    sim.run(0.1, progress_bar=False)

                    start = time.time()
                    sim.run(1.0, progress_bar=False)
                    times[j].append(time.time() - start)

                    print("    ", min(times[j]), max(times[j]),
                          sum(times[j]) / len(times[j]))

            results = [
                dict([("times", times[j])] + [
                    (s.__name__, p[i]) for i, s in enumerate(simplifications)])
                for j, p in enumerate(params)]

            with open("compare_simplifications_data.pkl", "wb") as f:
                pickle.dump(results, f)

    for r in results:
        print([k for k, v in r.items() if v is True])
        times = r["times"]
        print("    ", min(times), max(times), sum(times) / len(times))


@click.command()
@click.argument("plot")
@click.option("--load/--no-load", default=False)
def main(plot, load):
    globals()[plot](load_data=load)


if __name__ == "__main__":
    main()
