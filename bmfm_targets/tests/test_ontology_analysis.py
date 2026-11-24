r"""
Known-answer tests of the ontology analysis on a hand-built DAG.

                     root
                /     |     \\
             CL:1   CL:2    CL:3 (leaf)
            /    \\  /   \\
      CL:11   CL:12    CL:4
     (leaf)   (leaf)     |
                       CL:41 (leaf)
"""

import gzip
import hashlib
import logging
import shutil
import sys
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sklearn.metrics import silhouette_score

from bmfm_targets.evaluation.ontology import (
    AlignmentConfig,
    GreedyFlowConfig,
    OntologyAnalysis,
    OntologyConfig,
    OntologyGraph,
    ProbabilityGraphConfig,
    SankeyConfig,
    alignment,
    plots,
    run_ontology_analysis,
)
from bmfm_targets.training.callbacks import OntologyAnalysisCallback

EDGES = [
    ("CL:0", "CL:1"),
    ("CL:0", "CL:2"),
    ("CL:0", "CL:3"),
    ("CL:1", "CL:11"),
    ("CL:1", "CL:12"),
    ("CL:2", "CL:12"),
    ("CL:2", "CL:4"),
    ("CL:4", "CL:41"),
]
LEAVES = ["CL:11", "CL:12", "CL:3", "CL:41"]
# rows are cells, columns are LEAVES
PROBS = np.array(
    [
        [0.7, 0.1, 0.1, 0.1],
        [0.4, 0.35, 0.05, 0.2],
        [0.1, 0.1, 0.6, 0.2],
        [0.05, 0.4, 0.1, 0.45],
    ]
)
CELLS = ["a", "b", "c", "d"]
needs_dot = pytest.mark.skipif(shutil.which("dot") is None, reason="needs graphviz")


@pytest.fixture(scope="module")
def ontology_path(tmp_path_factory):
    graph = nx.DiGraph()
    for node in dict.fromkeys(n for edge in EDGES for n in edge):
        name = "cell" if node == "CL:0" else f"Type {node}"
        graph.add_node(node, cell_type_id=node, cell_type_name=name)
    graph.add_edges_from(EDGES)
    path = tmp_path_factory.mktemp("ontology") / "ontology.graphml"
    with gzip.open(path, "wb") as fp:
        nx.write_graphml(graph, fp)
    return path


@pytest.fixture(scope="module")
def graph(ontology_path):
    return OntologyGraph(ontology_path)


@pytest.fixture(scope="module")
def analysis(graph):
    return OntologyAnalysis(graph)


@pytest.fixture()
def probabilities():
    columns = [f"cell_type_ontology_term_id_{leaf}" for leaf in LEAVES]
    return pd.DataFrame(PROBS, index=CELLS, columns=columns)


@pytest.fixture()
def node_probs(analysis, probabilities):
    return analysis.compute_node_probabilities(probabilities)


def test_graph_lookups(graph):
    assert graph.leaf_ids == set(LEAVES)
    assert graph.relationship("CL:12", "CL:12") == "exact"
    assert graph.relationship("CL:2", "CL:41") == "ancestor"
    assert graph.relationship("CL:41", "CL:2") == "descendant"
    assert graph.relationship("CL:1", "CL:41") == "unrelated"
    assert graph.resolve("type cl:12") == "CL:12"
    assert graph.resolve("nonsense") is None
    # siblings meet at CL:1; CL:11 and CL:41 meet at the root (2 + 3 steps)
    assert graph.distance("CL:11", "CL:12") == 2
    assert graph.distance("CL:11", "CL:41") == 5


def test_level_is_shortest_distance_to_leaf_and_layer_is_longest(graph):
    assert graph.leaf_distance["CL:2"] == 1
    assert graph.height["CL:2"] == 2
    assert graph.leaf_distance["CL:0"] == 1
    assert graph.height["CL:0"] == 3


def test_missing_ontology_names_the_build_instructions(tmp_path):
    with pytest.raises(FileNotFoundError, match="label_ontology/README.md"):
        OntologyGraph(tmp_path / "absent.graphml")


def test_rollup_sums_each_leaf_once(node_probs):
    np.testing.assert_allclose(node_probs["CL:0"], 1.0)
    np.testing.assert_allclose(node_probs["CL:1"], [0.8, 0.75, 0.2, 0.45])
    np.testing.assert_allclose(node_probs["CL:2"], [0.2, 0.55, 0.3, 0.85])


def test_rollup_rejects_columns_of_another_label(analysis, probabilities):
    with pytest.raises(ValueError, match="matches a leaf"):
        analysis.compute_node_probabilities(probabilities, "cell_type")


@pytest.mark.parametrize(
    ("threshold", "expected"),
    [
        (0.3, ["CL:11", "CL:11", "CL:3", "CL:41"]),
        # No leaf of b or d reaches 0.5, so they move to level 1, which holds
        # CL:1, CL:2, CL:4 and the root. The root has the highest probability.
        (0.5, ["CL:11", "CL:0", "CL:3", "CL:0"]),
        # Nothing reaches 1.1: most probable node overall.
        (1.1, ["CL:0"] * 4),
    ],
)
def test_predict(analysis, node_probs, threshold, expected):
    assert analysis.predict(node_probs, threshold).tolist() == expected


def test_uncertainty_is_zero_for_leaves_and_one_for_uniform(analysis, probabilities):
    uniform = probabilities * 0 + 0.25
    node_probs = analysis.compute_node_probabilities(uniform)
    pred = np.array(["CL:0", "CL:1", "CL:11", "CL:3"], dtype=object)
    np.testing.assert_allclose(analysis.uncertainty(node_probs, pred), [1, 1, 0, 0])


def test_hierarchical_metrics(analysis):
    pred = np.array(["CL:11", "CL:1", "CL:41", "CL:3", "CL:3"], dtype=object)
    true = np.array(["CL:11", "CL:11", "CL:2", "CL:11", "not a type"], dtype=object)
    metrics = analysis.hierarchical_metrics(pred, true)
    assert metrics["n_valid"] == 4
    assert metrics["exact_accuracy"] == 0.25
    assert metrics["ancestor_rate"] == 0.25
    assert metrics["descendant_rate"] == 0.25
    assert metrics["hierarchical_accuracy"] == 0.5
    # shared / predicted ancestors: 3/3, 2/2, 2/4, 1/2
    assert metrics["hprecision"] == pytest.approx((1 + 1 + 0.5 + 0.5) / 4)
    # shared / true ancestors: 3/3, 2/3, 2/2, 1/3
    assert metrics["hrecall"] == pytest.approx((1 + 2 / 3 + 1 + 1 / 3) / 4)


def test_hierarchical_metrics_are_nan_without_known_labels(analysis):
    metrics = analysis.hierarchical_metrics(
        np.array(["CL:3"], dtype=object), np.array(["x"], dtype=object)
    )
    assert metrics["n_valid"] == 0
    assert np.isnan(metrics["exact_accuracy"])


def test_clustering_metrics_perfect_labels_and_singleton_dropped():
    rng = np.random.default_rng(0)
    emb = np.vstack([rng.normal(0, 1, (10, 3)), rng.normal(3, 1, (11, 3))])
    pred = np.array(["x"] * 10 + ["y"] * 10 + ["z"], dtype=object)
    out = OntologyAnalysis.clustering_metrics(pred, pred, emb)
    assert out["nmi"] == pytest.approx(1)
    assert out["ari"] == 1
    assert out["asw"] == pytest.approx(silhouette_score(emb[:20], pred[:20]))
    assert np.isnan(OntologyAnalysis.clustering_metrics(pred, pred)["asw"])


def test_greedy_flow_counts(analysis, node_probs):
    # a, b follow root -> CL:1 -> CL:11; c goes to CL:3; d goes CL:2 -> CL:4 -> CL:41
    nodes, links = analysis.greedy_flow(
        node_probs, GreedyFlowConfig(min_flow_fraction=0, max_nodes_per_layer=None)
    )
    counts = nodes.set_index("cl_id")["count"].to_dict()
    assert counts == {
        "CL:0": 4,
        "CL:1": 2,
        "CL:3": 1,
        "CL:2": 1,
        "CL:11": 2,
        "CL:4": 1,
        "CL:41": 1,
    }
    layers = nodes.set_index("cl_id")["layer"]
    assert layers["CL:0"] == 0
    assert (layers[["CL:11", "CL:3", "CL:41"]] == 3).all()
    assert (links["src_layer"] < links["tgt_layer"]).all()
    prob = links.set_index(["src_key", "tgt_key"])["mean_selected_prob"]
    assert prob["L0|CL:0", "L2|CL:1"] == pytest.approx((0.8 + 0.75) / 2)


def test_greedy_flow_merges_small_nodes_and_keeps_all_cells(analysis, node_probs):
    nodes, links = analysis.greedy_flow(
        node_probs, GreedyFlowConfig(min_flow_fraction=0, max_nodes_per_layer=1)
    )
    last = nodes[nodes["layer"] == 3].set_index("cl_id")["count"].to_dict()
    assert last == {"CL:11": 2, "OTHER@L3": 2}
    assert links.loc[links["tgt_layer"] == 3, "count"].sum() == 4


def test_greedy_flow_merges_weak_links(analysis, node_probs):
    nodes, links = analysis.greedy_flow(
        node_probs, GreedyFlowConfig(min_flow_fraction=0.5, max_nodes_per_layer=None)
    )
    from_root = links[links["src_key"] == "L0|CL:0"].set_index("tgt_key")["count"]
    assert from_root.to_dict() == {
        "L1|OTHER@L1": 1,
        "L2|CL:1": 2,
        "L3|OTHER@L3": 1,
    }


def test_sankey_links_and_colors(graph):
    pytest.importorskip("plotly")
    true = np.array(["CL:11", "CL:11", "CL:2", "CL:3"], dtype=object)
    pred = np.array(["CL:11", "CL:3", "CL:41", "CL:3"], dtype=object)
    sankey = plots.plot_sankey(graph, pred, true, SankeyConfig(10, 10)).data[0]
    labels = sankey.node.label
    links = {
        (labels[s], labels[t]): (v, c)
        for s, t, v, c in zip(
            sankey.link.source, sankey.link.target, sankey.link.value, sankey.link.color
        )
    }
    colors = plots.RELATIONSHIP_COLORS
    assert links == {
        ("Type CL:11", "Type CL:11"): (1, colors["exact"]),
        ("Type CL:11", "Type CL:3"): (1, colors["unrelated"]),
        ("Type CL:2", "Type CL:41"): (1, colors["descendant"]),
        ("Type CL:3", "Type CL:3"): (1, colors["exact"]),
    }
    small = plots.plot_sankey(graph, pred, true, SankeyConfig(1, 1)).data[0]
    assert sum(small.link.value) == 4
    assert "other" in small.node.label


def test_probability_graph_contents_do_not_depend_on_input_order(graph, node_probs):
    pytest.importorskip("graphviz")
    mean_probs = {n: float(p.mean()) for n, p in node_probs.items()}
    config = ProbabilityGraphConfig(prob_threshold=0.22)
    forward = plots.build_probability_graph(graph, mean_probs, "CL:11", config)
    backward = plots.build_probability_graph(
        graph, dict(reversed(mean_probs.items())), "CL:11", config
    )
    assert forward.source == backward.source
    # means: CL:12 0.2375, CL:41 0.2375, CL:3 0.2125 (below the threshold)
    assert "Type CL:12\\n0.237" in forward.source
    assert "Type CL:3\\n" not in forward.source


def test_missing_dot_executable_raises(monkeypatch):
    pytest.importorskip("graphviz")
    pytest.importorskip("plotly")
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(ImportError, match="'dot' executable"):
        OntologyAnalysisCallback({"ground_truth_column": "x"})
    OntologyAnalysisCallback(
        {"ground_truth_column": "x", "probability_graph": None, "sankey": None}
    )


def test_missing_python_package_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "plotly", None)
    with pytest.raises(ImportError, match="'plotly' python package"):
        plots.check_plot_dependencies(needs_plotly=True, needs_graphviz=False)


def test_run_aligns_by_cell_name_and_resolves_names(
    ontology_path, probabilities, tmp_path, caplog
):
    obs = pd.DataFrame(
        {"ct": ["type cl:11", "Type CL:11", "CL:3", "unheard of"]}, index=CELLS
    )
    config = OntologyConfig("ct", ontology_path=str(ontology_path), thresholds=[0.3])
    config.sankey = config.greedy_flow = config.probability_graph = None
    with caplog.at_level(logging.WARNING):
        results = run_ontology_analysis(config, obs, probabilities.iloc[::-1], tmp_path)
    assert "unheard of" in caplog.text
    assert results.predictions["pred_t0_3"].tolist() == [
        "CL:11",
        "CL:11",
        "CL:3",
        "CL:41",
    ]
    row = pd.read_csv(tmp_path / "ontology_metrics.csv").iloc[0]
    assert row["exact_accuracy"] == 1
    assert row["n_valid"] == 3


def test_run_rejects_cells_without_probabilities(
    ontology_path, probabilities, tmp_path
):
    obs = pd.DataFrame({"ct": "CL:3"}, index=[*CELLS, "e"])
    config = OntologyConfig("ct", ontology_path=str(ontology_path))
    config.sankey = config.greedy_flow = config.probability_graph = None
    with pytest.raises(ValueError, match="1 cells have no probabilities"):
        run_ontology_analysis(config, obs, probabilities, tmp_path)


def _benchmark_task_config():
    from importlib.resources import files

    config_dir = str(files("bmfm_targets.evaluation") / "benchmark_configs")
    with initialize_config_dir(config_dir, version_base=None):
        return compose(
            "config_ontology",
            overrides=["~model", "~fields", "data_module=immune_atlas"],
        )


def _shipped_callback_config(ontology_path):
    cfg = _benchmark_task_config()
    (callback,) = cfg.task.callbacks
    callback = OmegaConf.create(OmegaConf.to_container(callback, resolve=True))
    callback.config.ontology_path = str(ontology_path)
    return callback


@needs_dot
def test_shipped_task_config_runs_end_to_end(ontology_path, tmp_path, monkeypatch):
    pytest.importorskip("plotly")
    pytest.importorskip("graphviz")
    monkeypatch.setenv("BMFM_TARGETS_SCEVAL_DATA", str(tmp_path))
    callback = instantiate(_shipped_callback_config(ontology_path), _convert_="object")
    assert isinstance(callback, OntologyAnalysisCallback)

    column = callback.config.ground_truth_column
    obs = pd.DataFrame(
        {column: ["Type CL:11", "Type CL:11", "Type CL:3", "Type CL:2"]}, index=CELLS
    )
    order = [2, 0, 3, 1]
    batches = [
        {
            "cell_type_ontology_term_id_logits": np.log(PROBS)[idx],
            "cell_name": np.array(CELLS)[idx],
            "embeddings": PROBS[idx],
        }
        for idx in (order[:2], order[2:])
    ]
    digests = []
    for run in ("first", "second"):
        trainer = SimpleNamespace(
            log_dir=str(tmp_path / run),
            predict_loop=SimpleNamespace(predictions=batches),
            datamodule=SimpleNamespace(
                predict_dataset=SimpleNamespace(processed_data=SimpleNamespace(obs=obs))
            ),
        )
        module = SimpleNamespace(
            label_dict={
                "cell_type_ontology_term_id": {c: i for i, c in enumerate(LEAVES)}
            }
        )
        callback.on_predict_end(trainer, module)
        out = tmp_path / run / "ontology"
        digests.append(
            {
                p.name: hashlib.md5(p.read_bytes()).hexdigest()
                for p in sorted(out.iterdir())
            }
        )
    metrics = pd.read_csv(out / "ontology_metrics.csv").set_index("threshold")
    # d is predicted CL:41, a descendant of its label CL:2
    assert metrics.loc[0.3, "exact_accuracy"] == 0.75
    assert metrics.loc[0.3, "hierarchical_accuracy"] == 1
    expected = {"greedy_flow.html", "greedy_flow_nodes.csv", "greedy_flow_links.csv"}
    expected |= {"ontology_metrics.csv", "ontology_predictions.csv"}
    expected |= {f"sankey_t0_{t}.html" for t in ("05", "1", "2", "3", "4", "5")}
    expected |= {f"probability_graph_CL_{n}.png" for n in (11, 2, 3)}
    assert set(digests[0]) == expected
    assert digests[0] == digests[1]


def test_callback_rejects_unknown_keys_and_missing_head(ontology_path):
    from omegaconf.errors import ConfigKeyError

    base = {
        "ground_truth_column": "ct",
        "ontology_path": str(ontology_path),
        "sankey": None,
        "greedy_flow": None,
        "probability_graph": None,
    }
    with pytest.raises(ConfigKeyError):
        OntologyAnalysisCallback({**base, "plot_types": ["heatmap"]})
    callback = OntologyAnalysisCallback(base)
    trainer = SimpleNamespace(
        predict_loop=SimpleNamespace(predictions=[{"cell_name": np.array(["a"])}])
    )
    with pytest.raises(ValueError, match="cell_type_ontology_term_id"):
        callback.on_predict_end(trainer, SimpleNamespace())


@pytest.mark.parametrize(
    ("threshold", "expected"),
    [
        # argmax path to the leaves
        (0.0, ["CL:11", "CL:11", "CL:3", "CL:41"]),
        # b stops at CL:1 (0.75) because CL:11 has 0.4; d at CL:2 (0.85)
        (0.5, ["CL:11", "CL:1", "CL:3", "CL:2"]),
        # only a (0.8) and d (0.85) leave the root
        (0.78, ["CL:1", "CL:0", "CL:0", "CL:2"]),
        (1.0, ["CL:0"] * 4),
    ],
)
def test_predict_topdown(analysis, node_probs, threshold, expected):
    assert analysis.predict_topdown(node_probs, threshold).tolist() == expected


def test_ppr_similarity_matches_linear_solve(graph):
    # PPR is the solution of (I - alpha P^T) x = (1 - alpha) e_i
    undirected = graph.graph.to_undirected()
    order = list(undirected)
    adjacency = nx.to_numpy_array(undirected, nodelist=order)
    transition = adjacency / adjacency.sum(axis=1, keepdims=True)
    n = len(order)
    exact = 0.15 * np.linalg.inv(np.eye(n) - 0.85 * transition.T)
    sim = 0.5 * (exact + exact.T)
    np.fill_diagonal(sim, 1.0)
    sim /= np.linalg.norm(sim, axis=1, keepdims=True)
    sim = 0.5 * (sim + sim.T)
    np.fill_diagonal(sim, 1.0)
    np.testing.assert_allclose(
        alignment.ppr_similarity(graph, order, 0.85), sim, atol=1e-9
    )


def test_type_centroids_skip_labels_outside_the_ontology(graph):
    labels = np.array(["CL:3", "CL:11", "CL:3", "other", np.nan], dtype=object)
    emb = np.array([[0.0, 2], [1, 1], [2, 4], [9, 9], [9, 9]])
    centroids, types = alignment.type_centroids(graph, emb, labels)
    assert types == ["CL:11", "CL:3"]
    np.testing.assert_allclose(centroids, [[1, 1], [1, 3]])


def test_neighborhood_metrics_known_answers():
    ont = np.array(
        [[1, 0.9, 0.8, 0.1], [0.9, 1, 0.8, 0.1], [0.9, 0.8, 1, 0], [0, 0.1, 0.2, 1]]
    )
    same = alignment.neighborhood_metrics(ont, ont, 2)
    assert (same == 1).all(axis=None)
    emb = ont.copy()
    emb[0] = [1, 0.1, 0.8, 0.9]  # neighbors 3, 2 against 1, 2
    row = alignment.neighborhood_metrics(ont, emb, 2).iloc[0]
    assert row["jaccard"] == pytest.approx(1 / 3)
    assert row["recall"] == 0.5
    assert row["map"] == pytest.approx((1 / 2) / 2)  # the hit has rank 2


@pytest.fixture(scope="module")
def big_graph(tmp_path_factory):
    tree = nx.balanced_tree(3, 4, create_using=nx.DiGraph)
    graph = nx.DiGraph()
    for node in tree:
        graph.add_node(f"CL:{node}", cell_type_id=f"CL:{node}", cell_type_name="x")
    graph.add_edges_from((f"CL:{a}", f"CL:{b}") for a, b in tree.edges)
    path = tmp_path_factory.mktemp("big") / "ontology.graphml"
    with gzip.open(path, "wb") as fp:
        nx.write_graphml(graph, fp)
    return OntologyGraph(path)


def test_alignment_of_ontology_shaped_vectors_is_significant(big_graph):
    types = sorted(big_graph.leaf_ids)[:40]
    sim = alignment.ppr_similarity(big_graph, types)
    # vectors whose cosine similarity is the ontology similarity
    values, vectors = np.linalg.eigh(sim)
    assert values.min() > 0
    config = AlignmentConfig(bootstrap_reps=50, perm_reps=99)
    metrics, pairs, per_type = alignment.evaluate_alignment(
        big_graph, vectors * np.sqrt(values), types, config
    )
    assert metrics["pearson"] == pytest.approx(1)
    assert metrics["pearson_p"] == 0.01  # (0 + 1) / (99 + 1)
    assert metrics["recall"] == 1
    assert metrics["pearson_ci_lower"] == pytest.approx(1)
    assert len(pairs) == 40 * 39 / 2
    assert per_type["type"].tolist() == types


def test_permutation_p_values_are_uniform_for_random_vectors(big_graph):
    types = sorted(big_graph.leaf_ids)[:20]
    config = AlignmentConfig(bootstrap_reps=0, perm_reps=19)
    rng = np.random.default_rng(0)
    p = [
        alignment.evaluate_alignment(
            big_graph, rng.normal(size=(20, 8)), types, config
        )[0]["spearman_p"]
        for _ in range(60)
    ]
    assert 0.35 < np.mean(p) < 0.65
    assert np.mean(np.array(p) <= 0.05) < 0.15


def test_bootstrap_interval_contains_estimate_and_is_seeded(big_graph):
    types = sorted(big_graph.leaf_ids)[:30]
    ont = alignment.ppr_similarity(big_graph, types)
    noise = np.random.default_rng(1).normal(size=ont.shape) * 0.01
    emb = ont + noise + noise.T
    ci = alignment.bootstrap_interval(ont, emb, 100, seed=3)
    assert ci == alignment.bootstrap_interval(ont, emb, 100, seed=3)
    upper = np.triu_indices(30, k=1)
    estimate = np.corrcoef(ont[upper], emb[upper])[0, 1]
    assert ci["pearson_ci_lower"] < estimate < ci["pearson_ci_upper"]


def test_alignment_needs_four_types(graph):
    with pytest.raises(ValueError, match="at least 4 types"):
        alignment.evaluate_alignment(graph, np.eye(3), LEAVES[:3], AlignmentConfig())


def test_alpha_sweep_rows(graph):
    sweep = alignment.alpha_sweep(graph, PROBS, LEAVES, alphas=(0.5, 0.9))
    assert sweep["alpha"].tolist() == [0.5, 0.9]
    assert sweep["pearson"].nunique() == 2


@pytest.mark.parametrize(
    "key",
    [
        "model.classifiers.ct.out_proj.weight",
        "model.classifiers.ct.weight",
        "model.cls.label_predictions.predictions.label_decoders.ct.decoder.weight",
    ],
)
def test_load_classifier_weights_orders_labels_by_row(tmp_path, key):
    weights = torch.arange(6.0).reshape(3, 2)
    path = tmp_path / "x.ckpt"
    torch.save(
        {
            "state_dict": {
                key: weights,
                "model.classifiers.ct.dense.weight": torch.ones(2, 2),
            },
            "hyper_parameters": {"label_dict": {"ct": {"b": 1, "c": 2, "a": 0}}},
        },
        path,
    )
    loaded, labels = alignment.load_classifier_weights(path, "ct")
    assert labels == ["a", "b", "c"]
    np.testing.assert_array_equal(loaded, weights.numpy())
    with pytest.raises(ValueError, match="No classifier weights"):
        alignment.load_classifier_weights(path, "absent")


def test_load_classifier_weights_rejects_label_mismatch(tmp_path):
    path = tmp_path / "x.ckpt"
    torch.save(
        {
            "state_dict": {"classifiers.ct.weight": torch.ones(3, 2)},
            "hyper_parameters": {"label_dict": {"ct": {"a": 0, "b": 1}}},
        },
        path,
    )
    with pytest.raises(ValueError, match="do not match"):
        alignment.load_classifier_weights(path, "ct")


def test_run_writes_alignment_for_each_source(ontology_path, probabilities, tmp_path):
    column = "cell_type_ontology_term_id"
    ckpt = tmp_path / "x.ckpt"
    torch.save(
        {
            "state_dict": {f"classifiers.{column}.weight": torch.eye(4) + 0.1},
            "hyper_parameters": {
                "label_dict": {column: {c: i for i, c in enumerate(LEAVES)}}
            },
        },
        ckpt,
    )
    config = OntologyConfig(
        "ct",
        ontology_path=str(ontology_path),
        thresholds=[0.3],
        topdown=True,
        sankey=None,
        greedy_flow=None,
        probability_graph=None,
        alignment=AlignmentConfig(
            bootstrap_reps=0, perm_reps=5, weight_checkpoint_path=str(ckpt)
        ),
    )
    obs = pd.DataFrame({"ct": LEAVES}, index=CELLS)
    embeddings = pd.DataFrame(PROBS, index=CELLS)
    results = run_ontology_analysis(config, obs, probabilities, tmp_path, embeddings)
    assert results.alignment["source"].tolist() == ["ground_truth", "weights"]
    assert (results.alignment["n_types"] == 4).all()
    assert (tmp_path / "alignment_pairs_weights.csv").exists()
    assert results.metrics["decoder"].tolist() == ["bottomup", "topdown"]
    with pytest.raises(ValueError, match="needs embeddings"):
        run_ontology_analysis(config, obs, probabilities, tmp_path)


def test_alignment_settings_are_accepted_as_a_mapping(ontology_path):
    callback = OntologyAnalysisCallback(
        {
            "ground_truth_column": "ct",
            "sankey": None,
            "greedy_flow": None,
            "probability_graph": None,
            "alignment": {"knn_k": 5},
        }
    )
    assert callback.config.alignment == AlignmentConfig(knn_k=5)
