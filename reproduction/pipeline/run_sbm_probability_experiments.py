import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder


EXPERIMENTS = {
    "id_pair": {
        "feature_columns": ["source", "target"],
        "feature_encoding": "categorical",
        "description": "Only node IDs are visible to the classifier.",
    },
    "id_pair_one_hot": {
        "feature_columns": ["source", "target"],
        "feature_encoding": "one_hot",
        "description": "Only node IDs are visible, one-hot encoded before TabPFN.",
    },
    "spectral_features": {
        "feature_columns": [
            "spectral_cluster_low",
            "spectral_cluster_high",
            "spectral_same_cluster",
            "spectral_dot",
            "spectral_l2_distance",
            "spectral_l1_distance",
            "spectral_absdiff_0",
            "spectral_absdiff_1",
            "spectral_absdiff_2",
            "spectral_absdiff_3",
        ],
        "feature_encoding": "mixed",
        "categorical_feature_columns": [
            "spectral_cluster_low",
            "spectral_cluster_high",
            "spectral_same_cluster",
        ],
        "requires_support_graph": True,
        "description": "Leakage-safe global spectral features from a held-out support graph.",
    },
    "block_pair": {
        "feature_columns": ["block_low", "block_high"],
        "feature_encoding": "categorical",
        "description": "True SBM block-pair metadata is visible to the classifier.",
    },
}
AVAILABLE_MODELS = ("empirical", "tabpfn")


def parse_models(value):
    models = value.split(",")
    unknown_models = sorted(set(models) - set(AVAILABLE_MODELS))
    if unknown_models:
        raise argparse.ArgumentTypeError(
            f"unknown models {unknown_models}; choose from {AVAILABLE_MODELS}"
        )
    return models


def parse_experiments(value):
    experiments = value.split(",")
    unknown_experiments = sorted(set(experiments) - set(EXPERIMENTS))
    if unknown_experiments:
        raise argparse.ArgumentTypeError(
            f"unknown experiments {unknown_experiments}; choose from {tuple(EXPERIMENTS)}"
        )
    return experiments


def load_pair_table(input_dir):
    pair_table_path = input_dir / "pair_table.csv"
    if not pair_table_path.exists():
        raise FileNotFoundError(
            f"{pair_table_path} does not exist; generate it with python3 sbm_gen.py"
        )
    return pd.read_csv(pair_table_path)


def make_train_test_split(pair_table, test_fraction, seed):
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")

    rng = np.random.default_rng(seed)
    indices = pair_table.index.to_numpy(copy=True)
    rng.shuffle(indices)

    test_size = round(len(indices) * test_fraction)
    test_size = min(max(test_size, 1), len(indices) - 1)
    test_indices = indices[:test_size]

    is_test = pair_table.index.isin(test_indices)
    train = pair_table.loc[~is_test].sample(frac=1.0, random_state=seed)
    test = pair_table.loc[is_test].sample(frac=1.0, random_state=seed)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def apply_context_size(train, context_size, seed):
    if context_size is None:
        return train
    if context_size < 1:
        raise ValueError("context_size must be positive")
    if context_size > len(train):
        raise ValueError(f"context_size {context_size} exceeds train rows {len(train)}")
    return train.sample(n=context_size, random_state=seed).reset_index(drop=True)


def selected_experiments_require_support_graph(args):
    return any(
        EXPERIMENTS[experiment_name].get("requires_support_graph", False)
        for experiment_name in args.experiments
    )


def split_support_and_supervised_train(train, support_fraction, seed):
    if not 0.0 < support_fraction < 1.0:
        raise ValueError("support_fraction must be between 0 and 1")

    rng = np.random.default_rng(seed + 30_000)
    indices = train.index.to_numpy(copy=True)
    rng.shuffle(indices)
    support_size = round(len(indices) * support_fraction)
    support_size = min(max(support_size, 1), len(indices) - 1)
    support_indices = indices[:support_size]

    is_support = train.index.isin(support_indices)
    support = train.loc[is_support].sample(frac=1.0, random_state=seed + 30_001)
    supervised_train = train.loc[~is_support].sample(
        frac=1.0,
        random_state=seed + 30_002,
    )

    if supervised_train["edge"].nunique() != 2:
        raise ValueError("supervised train split must contain both edge classes")

    return support.reset_index(drop=True), supervised_train.reset_index(drop=True)


def build_support_adjacency(support, frames):
    support_edges = support.loc[support["edge"] == 1, ["source", "target"]]
    nodes = pd.concat(
        [series for frame in frames for series in (frame["source"], frame["target"])],
        ignore_index=True,
    )
    nodes = sorted(int(node) for node in nodes.unique())
    node_to_index = {node: index for index, node in enumerate(nodes)}
    adjacency = np.zeros((len(nodes), len(nodes)), dtype=np.float64)

    for source, target in support_edges.itertuples(index=False):
        source = int(source)
        target = int(target)
        source_index = node_to_index[source]
        target_index = node_to_index[target]
        adjacency[source_index, target_index] = 1.0
        adjacency[target_index, source_index] = 1.0

    return nodes, node_to_index, adjacency


def spectral_node_features(support, frames, num_components, num_clusters, seed):
    nodes, node_to_index, adjacency = build_support_adjacency(support, frames)
    degrees = adjacency.sum(axis=1)
    inv_sqrt_degrees = np.zeros_like(degrees)
    positive_degree = degrees > 0
    inv_sqrt_degrees[positive_degree] = 1.0 / np.sqrt(degrees[positive_degree])
    normalized_adjacency = (
        inv_sqrt_degrees[:, None] * adjacency * inv_sqrt_degrees[None, :]
    )

    eigenvalues, eigenvectors = np.linalg.eigh(normalized_adjacency)
    order = np.argsort(eigenvalues)[::-1]
    components = min(num_components, len(nodes))
    embedding = eigenvectors[:, order[:components]] * eigenvalues[order[:components]]
    if components < num_components:
        embedding = np.pad(
            embedding,
            ((0, 0), (0, num_components - components)),
            mode="constant",
        )

    row_norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    normalized_embedding = np.divide(
        embedding,
        row_norms,
        out=np.zeros_like(embedding),
        where=row_norms > 0,
    )

    clusters = KMeans(
        n_clusters=min(num_clusters, len(nodes)),
        n_init=20,
        random_state=seed,
    ).fit_predict(normalized_embedding)

    return {
        node: {
            "embedding": normalized_embedding[index],
            "cluster": int(clusters[index]),
            "degree": float(degrees[index]),
        }
        for node, index in node_to_index.items()
    }


def add_spectral_features(frame, node_features, num_components):
    rows = []
    zero_embedding = np.zeros(num_components, dtype=np.float64)

    for source, target in frame[["source", "target"]].itertuples(index=False):
        source = int(source)
        target = int(target)
        source_features = node_features.get(
            source,
            {
                "embedding": zero_embedding,
                "cluster": -1,
                "degree": 0.0,
            },
        )
        target_features = node_features.get(
            target,
            {
                "embedding": zero_embedding,
                "cluster": -1,
                "degree": 0.0,
            },
        )
        source_embedding = source_features["embedding"]
        target_embedding = target_features["embedding"]
        absdiff = np.abs(source_embedding - target_embedding)
        source_cluster = source_features["cluster"]
        target_cluster = target_features["cluster"]
        cluster_low, cluster_high = sorted((source_cluster, target_cluster))

        row = {
            "spectral_cluster_source": source_cluster,
            "spectral_cluster_target": target_cluster,
            "spectral_cluster_low": cluster_low,
            "spectral_cluster_high": cluster_high,
            "spectral_same_cluster": int(source_cluster == target_cluster),
            "spectral_dot": float(np.dot(source_embedding, target_embedding)),
            "spectral_l2_distance": float(
                np.linalg.norm(source_embedding - target_embedding)
            ),
            "spectral_l1_distance": float(np.abs(source_embedding - target_embedding).sum()),
            "spectral_degree_source": source_features["degree"],
            "spectral_degree_target": target_features["degree"],
        }
        for component_index in range(num_components):
            row[f"spectral_source_{component_index}"] = float(
                source_embedding[component_index]
            )
            row[f"spectral_target_{component_index}"] = float(
                target_embedding[component_index]
            )
            row[f"spectral_absdiff_{component_index}"] = float(
                absdiff[component_index]
            )
            row[f"spectral_product_{component_index}"] = float(
                source_embedding[component_index] * target_embedding[component_index]
            )
        rows.append(row)

    feature_frame = pd.DataFrame(rows)
    return pd.concat([frame.reset_index(drop=True), feature_frame], axis=1)


def positive_class_probabilities(model, features):
    positive_class_indices = np.flatnonzero(model.classes_ == 1)
    if len(positive_class_indices) != 1:
        raise ValueError(f"expected class 1, found classes {model.classes_}")
    return model.predict_proba(features)[:, positive_class_indices.item()]


def build_tabpfn_inference_config(max_unique_for_categorical_features):
    from tabpfn.inference_config import InferenceConfig

    annotations = getattr(InferenceConfig, "__annotations__", {})
    if "MAX_UNIQUE_FOR_CATEGORICAL_FEATURES" not in annotations:
        raise RuntimeError(
            "This installed TabPFN version does not expose "
            "MAX_UNIQUE_FOR_CATEGORICAL_FEATURES."
        )

    return {
        "MAX_UNIQUE_FOR_CATEGORICAL_FEATURES": max_unique_for_categorical_features,
    }


def max_train_cardinality(train, feature_columns):
    return max(int(train[column].nunique()) for column in feature_columns)


def make_one_hot_encoder():
    try:
        return OneHotEncoder(
            categories="auto",
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float32,
        )
    except TypeError:
        return OneHotEncoder(
            categories="auto",
            handle_unknown="ignore",
            sparse=False,
            dtype=np.float32,
        )


def unseen_category_rate(train, test, feature_columns):
    unseen_mask = np.zeros(len(test), dtype=bool)
    for column in feature_columns:
        train_values = set(train[column].unique())
        unseen_mask |= ~test[column].isin(train_values).to_numpy()
    return float(unseen_mask.mean())


def prepare_tabpfn_features(train, test, experiment, args):
    feature_columns = experiment["feature_columns"]
    feature_encoding = experiment["feature_encoding"]

    if feature_encoding == "categorical":
        categorical_limit = (
            args.max_unique_for_categorical_features
            or max_train_cardinality(train, feature_columns)
        )
        model_args = {
            "categorical_features_indices": list(range(len(feature_columns))),
            "inference_config": build_tabpfn_inference_config(categorical_limit),
        }
        diagnostics = {
            "model_feature_count": len(feature_columns),
            "unseen_feature_rate": unseen_category_rate(train, test, feature_columns),
        }
        return train[feature_columns], test[feature_columns], model_args, diagnostics

    if feature_encoding == "one_hot":
        encoder = make_one_hot_encoder()
        train_features = encoder.fit_transform(train[feature_columns]).astype(np.float32)
        test_features = encoder.transform(test[feature_columns]).astype(np.float32)
        diagnostics = {
            "model_feature_count": int(train_features.shape[1]),
            "unseen_feature_rate": unseen_category_rate(train, test, feature_columns),
        }
        return train_features, test_features, {}, diagnostics

    if feature_encoding == "numeric":
        train_features = train[feature_columns].astype(np.float32).to_numpy()
        test_features = test[feature_columns].astype(np.float32).to_numpy()
        diagnostics = {
            "model_feature_count": int(train_features.shape[1]),
            "unseen_feature_rate": 0.0,
        }
        return train_features, test_features, {}, diagnostics

    if feature_encoding == "mixed":
        categorical_columns = experiment.get("categorical_feature_columns", [])
        categorical_indices = [
            feature_columns.index(column)
            for column in categorical_columns
            if column in feature_columns
        ]
        categorical_limit = (
            args.max_unique_for_categorical_features
            or max_train_cardinality(train, categorical_columns)
            if categorical_columns
            else None
        )
        model_args = {}
        if categorical_indices:
            model_args["categorical_features_indices"] = categorical_indices
            model_args["inference_config"] = build_tabpfn_inference_config(
                categorical_limit
            )
        train_features = train[feature_columns].astype(np.float32)
        test_features = test[feature_columns].astype(np.float32)
        diagnostics = {
            "model_feature_count": len(feature_columns),
            "unseen_feature_rate": (
                unseen_category_rate(train, test, categorical_columns)
                if categorical_columns
                else 0.0
            ),
        }
        return train_features, test_features, model_args, diagnostics

    raise ValueError(f"unsupported feature encoding: {feature_encoding}")


def predict_empirical(train, test, feature_columns):
    global_probability = (train["edge"].sum() + 1.0) / (len(train) + 2.0)
    probabilities = (
        train.groupby(feature_columns)["edge"]
        .agg(["sum", "count"])
        .assign(probability=lambda frame: (frame["sum"] + 1.0) / (frame["count"] + 2.0))
        ["probability"]
    )

    test_index = pd.MultiIndex.from_frame(test[feature_columns])
    aligned = probabilities.reindex(test_index)
    unseen_feature_rate = float(aligned.isna().mean())
    diagnostics = {
        "model_feature_count": len(feature_columns),
        "unseen_feature_rate": unseen_feature_rate,
    }
    return aligned.fillna(global_probability).to_numpy(), diagnostics


def predict_tabpfn(train, test, experiment, args):
    from tabpfn import TabPFNClassifier

    if train["edge"].nunique() != 2:
        raise ValueError("TabPFN train context must contain both edge classes")

    train_features, test_features, feature_model_args, diagnostics = prepare_tabpfn_features(
        train, test, experiment, args
    )
    model_args = {
        "device": args.device,
        "n_estimators": args.n_estimators,
        "random_state": args.seed,
    }
    model_args.update(feature_model_args)
    if args.checkpoint_name is not None:
        model_args["model_path"] = Path(args.model_dir) / args.checkpoint_name

    model = TabPFNClassifier(**model_args)
    model.fit(train_features, train["edge"])
    return positive_class_probabilities(model, test_features), diagnostics


def calculate_probability_metrics(test, probabilities):
    oracle = test["oracle_probability"].to_numpy(dtype=float)
    labels = test["edge"].to_numpy(dtype=int)
    clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    same_block = test["same_block"].to_numpy(dtype=bool)

    metrics = {
        "rows": int(len(test)),
        "oracle_mse": float(np.mean((probabilities - oracle) ** 2)),
        "oracle_mae": float(np.mean(np.abs(probabilities - oracle))),
        "mean_predicted_probability": float(np.mean(probabilities)),
        "mean_oracle_probability": float(np.mean(oracle)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(log_loss(labels, clipped, labels=[0, 1])),
        "within_predicted_probability": float(np.mean(probabilities[same_block])),
        "within_oracle_probability": float(np.mean(oracle[same_block])),
        "between_predicted_probability": float(np.mean(probabilities[~same_block])),
        "between_oracle_probability": float(np.mean(oracle[~same_block])),
        "within_rows": int(same_block.sum()),
        "between_rows": int((~same_block).sum()),
    }
    return metrics


def build_prediction_frame(test, feature_columns, probabilities):
    columns = [
        "source",
        "target",
        "source_block",
        "target_block",
        "block_low",
        "block_high",
        "same_block",
        "oracle_probability",
        "edge",
    ]
    columns = list(dict.fromkeys(columns + feature_columns))
    predictions = test[columns].copy()
    predictions["predicted_probability"] = probabilities
    predictions["probability_error"] = (
        predictions["predicted_probability"] - predictions["oracle_probability"]
    )
    return predictions


def run_experiments(args):
    pair_table = load_pair_table(args.input_dir)
    train, test = make_train_test_split(pair_table, args.test_fraction, args.seed)
    support = pd.DataFrame(columns=train.columns)
    if selected_experiments_require_support_graph(args):
        support, train = split_support_and_supervised_train(
            train,
            args.support_fraction,
            args.seed,
        )
        node_features = spectral_node_features(
            support,
            [support, train, test],
            args.num_spectral_components,
            args.num_spectral_clusters,
            args.seed,
        )
        train = add_spectral_features(train, node_features, args.num_spectral_components)
        test = add_spectral_features(test, node_features, args.num_spectral_components)

    train = apply_context_size(train, args.context_size, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if selected_experiments_require_support_graph(args):
        support.to_csv(args.output_dir / "support_pairs.csv", index=False)
    train.to_csv(args.output_dir / "train_pairs.csv", index=False)
    test.to_csv(args.output_dir / "test_pairs.csv", index=False)

    results = []
    for experiment_name in args.experiments:
        experiment = EXPERIMENTS[experiment_name]
        feature_columns = experiment["feature_columns"]
        for model_name in args.models:
            print(
                f"experiment={experiment_name} model={model_name} "
                f"train_rows={len(train)} test_rows={len(test)}",
                flush=True,
            )
            if model_name == "empirical":
                probabilities, diagnostics = predict_empirical(
                    train, test, feature_columns
                )
            elif model_name == "tabpfn":
                probabilities, diagnostics = predict_tabpfn(
                    train, test, experiment, args
                )
            else:
                raise ValueError(f"unsupported model {model_name}")

            prediction_frame = build_prediction_frame(test, feature_columns, probabilities)
            prediction_path = (
                args.output_dir / f"{experiment_name}_{model_name}_predictions.csv"
            )
            prediction_frame.to_csv(prediction_path, index=False)

            metrics = calculate_probability_metrics(test, probabilities)
            metrics.update(
                {
                    "experiment": experiment_name,
                    "model": model_name,
                    "feature_columns": feature_columns,
                    "feature_encoding": experiment["feature_encoding"],
                    "model_feature_count": diagnostics["model_feature_count"],
                    "unseen_feature_rate": diagnostics["unseen_feature_rate"],
                    "prediction_path": str(prediction_path),
                }
            )
            results.append(metrics)

    return support, train, test, pd.DataFrame(results)


def write_outputs(args, support, train, test, results):
    results_path = args.output_dir / "metrics.csv"
    config_path = args.output_dir / "config.json"

    results.to_csv(results_path, index=False)
    config = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "test_fraction": args.test_fraction,
        "support_fraction": (
            args.support_fraction
            if selected_experiments_require_support_graph(args)
            else None
        ),
        "context_size": args.context_size,
        "seed": args.seed,
        "experiments": {
            name: EXPERIMENTS[name] for name in args.experiments
        },
        "models": args.models,
        "train_rows": len(train),
        "support_rows": len(support),
        "support_positive_edges": int(support["edge"].sum()) if len(support) else 0,
        "test_rows": len(test),
        "num_spectral_components": (
            args.num_spectral_components
            if selected_experiments_require_support_graph(args)
            else None
        ),
        "num_spectral_clusters": (
            args.num_spectral_clusters
            if selected_experiments_require_support_graph(args)
            else None
        ),
        "device": args.device,
        "n_estimators": args.n_estimators,
        "max_unique_for_categorical_features": args.max_unique_for_categorical_features,
        "model_path": (
            str(Path(args.model_dir) / args.checkpoint_name)
            if args.checkpoint_name is not None
            else None
        ),
    }
    with config_path.open("w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2)

    print(f"Wrote {results_path}")
    print(f"Wrote {config_path}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run probability-focused SBM link prediction experiments."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data/sbm_current"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/sbm_probability_experiments/current"),
    )
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument(
        "--support-fraction",
        type=float,
        default=0.5,
        help=(
            "Fraction of non-test pairs reserved for the support graph when an "
            "experiment requires graph-derived features."
        ),
    )
    parser.add_argument("--num-spectral-components", type=int, default=4)
    parser.add_argument("--num-spectral-clusters", type=int, default=4)
    parser.add_argument("--context-size", type=int)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument(
        "--experiments",
        type=parse_experiments,
        default=list(EXPERIMENTS),
        help=f"comma-separated subset of {tuple(EXPERIMENTS)}",
    )
    parser.add_argument(
        "--models",
        type=parse_models,
        default=["empirical"],
        help=f"comma-separated subset of {AVAILABLE_MODELS}",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument(
        "--max-unique-for-categorical-features",
        type=int,
        help=(
            "TabPFN categorical-cardinality limit. Defaults to the largest "
            "train cardinality among the selected feature columns."
        ),
    )
    parser.add_argument("--model-dir", default="/home/alessandrocagiano/data/models_ckpt/")
    parser.add_argument("--checkpoint-name")
    return parser


def main():
    args = build_parser().parse_args()
    support, train, test, results = run_experiments(args)
    write_outputs(args, support, train, test, results)
    print(results[["experiment", "model", "oracle_mse", "oracle_mae"]])


if __name__ == "__main__":
    main()
