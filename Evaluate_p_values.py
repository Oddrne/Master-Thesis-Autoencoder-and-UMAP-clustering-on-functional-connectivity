import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from Evaluate_clusterings_with_CCA import permutation_test_cca_cv
from Evaluate_clusterings_with_MLR import permutation_test_mlr_cv


def create_permutation_tests_for_selected_variables_json(
    data: pd.DataFrame,
    input_json_path: str | Path,
    output_json_path: str | Path,
    cluster_col_template: str = "Cluster_{k}",
    selected_variable_keys: tuple[str, ...] | str = "auto",
    tests: tuple[str, ...] = ("cca", "mlr"),
    cv_splits: int = 5,
    random_state: int = 42,
    n_permutations: int = 1000,
    cca_score_key: str = "cv_mean_cc",
    mlr_score_key: str = "mean_accuracy",
    alternative: str = "greater",
    store_null_scores: bool = False,
    print_results: bool = True,
):
    """
    Run permutation tests for every saved selected-variable list in a JSON file.

    This can test:
        - CCA on CCA-selected variables
        - CCA on MLR-selected variables
        - MLR on CCA-selected variables
        - MLR on MLR-selected variables

    depending on `selected_variable_keys` and `tests`.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing cluster columns and behavioural variables.

    input_json_path : str | Path
        Existing JSON file containing selected variable lists.

    output_json_path : str | Path
        New JSON file where permutation results will be saved.

    cluster_col_template : str
        Template for the cluster columns in `data`.
        Default assumes Cluster_2, Cluster_3, ..., Cluster_10.

    selected_variable_keys : tuple[str, ...] | str
        Which selected-variable lists to use.

        Common options:
            ("cca_selected_variables", "mlr_selected_variables")

        If "auto", the function detects top-level keys inside each
        Cluster_X_results block that end with "_selected_variables".

    tests : tuple[str, ...]
        Which permutation tests to run.
        Options:
            ("cca",)
            ("mlr",)
            ("cca", "mlr")

    cv_splits : int
        Requested number of CV folds.

    random_state : int
        Random seed.

    n_permutations : int
        Number of label permutations.

    cca_score_key : str
        Score used for CCA p-value.
        Recommended: "cv_mean_cc".

    mlr_score_key : str
        Score used for MLR p-value.
        Recommended: "mean_accuracy".

    alternative : str
        Usually "greater".

    store_null_scores : bool
        If True, saves all null scores.
        This can make the JSON large.

    print_results : bool
        Whether to print progress.

    Returns
    -------
    permutation_results : dict
        JSON-compatible results dictionary.

    summary_df : pd.DataFrame
        Compact summary table.
    """

    input_json_path = Path(input_json_path)
    output_json_path = Path(output_json_path)

    if input_json_path.resolve() == output_json_path.resolve():
        raise ValueError(
            "input_json_path and output_json_path are the same. "
            "Choose a different output path."
        )

    with input_json_path.open("r", encoding="utf-8") as f:
        loaded_json = json.load(f)

    # Support both:
    #   {"Cluster_2_results": ...}
    # and:
    #   {"results": {"Cluster_2_results": ...}}
    if isinstance(loaded_json, dict) and "results" in loaded_json:
        cluster_results_container = loaded_json["results"]
    else:
        cluster_results_container = loaded_json

    def _cluster_number_from_key(key: str) -> int | None:
        match = re.search(r"Cluster_(\d+)_results", key)
        if match is None:
            return None
        return int(match.group(1))

    def _json_safe(obj):
        if isinstance(obj, dict):
            return {str(k): _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_json_safe(v) for v in obj]
        if isinstance(obj, tuple):
            return [_json_safe(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return _json_safe(obj.tolist())
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            if np.isnan(obj):
                return None
            return float(obj)
        if isinstance(obj, float) and np.isnan(obj):
            return None
        return obj

    def _remove_null_scores_if_needed(result: dict):
        if not store_null_scores and isinstance(result, dict):
            result = result.copy()
            result.pop("null_scores", None)
        return result

    def _get_selected_variable_keys(cluster_result: dict) -> list[str]:
        if selected_variable_keys == "auto":
            keys = []
            for key, value in cluster_result.items():
                if (
                    key.endswith("_selected_variables")
                    and isinstance(value, list)
                    and all(isinstance(v, str) for v in value)
                ):
                    keys.append(key)
            return keys

        return list(selected_variable_keys)

    permutation_results = {
        "source_json": str(input_json_path),
        "analysis": "Permutation tests for saved selected-variable sets",
        "cluster_col_template": cluster_col_template,
        "selected_variable_keys": selected_variable_keys,
        "tests": list(tests),
        "cv_splits_requested": cv_splits,
        "random_state": random_state,
        "n_permutations": n_permutations,
        "cca_score_key": cca_score_key,
        "mlr_score_key": mlr_score_key,
        "alternative": alternative,
        "store_null_scores": store_null_scores,
        "results": {},
    }

    summary_rows = []

    cluster_items = []

    for cluster_key, cluster_result in cluster_results_container.items():
        k = _cluster_number_from_key(cluster_key)
        if k is not None:
            cluster_items.append((k, cluster_key, cluster_result))

    cluster_items = sorted(cluster_items, key=lambda x: x[0])

    for k, cluster_key, cluster_result in cluster_items:
        cluster_col = cluster_col_template.format(k=k)

        cluster_output = {
            "cluster_number": k,
            "cluster_col": cluster_col,
            "selected_variable_tests": {},
        }

        if print_results:
            print(f"\n=== Permutation tests for {cluster_col} ===")

        if cluster_col not in data.columns:
            error_msg = f"Cluster column '{cluster_col}' not found in data."

            cluster_output["error"] = error_msg
            permutation_results["results"][cluster_key] = cluster_output

            summary_rows.append({
                "cluster_number": k,
                "cluster_col": cluster_col,
                "selected_variables_key": None,
                "test": None,
                "observed_score": np.nan,
                "p_value": np.nan,
                "null_mean": np.nan,
                "null_std": np.nan,
                "n_valid_permutations": 0,
                "variables_used": [],
                "error": error_msg,
            })

            if print_results:
                print(f"Skipping: {error_msg}")

            continue

        n_unique = data[cluster_col].dropna().nunique()

        if n_unique < 2:
            error_msg = f"Need at least two unique clusters, found {n_unique}."

            cluster_output["error"] = error_msg
            permutation_results["results"][cluster_key] = cluster_output

            summary_rows.append({
                "cluster_number": k,
                "cluster_col": cluster_col,
                "selected_variables_key": None,
                "test": None,
                "observed_score": np.nan,
                "p_value": np.nan,
                "null_mean": np.nan,
                "null_std": np.nan,
                "n_valid_permutations": 0,
                "variables_used": [],
                "error": error_msg,
            })

            if print_results:
                print(f"Skipping: {error_msg}")

            continue

        cluster_counts = data[cluster_col].dropna().value_counts()
        min_cluster_size = int(cluster_counts.min())

        if min_cluster_size < 2:
            error_msg = (
                f"Smallest cluster has fewer than 2 subjects "
                f"(min_cluster_size={min_cluster_size})."
            )

            cluster_output["error"] = error_msg
            permutation_results["results"][cluster_key] = cluster_output

            summary_rows.append({
                "cluster_number": k,
                "cluster_col": cluster_col,
                "selected_variables_key": None,
                "test": None,
                "observed_score": np.nan,
                "p_value": np.nan,
                "null_mean": np.nan,
                "null_std": np.nan,
                "n_valid_permutations": 0,
                "variables_used": [],
                "error": error_msg,
            })

            if print_results:
                print(f"Skipping: {error_msg}")

            continue

        cv_splits_this = min(cv_splits, min_cluster_size)

        selected_keys_for_cluster = _get_selected_variable_keys(cluster_result)

        if not selected_keys_for_cluster:
            if print_results:
                print("No selected-variable keys found for this cluster.")

        for selected_key in selected_keys_for_cluster:
            selected_variables = cluster_result.get(selected_key, [])
            selected_variables = list(dict.fromkeys(selected_variables))

            missing_variables = [
                var for var in selected_variables
                if var not in data.columns
            ]

            available_variables = [
                var for var in selected_variables
                if var in data.columns
            ]

            non_numeric_variables = [
                var for var in available_variables
                if not pd.api.types.is_numeric_dtype(data[var])
            ]

            numeric_variables = [
                var for var in available_variables
                if pd.api.types.is_numeric_dtype(data[var])
            ]

            selection_output = {
                "selected_variables": selected_variables,
                "missing_variables": missing_variables,
                "non_numeric_variables": non_numeric_variables,
                "variables_tested": numeric_variables,
                "cv_splits_used": cv_splits_this,
                "permutation_tests": {},
            }

            if print_results:
                print(f"\nSelected-variable set: {selected_key}")
                print(f"Variables: {selected_variables}")

            if not numeric_variables:
                error_msg = "No usable numeric variables found."

                selection_output["error"] = error_msg

                summary_rows.append({
                    "cluster_number": k,
                    "cluster_col": cluster_col,
                    "selected_variables_key": selected_key,
                    "test": None,
                    "observed_score": np.nan,
                    "p_value": np.nan,
                    "null_mean": np.nan,
                    "null_std": np.nan,
                    "n_valid_permutations": 0,
                    "variables_used": [],
                    "error": error_msg,
                })

                cluster_output["selected_variable_tests"][selected_key] = selection_output
                continue

            if "cca" in tests:
                try:
                    cca_perm = permutation_test_cca_cv(
                        data=data,
                        cluster_col=cluster_col,
                        variable_cols=numeric_variables,
                        cv_splits=cv_splits_this,
                        random_state=random_state,
                        n_permutations=n_permutations,
                        score_key=cca_score_key,
                        alternative=alternative,
                        print_results=False,
                    )

                    cca_perm = _remove_null_scores_if_needed(cca_perm)

                    selection_output["permutation_tests"]["cca"] = cca_perm

                    summary_rows.append({
                        "cluster_number": k,
                        "cluster_col": cluster_col,
                        "selected_variables_key": selected_key,
                        "test": "cca",
                        "tested_score": cca_score_key,
                        "observed_score": cca_perm["observed_score"],
                        "p_value": cca_perm["p_value"],
                        "null_mean": cca_perm["null_mean"],
                        "null_std": cca_perm["null_std"],
                        "n_valid_permutations": cca_perm["n_valid_permutations"],
                        "variables_used": cca_perm["observed_result"]["variables_used"],
                        "error": None,
                    })

                    if print_results:
                        print(
                            f"CCA: observed={cca_perm['observed_score']:.4f}, "
                            f"p={cca_perm['p_value']:.4f}"
                        )

                except Exception as e:
                    error_msg = str(e)

                    selection_output["permutation_tests"]["cca"] = {
                        "error": error_msg,
                        "p_value": None,
                    }

                    summary_rows.append({
                        "cluster_number": k,
                        "cluster_col": cluster_col,
                        "selected_variables_key": selected_key,
                        "test": "cca",
                        "tested_score": cca_score_key,
                        "observed_score": np.nan,
                        "p_value": np.nan,
                        "null_mean": np.nan,
                        "null_std": np.nan,
                        "n_valid_permutations": 0,
                        "variables_used": [],
                        "error": error_msg,
                    })

                    if print_results:
                        print(f"CCA failed: {error_msg}")

            if "mlr" in tests:
                try:
                    mlr_perm = permutation_test_mlr_cv(
                        data=data,
                        cluster_col=cluster_col,
                        variable_cols=numeric_variables,
                        cv_splits=cv_splits_this,
                        random_state=random_state,
                        n_permutations=n_permutations,
                        score_key=mlr_score_key,
                        alternative=alternative,
                        print_results=False,
                    )

                    mlr_perm = _remove_null_scores_if_needed(mlr_perm)

                    selection_output["permutation_tests"]["mlr"] = mlr_perm

                    summary_rows.append({
                        "cluster_number": k,
                        "cluster_col": cluster_col,
                        "selected_variables_key": selected_key,
                        "test": "mlr",
                        "tested_score": mlr_score_key,
                        "observed_score": mlr_perm["observed_score"],
                        "p_value": mlr_perm["p_value"],
                        "null_mean": mlr_perm["null_mean"],
                        "null_std": mlr_perm["null_std"],
                        "n_valid_permutations": mlr_perm["n_valid_permutations"],
                        "variables_used": mlr_perm["observed_result"]["variables_used"],
                        "error": None,
                    })

                    if print_results:
                        print(
                            f"MLR: observed={mlr_perm['observed_score']:.4f}, "
                            f"p={mlr_perm['p_value']:.4f}"
                        )

                except Exception as e:
                    error_msg = str(e)

                    selection_output["permutation_tests"]["mlr"] = {
                        "error": error_msg,
                        "p_value": None,
                    }

                    summary_rows.append({
                        "cluster_number": k,
                        "cluster_col": cluster_col,
                        "selected_variables_key": selected_key,
                        "test": "mlr",
                        "tested_score": mlr_score_key,
                        "observed_score": np.nan,
                        "p_value": np.nan,
                        "null_mean": np.nan,
                        "null_std": np.nan,
                        "n_valid_permutations": 0,
                        "variables_used": [],
                        "error": error_msg,
                    })

                    if print_results:
                        print(f"MLR failed: {error_msg}")

            cluster_output["selected_variable_tests"][selected_key] = selection_output

        permutation_results["results"][cluster_key] = cluster_output

    summary_df = pd.DataFrame(summary_rows)

    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(permutation_results), f, indent=4)

    if print_results:
        print(f"\nSaved permutation results to: {output_json_path}")

    return permutation_results, summary_df

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _cluster_number_from_key(key: str):
    match = re.search(r"Cluster_(\d+)_results", key)
    if match is None:
        return None
    return int(match.group(1))


def _load_cluster_results(json_path):
    """
    Supports both:
        {"Cluster_2_results": {...}}
    and:
        {"results": {"Cluster_2_results": {...}}}
    """
    json_path = Path(json_path)

    with json_path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)

    if isinstance(loaded, dict) and "results" in loaded:
        return loaded["results"]

    return loaded


def extract_cca_pvalue_results_from_json(
    json_path,
    file_label=None,
    selected_variable_key=None,
    direct_result_keys=None,
):
    """
    Extract CCA value and p-value from one JSON file.

    Parameters
    ----------
    json_path : str | Path
        Path to JSON file.

    file_label : str | None
        Label used in the plot. If None, the filename stem is used.

    selected_variable_key : str | None
        Relevant for permutation-test JSONs.

        Examples:
            "cca_selected_variables"
            "mlr_selected_variables"

        If None, the function extracts all selected-variable sets that contain
        a CCA permutation test.

    direct_result_keys : list[str] | None
        Relevant for simpler JSONs where CCA results are directly stored inside
        each Cluster_X_results block.

    Returns
    -------
    pd.DataFrame
        Columns:
            file_label
            cluster_number
            selected_variable_key
            result_source
            cca_value
            p_value
    """

    json_path = Path(json_path)

    if file_label is None:
        file_label = json_path.stem

    if direct_result_keys is None:
        direct_result_keys = [
            "cca_mlr_selected_variables_results",
            "cca_selected_variables_results",
            "cca_removed_subjects_results",
            "cca_all_variables_results",
        ]

    cluster_results = _load_cluster_results(json_path)

    rows = []

    for cluster_key, cluster_result in cluster_results.items():
        k = _cluster_number_from_key(cluster_key)

        if k is None or not isinstance(cluster_result, dict):
            continue

        # ----------------------------------------------------
        # Case 1: permutation wrapper format
        # ----------------------------------------------------
        if "selected_variable_tests" in cluster_result:
            selected_tests = cluster_result.get("selected_variable_tests", {})

            for sel_key, sel_result in selected_tests.items():
                if selected_variable_key is not None and sel_key != selected_variable_key:
                    continue

                if not isinstance(sel_result, dict):
                    continue

                permutation_tests = sel_result.get("permutation_tests", {})
                cca_result = permutation_tests.get("cca", None)

                if not isinstance(cca_result, dict):
                    continue

                cca_value = cca_result.get("observed_score", None)
                p_value = cca_result.get("p_value", None)

                rows.append({
                    "file_label": file_label,
                    "cluster_number": k,
                    "selected_variable_key": sel_key,
                    "result_source": "permutation_tests.cca",
                    "cca_value": cca_value,
                    "p_value": p_value,
                })

        # ----------------------------------------------------
        # Case 2: direct CCA-result format
        # ----------------------------------------------------
        for result_key in direct_result_keys:
            if result_key not in cluster_result:
                continue

            cca_result = cluster_result.get(result_key, {})

            if not isinstance(cca_result, dict):
                continue

            # Prefer observed_score if it exists, otherwise use cv_mean_cc.
            # Fall back to cc if only in-sample CCA is available.
            cca_value = (
                cca_result.get("observed_score", None)
                if cca_result.get("observed_score", None) is not None
                else cca_result.get("cv_mean_cc", cca_result.get("cc", None))
            )

            p_value = cca_result.get("p_value", None)

            rows.append({
                "file_label": file_label,
                "cluster_number": k,
                "selected_variable_key": None,
                "result_source": result_key,
                "cca_value": cca_value,
                "p_value": p_value,
            })

    df = pd.DataFrame(rows)

    if not df.empty:
        df["cca_value"] = pd.to_numeric(df["cca_value"], errors="coerce")
        df["p_value"] = pd.to_numeric(df["p_value"], errors="coerce")
        df = df.sort_values(["file_label", "selected_variable_key", "result_source", "cluster_number"])

    return df


def plot_cca_with_pvalue_markers_from_jsons(
    json_paths,
    labels=None,
    selected_variable_key=None,
    direct_result_keys=None,
    p_threshold=0.05,
    min_marker_size=40,
    max_marker_size=350,
    annotate_pvalues=True,
    title=None,
    figsize=(12, 6),
    save_path=None,
    show=True,
):
    """
    Plot CCA values across cluster numbers, with permutation p-values encoded
    directly in the marker size.

    This avoids using two y-axes. Instead:
        x-axis = cluster number
        y-axis = observed CCA value
        marker size = -log10(p-value)
        marker text = p-value, optionally

    Larger markers mean stronger evidence against the permutation null.
    """

    json_paths = [Path(p) for p in json_paths]

    if labels is None:
        labels = [p.stem for p in json_paths]

    if len(labels) != len(json_paths):
        raise ValueError("labels must have the same length as json_paths.")

    dfs = []

    for json_path, label in zip(json_paths, labels):
        df = extract_cca_pvalue_results_from_json(
            json_path=json_path,
            file_label=label,
            selected_variable_key=selected_variable_key,
            direct_result_keys=direct_result_keys,
        )

        if not df.empty:
            dfs.append(df)

    if not dfs:
        raise ValueError("No CCA/p-value results found in the provided JSON files.")

    all_results_df = pd.concat(dfs, ignore_index=True)

    all_results_df["cca_value"] = pd.to_numeric(
        all_results_df["cca_value"],
        errors="coerce"
    )

    all_results_df["p_value"] = pd.to_numeric(
        all_results_df["p_value"],
        errors="coerce"
    )

    all_results_df = all_results_df.dropna(
        subset=["cluster_number", "cca_value"]
    ).copy()

    if all_results_df.empty:
        raise ValueError("No valid CCA values found after cleaning.")

    # p-values may be missing for non-permutation JSONs.
    # These get the smallest marker size.
    safe_p = all_results_df["p_value"].copy()
    safe_p = safe_p.fillna(1.0)
    safe_p = safe_p.clip(lower=1e-300, upper=1.0)

    all_results_df["neg_log10_p"] = -np.log10(safe_p)

    max_sig = all_results_df["neg_log10_p"].max()

    if max_sig > 0:
        all_results_df["marker_size"] = (
            min_marker_size
            + (all_results_df["neg_log10_p"] / max_sig)
            * (max_marker_size - min_marker_size)
        )
    else:
        all_results_df["marker_size"] = min_marker_size

    fig, ax = plt.subplots(figsize=figsize)

    group_cols = ["file_label", "selected_variable_key", "result_source"]

    grouped = list(all_results_df.groupby(group_cols, dropna=False))

    # Small horizontal offsets avoid complete overlap when several files have
    # results for the same cluster number.
    n_groups = len(grouped)

    if n_groups == 1:
        offsets = [0.0]
    else:
        offsets = np.linspace(-0.18, 0.18, n_groups)

    for offset, (group_values, group_df) in zip(offsets, grouped):
        file_label, sel_key, result_source = group_values

        group_df = group_df.sort_values("cluster_number").copy()
        x = group_df["cluster_number"].to_numpy(dtype=float) + offset
        y = group_df["cca_value"].to_numpy(dtype=float)
        sizes = group_df["marker_size"].to_numpy(dtype=float)

        label_parts = [str(file_label)]

        if pd.notna(sel_key):
            label_parts.append(str(sel_key))

        if result_source not in ["permutation_tests.cca", None]:
            label_parts.append(str(result_source))

        group_label = " | ".join(label_parts)

        line = ax.plot(
            x,
            y,
            linestyle="-",
            linewidth=1.5,
            alpha=0.75,
            label=group_label,
        )

        line_color = line[0].get_color()

        ax.scatter(
            x,
            y,
            s=sizes,
            color=line_color,
            alpha=0.65,
            edgecolors="black",
            linewidths=0.6,
        )

        # Clearly mark non-significant results
        # These are not "wrong", but they are above the chosen p-value threshold.
        p_values = group_df["p_value"].to_numpy(dtype=float)
        non_significant_mask = np.isnan(p_values) | (p_values > p_threshold)

        ax.scatter(
            x[non_significant_mask],
            y[non_significant_mask],
            s=sizes[non_significant_mask] * 1.25,
            marker="X",
            color="red",
            edgecolors="black",
            linewidths=0.8,
            alpha=0.9,
            label="p > 0.05 / not significant",
        )
        
        if annotate_pvalues:
            for _, row in group_df.iterrows():
                p = row["p_value"]

                if pd.isna(p):
                    continue

                if p < 0.001:
                    p_text = "p<.001"
                else:
                    p_text = f"p={p:.3f}"

                x_pos = float(row["cluster_number"]) + offset
                y_pos = float(row["cca_value"])

                ax.annotate(
                    p_text,
                    xy=(x_pos, y_pos),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    alpha=0.85,
                )

    ax.set_xlabel("Cluster number")
    ax.set_ylabel("Observed CCA value")

    ax.set_ylim(0, 1)
    
    cluster_ticks = sorted(all_results_df["cluster_number"].dropna().unique())
    ax.set_xticks(cluster_ticks)

    if title is None:
        title = "CCA values with permutation p-values encoded by marker size"

    ax.set_title(title)

    ax.grid(True, alpha=0.3)

    # Marker-size legend for p-values
    legend_p_values = [0.05, 0.01, 0.001]
    legend_handles = []

    for p in legend_p_values:
        neg_log = -np.log10(p)

        if max_sig > 0:
            size = (
                min_marker_size
                + (neg_log / max_sig)
                * (max_marker_size - min_marker_size)
            )
        else:
            size = min_marker_size

        size = min(size, max_marker_size)

        legend_handles.append(
            ax.scatter(
                [],
                [],
                s=size,
                color="gray",
                alpha=0.65,
                edgecolors="black",
                linewidths=0.6,
                label=f"p = {p}"
            )
        )

    handles, labels = ax.get_legend_handles_labels()

    unique = dict(zip(labels, handles))

    main_legend = ax.legend(
        unique.values(),
        unique.keys(),
        loc="lower left",
        fontsize=8,
        title="JSON file / result set"
    )

    ax.add_artist(main_legend)

    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=8,
        title="Marker size"
    )

    fig.text(
        0.5,
        -0.02,
        "Marker size is proportional to -log10(p). Larger markers indicate lower permutation p-values.",
        ha="center",
        fontsize=9,
    )

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return all_results_df


import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _cluster_number_from_key(key: str):
    match = re.search(r"Cluster_(\d+)_results", key)
    if match is None:
        return None
    return int(match.group(1))


def _load_results_container(json_path):
    """
    Supports both:
        {"Cluster_2_results": {...}}
    and:
        {"results": {"Cluster_2_results": {...}}}
    """
    json_path = Path(json_path)

    with json_path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)

    if isinstance(loaded, dict) and "results" in loaded:
        return loaded["results"]

    return loaded


def extract_selected_variable_permutation_scores(
    json_path,
    condition_label=None,
    score_type="cca",
    selected_variable_keys=("cca_selected_variables", "mlr_selected_variables"),
):
    """
    Extract observed score and p-value for selected-variable permutation results.

    Expected JSON structure:
        Cluster_X_results
            selected_variable_tests
                cca_selected_variables
                    permutation_tests
                        cca / mlr
                mlr_selected_variables
                    permutation_tests
                        cca / mlr

    Parameters
    ----------
    score_type : {"cca", "mlr"}
        Which test result to extract.

        "cca":
            extracts observed CCA value and CCA permutation p-value.

        "mlr":
            extracts observed MLR accuracy and MLR permutation p-value.
    """

    json_path = Path(json_path)

    if condition_label is None:
        condition_label = json_path.stem

    results_container = _load_results_container(json_path)

    rows = []

    for cluster_key, cluster_result in results_container.items():
        cluster_number = _cluster_number_from_key(cluster_key)

        if cluster_number is None:
            continue

        if not isinstance(cluster_result, dict):
            continue

        selected_tests = cluster_result.get("selected_variable_tests", {})

        if not isinstance(selected_tests, dict):
            continue

        for selected_key in selected_variable_keys:
            selected_result = selected_tests.get(selected_key, None)

            if not isinstance(selected_result, dict):
                continue

            permutation_tests = selected_result.get("permutation_tests", {})
            test_result = permutation_tests.get(score_type, None)

            if not isinstance(test_result, dict):
                continue

            observed_score = test_result.get("observed_score", None)
            p_value = test_result.get("p_value", None)

            observed_result = test_result.get("observed_result", {})
            variables_used = observed_result.get("variables_used", [])

            rows.append({
                "condition": condition_label,
                "cluster_number": cluster_number,
                "selected_variable_key": selected_key,
                "score_type": score_type,
                "observed_score": observed_score,
                "p_value": p_value,
                "variables_used": variables_used,
                "source_file": str(json_path),
            })

    df = pd.DataFrame(rows)

    if not df.empty:
        df["observed_score"] = pd.to_numeric(df["observed_score"], errors="coerce")
        df["p_value"] = pd.to_numeric(df["p_value"], errors="coerce")
        df["cluster_number"] = pd.to_numeric(df["cluster_number"], errors="coerce")

        df = df.sort_values(
            ["condition", "selected_variable_key", "cluster_number"]
        ).reset_index(drop=True)

    return df


def plot_selected_variable_scores_all_conditions(
    json_paths,
    condition_labels=None,
    score_type="cca",
    selected_variable_keys=("cca_selected_variables", "mlr_selected_variables"),
    selected_variable_titles=None,
    p_threshold=0.05,
    figsize=(7, 4),
    y_lim=(0, 1),
    marker_min_size=25,
    marker_max_size=160,
    mark_non_significant=True,
    annotate_non_significant=False,
    title=None,
    save_path=None,
    show=True,
):
    """
    Plot observed CCA or MLR value with associated p-values for both
    CCA-selected and MLR-selected variable sets, across all conditions.

    Design:
        - One figure with two panels:
            left:  cca_selected_variables
            right: mlr_selected_variables

        - x-axis:
            cluster number

        - y-axis:
            observed CCA value if score_type="cca"
            observed MLR accuracy if score_type="mlr"

        - line color:
            condition / JSON file

        - marker size:
            -log10(p-value)

        - red X:
            p-value > p_threshold

    Parameters
    ----------
    json_paths : list[str | Path]
        Usually 8 permutation-test JSON files, one per condition.

    condition_labels : list[str] | None
        Labels for each condition. If None, filenames are used.

    score_type : {"cca", "mlr"}
        Which observed score and p-value to plot.

        "cca":
            plots observed CCA value and CCA p-value.

        "mlr":
            plots observed MLR accuracy and MLR p-value.

    selected_variable_keys : tuple[str, ...]
        Usually:
            ("cca_selected_variables", "mlr_selected_variables")

    selected_variable_titles : dict | None
        Optional prettier subplot titles.

    p_threshold : float
        Threshold used to mark non-significant points.

    figsize : tuple
        Default is (7, 4), as requested.

    y_lim : tuple | None
        Default fixed y-axis from 0 to 1.

    marker_min_size, marker_max_size : int
        Marker-size range for p-value encoding.

    mark_non_significant : bool
        Whether to overlay red X markers where p > p_threshold.

    annotate_non_significant : bool
        Whether to write "ns" above non-significant points.

    save_path : str | Path | None
        If given, saves the figure.

    Returns
    -------
    all_results_df : pd.DataFrame
        Extracted results used for plotting.
    """

    if score_type not in {"cca", "mlr"}:
        raise ValueError("score_type must be either 'cca' or 'mlr'.")

    json_paths = [Path(p) for p in json_paths]

    if condition_labels is None:
        condition_labels = [p.stem for p in json_paths]

    if len(condition_labels) != len(json_paths):
        raise ValueError("condition_labels must have the same length as json_paths.")

    if selected_variable_titles is None:
        selected_variable_titles = {
            "cca_selected_variables": "CCA-selected variables",
            "mlr_selected_variables": "MLR-selected variables",
        }

    dfs = []

    for json_path, condition_label in zip(json_paths, condition_labels):
        df = extract_selected_variable_permutation_scores(
            json_path=json_path,
            condition_label=condition_label,
            score_type=score_type,
            selected_variable_keys=selected_variable_keys,
        )

        if not df.empty:
            dfs.append(df)

    if not dfs:
        raise ValueError("No valid selected-variable permutation results found.")

    all_results_df = pd.concat(dfs, ignore_index=True)

    all_results_df = all_results_df.dropna(
        subset=["cluster_number", "observed_score"]
    ).copy()

    if all_results_df.empty:
        raise ValueError("No valid observed scores found after cleaning.")

    # Prepare p-values for marker-size encoding
    safe_p = all_results_df["p_value"].copy()
    safe_p = safe_p.fillna(1.0)
    safe_p = safe_p.clip(lower=1e-300, upper=1.0)

    all_results_df["neg_log10_p"] = -np.log10(safe_p)

    max_neg_log_p = all_results_df["neg_log10_p"].max()

    if max_neg_log_p > 0:
        all_results_df["marker_size"] = (
            marker_min_size
            + (all_results_df["neg_log10_p"] / max_neg_log_p)
            * (marker_max_size - marker_min_size)
        )
    else:
        all_results_df["marker_size"] = marker_min_size

    n_panels = len(selected_variable_keys)

    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=figsize,
        sharey=True,
    )

    if n_panels == 1:
        axes = [axes]

    condition_order = list(condition_labels)

    # Slight x-offsets so 8 conditions do not fully overlap at each cluster number
    if len(condition_order) == 1:
        condition_offsets = {condition_order[0]: 0.0}
    else:
        offsets = np.linspace(-0.22, 0.22, len(condition_order))
        condition_offsets = dict(zip(condition_order, offsets))

    for ax, selected_key in zip(axes, selected_variable_keys):
        panel_df = all_results_df[
            all_results_df["selected_variable_key"] == selected_key
        ].copy()

        for condition in condition_order:
            condition_df = panel_df[
                panel_df["condition"] == condition
            ].copy()

            if condition_df.empty:
                continue

            condition_df = condition_df.sort_values("cluster_number")

            x = (
                condition_df["cluster_number"].to_numpy(dtype=float)
                + condition_offsets[condition]
            )

            y = condition_df["observed_score"].to_numpy(dtype=float)
            sizes = condition_df["marker_size"].to_numpy(dtype=float)
            p_values = condition_df["p_value"].to_numpy(dtype=float)

            line = ax.plot(
                x,
                y,
                marker=None,
                linestyle="-",
                linewidth=1.2,
                alpha=0.75,
                label=condition,
            )

            line_color = line[0].get_color()

            ax.scatter(
                x,
                y,
                s=sizes,
                color=line_color,
                alpha=0.75,
                edgecolors="black",
                linewidths=0.4,
                zorder=3,
            )

            if mark_non_significant:
                non_sig_mask = np.isnan(p_values) | (p_values > p_threshold)

                if np.any(non_sig_mask):
                    ax.scatter(
                        x[non_sig_mask],
                        y[non_sig_mask],
                        s=np.maximum(sizes[non_sig_mask] * 1.15, 80),
                        marker="X",
                        color="red",
                        edgecolors="black",
                        linewidths=0.5,
                        alpha=0.95,
                        zorder=4,
                    )

                    if annotate_non_significant:
                        for x_i, y_i in zip(x[non_sig_mask], y[non_sig_mask]):
                            ax.annotate(
                                "ns",
                                xy=(x_i, y_i),
                                xytext=(0, 7),
                                textcoords="offset points",
                                ha="center",
                                va="bottom",
                                fontsize=7,
                                color="red",
                            )

        ax.set_title(
            selected_variable_titles.get(selected_key, selected_key),
            fontsize=10,
        )

        ax.set_xlabel("Cluster number")
        ax.grid(True, alpha=0.25)

        cluster_ticks = sorted(
            all_results_df["cluster_number"].dropna().unique()
        )

        ax.set_xticks(cluster_ticks)

        if y_lim is not None:
            ax.set_ylim(y_lim)

    if score_type == "cca":
        y_label = "Observed CCA value"
        default_title = "CCA score and permutation p-values across conditions"
    else:
        y_label = "Observed MLR accuracy"
        default_title = "MLR score and permutation p-values across conditions"

    axes[0].set_ylabel(y_label)

    if title is None:
        title = default_title

    fig.suptitle(title, fontsize=11)

    # One shared legend for all condition lines
    handles, labels = axes[-1].get_legend_handles_labels()

    unique = dict(zip(labels, handles))

    fig.legend(
        unique.values(),
        unique.keys(),
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=8,
        title="Condition",
    )

    # Small p-value marker explanation
    fig.text(
        0.5,
        -0.03,
        f"Marker size ∝ -log10(p). Red X indicates p > {p_threshold}.",
        ha="center",
        fontsize=8,
    )

    fig.tight_layout(rect=[0, 0.03, 0.86, 0.92])

    if save_path is not None:
        save_path = Path(save_path)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return all_results_df



def plot_cca_mlr_side_by_side_same_selected_variables(
    json_paths,
    condition_labels=None,
    selected_variable_key="cca_selected_variables",
    selected_variable_title=None,
    p_threshold=0.05,
    figsize=(7, 4),
    y_lim=(0, 1),
    marker_min_size=25,
    marker_max_size=160,
    mark_non_significant=True,
    annotate_non_significant=False,
    title=None,
    save_path=None,
    show=True,
):
    """
    Plot CCA and MLR scores side by side using the same selected-variable set.

    Example:
        selected_variable_key="cca_selected_variables"

    Then:
        left panel  = CCA score using CCA-selected variables
        right panel = MLR score using CCA-selected variables

    The p-value associated with each score is encoded by marker size.
    Non-significant points, p > p_threshold, are marked with a red X.

    Requires the helper:
        extract_selected_variable_permutation_scores(...)
    from the previous plotting code.
    """

    json_paths = [Path(p) for p in json_paths]

    if condition_labels is None:
        condition_labels = [p.stem for p in json_paths]

    if len(condition_labels) != len(json_paths):
        raise ValueError("condition_labels must have the same length as json_paths.")

    if selected_variable_title is None:
        selected_variable_title = selected_variable_key.replace("_", " ")

    dfs = []

    for score_type in ["cca", "mlr"]:
        for json_path, condition_label in zip(json_paths, condition_labels):
            df = extract_selected_variable_permutation_scores(
                json_path=json_path,
                condition_label=condition_label,
                score_type=score_type,
                selected_variable_keys=(selected_variable_key,),
            )

            if not df.empty:
                dfs.append(df)

    if not dfs:
        raise ValueError(
            f"No valid CCA/MLR permutation results found for "
            f"selected_variable_key='{selected_variable_key}'."
        )

    all_results_df = pd.concat(dfs, ignore_index=True)

    all_results_df = all_results_df.dropna(
        subset=["cluster_number", "observed_score"]
    ).copy()

    if all_results_df.empty:
        raise ValueError("No valid observed scores found after cleaning.")

    # Marker size encodes p-value strength across both panels
    safe_p = all_results_df["p_value"].copy()
    safe_p = safe_p.fillna(1.0)
    safe_p = safe_p.clip(lower=1e-300, upper=1.0)

    all_results_df["neg_log10_p"] = -np.log10(safe_p)

    max_neg_log_p = all_results_df["neg_log10_p"].max()

    if max_neg_log_p > 0:
        all_results_df["marker_size"] = (
            marker_min_size
            + (all_results_df["neg_log10_p"] / max_neg_log_p)
            * (marker_max_size - marker_min_size)
        )
    else:
        all_results_df["marker_size"] = marker_min_size

    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
        sharey=True,
    )

    score_panels = [
        ("cca", "CCA value"),
        ("mlr", "MLR accuracy"),
    ]

    condition_order = list(condition_labels)

    # Small offsets reduce overlap between the 8 conditions
    if len(condition_order) == 1:
        condition_offsets = {condition_order[0]: 0.0}
    else:
        offsets = np.linspace(-0.22, 0.22, len(condition_order))
        condition_offsets = dict(zip(condition_order, offsets))

    for ax, (score_type, panel_title) in zip(axes, score_panels):
        panel_df = all_results_df[
            all_results_df["score_type"] == score_type
        ].copy()

        for condition in condition_order:
            condition_df = panel_df[
                panel_df["condition"] == condition
            ].copy()

            if condition_df.empty:
                continue

            condition_df = condition_df.sort_values("cluster_number")

            x = (
                condition_df["cluster_number"].to_numpy(dtype=float)
                + condition_offsets[condition]
            )

            y = condition_df["observed_score"].to_numpy(dtype=float)
            sizes = condition_df["marker_size"].to_numpy(dtype=float)
            p_values = condition_df["p_value"].to_numpy(dtype=float)

            line = ax.plot(
                x,
                y,
                linestyle="-",
                linewidth=1.2,
                alpha=0.75,
                label=condition,
            )

            line_color = line[0].get_color()

            ax.scatter(
                x,
                y,
                s=sizes,
                color=line_color,
                alpha=0.75,
                edgecolors="black",
                linewidths=0.4,
                zorder=3,
            )

            if mark_non_significant:
                non_sig_mask = np.isnan(p_values) | (p_values > p_threshold)

                if np.any(non_sig_mask):
                    ax.scatter(
                        x[non_sig_mask],
                        y[non_sig_mask],
                        s=np.maximum(sizes[non_sig_mask] * 1.15, 80),
                        marker="X",
                        color="red",
                        edgecolors="black",
                        linewidths=0.5,
                        alpha=0.95,
                        zorder=4,
                    )

                    if annotate_non_significant:
                        for x_i, y_i in zip(x[non_sig_mask], y[non_sig_mask]):
                            ax.annotate(
                                "ns",
                                xy=(x_i, y_i),
                                xytext=(0, 7),
                                textcoords="offset points",
                                ha="center",
                                va="bottom",
                                fontsize=7,
                                color="red",
                            )

        ax.set_title(panel_title, fontsize=10)
        ax.set_xlabel("Cluster number")
        ax.grid(True, alpha=0.25)

        cluster_ticks = sorted(
            all_results_df["cluster_number"].dropna().unique()
        )

        ax.set_xticks(cluster_ticks)

        if y_lim is not None:
            ax.set_ylim(y_lim)

    axes[0].set_ylabel("Observed score")

    if title is None:
        title = f"CCA and MLR performance using {selected_variable_title}"

    fig.suptitle(title, fontsize=11)

    # Shared condition legend
    handles, labels = axes[-1].get_legend_handles_labels()
    unique = dict(zip(labels, handles))

    fig.legend(
        unique.values(),
        unique.keys(),
        loc="center left",
        bbox_to_anchor=(0.86, 0.5),
        fontsize=8,
        title="Condition",
    )

    fig.text(
        0.5,
        -0.03,
        f"Marker size ∝ -log10(p). Red X indicates p > {p_threshold}.",
        ha="center",
        fontsize=8,
    )

    fig.tight_layout(rect=[0, 0.03, 0.86, 0.92])

    if save_path is not None:
        save_path = Path(save_path)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return all_results_df



def plot_selected_score_two_file_sets(
    left_json_paths,
    right_json_paths,
    left_title="Set 1",
    right_title="Set 2",
    left_labels=None,
    right_labels=None,
    score_type="cca",
    selected_variable_key="cca_selected_variables",
    selected_variable_title=None,
    p_threshold=0.05,
    figsize=(7, 4),
    y_lim=(0, 1),
    marker_min_size=25,
    marker_max_size=160,
    mark_non_significant=True,
    annotate_non_significant=False,
    title=None,
    legend_outside=True,
    save_path=None,
    show=True,
):
    """
    Plot one selected score type, CCA or MLR, with its associated p-value,
    for one selected-variable set, CCA-selected or MLR-selected variables,
    comparing two different sets of JSON files.

    Layout:
        left panel  = left_json_paths
        right panel = right_json_paths

    Parameters
    ----------
    left_json_paths, right_json_paths : list[str | Path]
        Two groups of permutation-test JSON files.

    left_title, right_title : str
        Titles for the left and right panels.

    left_labels, right_labels : list[str] | None
        Labels for the files in each panel.
        If None, filename stems are used.

    score_type : {"cca", "mlr"}
        Which observed score to plot.

        "cca":
            observed CCA value + CCA p-value.

        "mlr":
            observed MLR accuracy + MLR p-value.

    selected_variable_key : {"cca_selected_variables", "mlr_selected_variables"}
        Which selected-variable set to use.

    p_threshold : float
        Points with p > p_threshold are marked with a red X.

    figsize : tuple
        Default (7, 4).

    y_lim : tuple | None
        Default fixed y-axis from 0 to 1.

    Returns
    -------
    all_results_df : pd.DataFrame
        Extracted plotting data.
    """

    if score_type not in {"cca", "mlr"}:
        raise ValueError("score_type must be either 'cca' or 'mlr'.")

    left_json_paths = [Path(p) for p in left_json_paths]
    right_json_paths = [Path(p) for p in right_json_paths]

    if left_labels is None:
        left_labels = [p.stem for p in left_json_paths]

    if right_labels is None:
        right_labels = [p.stem for p in right_json_paths]

    if len(left_labels) != len(left_json_paths):
        raise ValueError("left_labels must have the same length as left_json_paths.")

    if len(right_labels) != len(right_json_paths):
        raise ValueError("right_labels must have the same length as right_json_paths.")

    if selected_variable_title is None:
        selected_variable_title = selected_variable_key.replace("_", " ")

    def _collect_set(json_paths, labels, set_name):
        dfs = []

        for json_path, label in zip(json_paths, labels):
            df = extract_selected_variable_permutation_scores(
                json_path=json_path,
                condition_label=label,
                score_type=score_type,
                selected_variable_keys=(selected_variable_key,),
            )

            if not df.empty:
                df["set_name"] = set_name
                dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        return pd.concat(dfs, ignore_index=True)

    left_df = _collect_set(left_json_paths, left_labels, left_title)
    right_df = _collect_set(right_json_paths, right_labels, right_title)

    all_results_df = pd.concat(
        [left_df, right_df],
        ignore_index=True
    )

    if all_results_df.empty:
        raise ValueError("No valid results found in either file set.")

    all_results_df = all_results_df.dropna(
        subset=["cluster_number", "observed_score"]
    ).copy()

    if all_results_df.empty:
        raise ValueError("No valid observed scores found after cleaning.")

    all_results_df["cluster_number"] = pd.to_numeric(
        all_results_df["cluster_number"],
        errors="coerce"
    )

    all_results_df["observed_score"] = pd.to_numeric(
        all_results_df["observed_score"],
        errors="coerce"
    )

    all_results_df["p_value"] = pd.to_numeric(
        all_results_df["p_value"],
        errors="coerce"
    )

    # Marker size encodes p-value strength globally across both panels
    safe_p = all_results_df["p_value"].copy()
    safe_p = safe_p.fillna(1.0)
    safe_p = safe_p.clip(lower=1e-300, upper=1.0)

    all_results_df["neg_log10_p"] = -np.log10(safe_p)

    max_neg_log_p = all_results_df["neg_log10_p"].max()

    if max_neg_log_p > 0:
        all_results_df["marker_size"] = (
            marker_min_size
            + (all_results_df["neg_log10_p"] / max_neg_log_p)
            * (marker_max_size - marker_min_size)
        )
    else:
        all_results_df["marker_size"] = marker_min_size

    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
        sharey=True,
    )

    panels = [
        (axes[0], left_title, left_labels),
        (axes[1], right_title, right_labels),
    ]

    # Same condition name gets same colour across both panels
    all_condition_labels = list(dict.fromkeys(left_labels + right_labels))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {
        label: color_cycle[i % len(color_cycle)]
        for i, label in enumerate(all_condition_labels)
    }

    for ax, panel_title, panel_labels in panels:
        panel_df = all_results_df[
            all_results_df["set_name"] == panel_title
        ].copy()

        if len(panel_labels) == 1:
            condition_offsets = {panel_labels[0]: 0.0}
        else:
            offsets = np.linspace(-0.22, 0.22, len(panel_labels))
            condition_offsets = dict(zip(panel_labels, offsets))

        for condition in panel_labels:
            condition_df = panel_df[
                panel_df["condition"] == condition
            ].copy()

            if condition_df.empty:
                continue

            condition_df = condition_df.sort_values("cluster_number")

            x = (
                condition_df["cluster_number"].to_numpy(dtype=float)
                + condition_offsets[condition]
            )

            y = condition_df["observed_score"].to_numpy(dtype=float)
            sizes = condition_df["marker_size"].to_numpy(dtype=float)
            p_values = condition_df["p_value"].to_numpy(dtype=float)

            line_color = color_map[condition]

            ax.plot(
                x,
                y,
                linestyle="-",
                linewidth=1.2,
                alpha=0.75,
                color=line_color,
                label=condition,
            )

            ax.scatter(
                x,
                y,
                s=sizes,
                color=line_color,
                alpha=0.75,
                edgecolors="black",
                linewidths=0.4,
                zorder=3,
            )

            if mark_non_significant:
                non_sig_mask = np.isnan(p_values) | (p_values > p_threshold)

                if np.any(non_sig_mask):
                    ax.scatter(
                        x[non_sig_mask],
                        y[non_sig_mask],
                        s=np.maximum(sizes[non_sig_mask] * 1.15, 80),
                        marker="X",
                        color="red",
                        edgecolors="black",
                        linewidths=0.5,
                        alpha=0.95,
                        zorder=4,
                    )

                    if annotate_non_significant:
                        for x_i, y_i in zip(x[non_sig_mask], y[non_sig_mask]):
                            ax.annotate(
                                "ns",
                                xy=(x_i, y_i),
                                xytext=(0, 7),
                                textcoords="offset points",
                                ha="center",
                                va="bottom",
                                fontsize=7,
                                color="red",
                            )

        ax.set_title(panel_title, fontsize=10)
        ax.set_xlabel("Cluster number")
        ax.grid(True, alpha=0.25)

        cluster_ticks = sorted(
            all_results_df["cluster_number"].dropna().unique()
        )

        ax.set_xticks(cluster_ticks)

        if y_lim is not None:
            ax.set_ylim(y_lim)

    if score_type == "cca":
        y_label = "Observed CCA value"
        default_title = f"CCA score using {selected_variable_title}"
    else:
        y_label = "Observed MLR accuracy"
        default_title = f"MLR score using {selected_variable_title}"

    axes[0].set_ylabel(y_label)

    if title is None:
        title = default_title

    fig.suptitle(title, fontsize=11)

    # Shared legend
    handles, labels = [], []

    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    unique = dict(zip(labels, handles))

    if legend_outside:
        fig.legend(
            unique.values(),
            unique.keys(),
            loc="center left",
            bbox_to_anchor=(0.86, 0.5),
            fontsize=8,
            title="Condition",
        )
        layout_rect = [0, 0.04, 0.84, 0.92]
    else:
        axes[0].legend(
            unique.values(),
            unique.keys(),
            loc="lower left",
            fontsize=7,
            title="Condition",
        )
        layout_rect = [0, 0.04, 1, 0.92]

    fig.text(
        0.5,
        -0.03,
        f"Marker size ∝ -log10(p). Red X indicates p > {p_threshold}.",
        ha="center",
        fontsize=8,
    )

    fig.tight_layout(rect=layout_rect)

    if save_path is not None:
        save_path = Path(save_path)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return all_results_df



import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def benjamini_hochberg_correction(p_values, alpha=0.05):
    """
    Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    p_values : list-like
        Raw p-values.

    alpha : float
        FDR threshold.

    Returns
    -------
    adjusted_p_values : np.ndarray
        BH-adjusted p-values.

    significant : np.ndarray
        Boolean array indicating significance after BH correction.
    """

    p_values = np.asarray(p_values, dtype=float)

    adjusted = np.full_like(p_values, np.nan, dtype=float)
    significant = np.full_like(p_values, False, dtype=bool)

    valid_mask = ~np.isnan(p_values)
    valid_p = p_values[valid_mask]

    m = len(valid_p)

    if m == 0:
        return adjusted, significant

    order = np.argsort(valid_p)
    sorted_p = valid_p[order]

    ranks = np.arange(1, m + 1)

    # Raw BH adjusted values
    sorted_adjusted = sorted_p * m / ranks

    # Enforce monotonicity from largest to smallest p-value
    sorted_adjusted = np.minimum.accumulate(sorted_adjusted[::-1])[::-1]

    # Cap at 1
    sorted_adjusted = np.minimum(sorted_adjusted, 1.0)

    # Put adjusted p-values back in original valid order
    adjusted_valid = np.empty_like(sorted_adjusted)
    adjusted_valid[order] = sorted_adjusted

    adjusted[valid_mask] = adjusted_valid
    significant[valid_mask] = adjusted_valid <= alpha

    return adjusted, significant


def apply_bh_fdr_to_permutation_jsons(
    json_paths,
    output_paths=None,
    alpha=0.05,
    correction_scope="global",
    tests=("cca", "mlr"),
    selected_variable_keys=("cca_selected_variables", "mlr_selected_variables"),
    in_place=True,
    make_backup=True,
    print_results=True,
):
    """
    Apply Benjamini-Hochberg FDR correction to permutation p-values stored in
    one or more permutation-test JSON files.

    By default, this updates the original JSON files directly.

    Expected JSON structure:

        results
            Cluster_X_results
                selected_variable_tests
                    cca_selected_variables
                        permutation_tests
                            cca
                                p_value
                            mlr
                                p_value
                    mlr_selected_variables
                        permutation_tests
                            cca
                                p_value
                            mlr
                                p_value

    Parameters
    ----------
    json_paths : list[str | Path]
        Input JSON files.

    output_paths : list[str | Path] | None
        Output JSON files.

        Only used if in_place=False.

    alpha : float
        FDR threshold, usually 0.05.

    correction_scope : str
        Defines which p-values are corrected together.

        Options:
            "global"
                Correct all p-values across all files, selected-variable sets,
                tests, and cluster numbers together.

            "within_file"
                Correct p-values separately within each JSON file.

            "by_test"
                Correct CCA p-values and MLR p-values separately across all files.

            "by_selected_variable_key"
                Correct p-values separately for cca_selected_variables and
                mlr_selected_variables across all files.

            "by_test_and_selected_variable_key"
                Correct separately for each combination, for example:
                    CCA test on CCA-selected variables
                    MLR test on CCA-selected variables
                    CCA test on MLR-selected variables
                    MLR test on MLR-selected variables

    tests : tuple[str, ...]
        Which permutation test result types to correct.

    selected_variable_keys : tuple[str, ...]
        Which selected-variable sets to include.

    in_place : bool
        If True, update the original JSON files directly.
        If False, write corrected files to output_paths.

    make_backup : bool
        If True and in_place=True, creates backup files before overwriting.

    print_results : bool
        Whether to print a short summary.

    Returns
    -------
    updated_jsons : dict
        Dictionary mapping input paths to updated JSON content.

    summary_df : pd.DataFrame
        Table containing raw and corrected p-values.
    """

    json_paths = [Path(p) for p in json_paths]

    if in_place:
        output_paths = json_paths
    else:
        if output_paths is None:
            output_paths = [
                p.with_name(p.stem + "_bh_fdr" + p.suffix)
                for p in json_paths
            ]
        else:
            output_paths = [Path(p) for p in output_paths]

        if len(output_paths) != len(json_paths):
            raise ValueError("output_paths must have the same length as json_paths.")

    allowed_scopes = {
        "global",
        "within_file",
        "by_test",
        "by_selected_variable_key",
        "by_test_and_selected_variable_key",
    }

    if correction_scope not in allowed_scopes:
        raise ValueError(
            f"correction_scope must be one of {sorted(allowed_scopes)}"
        )

    def _cluster_number_from_key(key: str):
        match = re.search(r"Cluster_(\d+)_results", key)
        if match is None:
            return None
        return int(match.group(1))

    def _get_results_container(loaded_json):
        """
        Supports both:
            {"Cluster_2_results": {...}}
        and:
            {"results": {"Cluster_2_results": {...}}}
        """
        if isinstance(loaded_json, dict) and "results" in loaded_json:
            return loaded_json["results"]
        return loaded_json

    def _json_safe(obj):
        """
        Convert NumPy values to JSON-safe Python types.
        """
        if isinstance(obj, dict):
            return {str(k): _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_json_safe(v) for v in obj]
        if isinstance(obj, tuple):
            return [_json_safe(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return _json_safe(obj.tolist())
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            if np.isnan(obj):
                return None
            return float(obj)
        if isinstance(obj, float) and np.isnan(obj):
            return None
        return obj

    loaded_by_path = {}
    pvalue_rows = []

    # ========================================================
    # 1. Read all JSON files and collect p-values
    # ========================================================

    for file_idx, json_path in enumerate(json_paths):
        with json_path.open("r", encoding="utf-8") as f:
            loaded_json = json.load(f)

        loaded_by_path[str(json_path)] = loaded_json
        results_container = _get_results_container(loaded_json)

        if not isinstance(results_container, dict):
            continue

        for cluster_key, cluster_result in results_container.items():
            cluster_number = _cluster_number_from_key(cluster_key)

            if cluster_number is None:
                continue

            if not isinstance(cluster_result, dict):
                continue

            selected_tests = cluster_result.get("selected_variable_tests", {})

            if not isinstance(selected_tests, dict):
                continue

            for selected_key in selected_variable_keys:
                selected_result = selected_tests.get(selected_key, None)

                if not isinstance(selected_result, dict):
                    continue

                permutation_tests = selected_result.get("permutation_tests", {})

                if not isinstance(permutation_tests, dict):
                    continue

                for test_name in tests:
                    test_result = permutation_tests.get(test_name, None)

                    if not isinstance(test_result, dict):
                        continue

                    raw_p = test_result.get("p_value", None)

                    try:
                        p_float = float(raw_p) if raw_p is not None else np.nan
                    except (TypeError, ValueError):
                        p_float = np.nan

                    pvalue_rows.append({
                        "file_idx": file_idx,
                        "json_path": str(json_path),
                        "cluster_key": cluster_key,
                        "cluster_number": cluster_number,
                        "selected_variable_key": selected_key,
                        "test": test_name,
                        "p_value": p_float,
                    })

    summary_df = pd.DataFrame(pvalue_rows)

    if summary_df.empty:
        raise ValueError("No permutation p-values found in the provided JSON files.")

    # ========================================================
    # 2. Define correction families
    # ========================================================

    if correction_scope == "global":
        summary_df["_correction_group"] = "global"

    elif correction_scope == "within_file":
        summary_df["_correction_group"] = summary_df["json_path"]

    elif correction_scope == "by_test":
        summary_df["_correction_group"] = summary_df["test"]

    elif correction_scope == "by_selected_variable_key":
        summary_df["_correction_group"] = summary_df["selected_variable_key"]

    elif correction_scope == "by_test_and_selected_variable_key":
        summary_df["_correction_group"] = (
            summary_df["test"].astype(str)
            + "__"
            + summary_df["selected_variable_key"].astype(str)
        )

    summary_df["p_value_fdr_bh"] = np.nan
    summary_df["significant_fdr_bh"] = False
    summary_df["fdr_alpha"] = alpha
    summary_df["correction_scope"] = correction_scope

    # ========================================================
    # 3. Apply BH correction within each family
    # ========================================================

    for _, group_df in summary_df.groupby("_correction_group"):
        adjusted, significant = benjamini_hochberg_correction(
            group_df["p_value"].to_numpy(),
            alpha=alpha,
        )

        summary_df.loc[group_df.index, "p_value_fdr_bh"] = adjusted
        summary_df.loc[group_df.index, "significant_fdr_bh"] = significant

    # ========================================================
    # 4. Write adjusted values back into loaded JSON objects
    # ========================================================

    for _, row in summary_df.iterrows():
        loaded_json = loaded_by_path[row["json_path"]]
        results_container = _get_results_container(loaded_json)

        test_result = (
            results_container[row["cluster_key"]]
            ["selected_variable_tests"][row["selected_variable_key"]]
            ["permutation_tests"][row["test"]]
        )

        if pd.isna(row["p_value_fdr_bh"]):
            corrected_p = None
        else:
            corrected_p = float(row["p_value_fdr_bh"])

        test_result["p_value_fdr_bh"] = corrected_p
        test_result["significant_fdr_bh"] = bool(row["significant_fdr_bh"])
        test_result["fdr_alpha"] = float(alpha)
        test_result["fdr_correction"] = "Benjamini-Hochberg"
        test_result["fdr_correction_scope"] = correction_scope

    # ========================================================
    # 5. Save files
    # ========================================================

    updated_jsons = {}

    for json_path, output_path in zip(json_paths, output_paths):
        loaded_json = loaded_by_path[str(json_path)]
        updated_jsons[str(json_path)] = loaded_json

        if in_place and make_backup:
            backup_path = json_path.with_name(
                json_path.stem + "_backup_before_bh_fdr" + json_path.suffix
            )

            if not backup_path.exists():
                with backup_path.open("w", encoding="utf-8") as f:
                    json.dump(_json_safe(loaded_json), f, indent=4)

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(_json_safe(loaded_json), f, indent=4)

        if print_results:
            if in_place:
                print(f"Updated JSON in place: {output_path}")
            else:
                print(f"Saved BH-FDR corrected JSON to: {output_path}")

    # ========================================================
    # 6. Print summary
    # ========================================================

    if print_results:
        n_tests = int(summary_df["p_value"].notna().sum())
        n_sig_raw = int((summary_df["p_value"] <= alpha).sum())
        n_sig_fdr = int(summary_df["significant_fdr_bh"].sum())

        print("\n=== Benjamini-Hochberg FDR summary ===")
        print(f"Correction scope: {correction_scope}")
        print(f"alpha: {alpha}")
        print(f"Number of valid p-values: {n_tests}")
        print(f"Raw p <= {alpha}: {n_sig_raw}")
        print(f"BH-FDR significant: {n_sig_fdr}")

    summary_df = summary_df.drop(columns=["_correction_group"])

    return updated_jsons, summary_df