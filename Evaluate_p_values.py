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