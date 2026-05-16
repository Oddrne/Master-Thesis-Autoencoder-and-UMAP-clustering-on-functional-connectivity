import numpy as np
import pandas as pd

from sklearn.cross_decomposition import CCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import json
import re
from pathlib import Path


# ============================================================
# 1. Core CCA helpers
# ============================================================

def _safe_corr(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()

    if len(a) != len(b) or len(a) < 2:
        return np.nan
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan

    return float(np.corrcoef(a, b)[0, 1])


def _prepare_cluster_matrix(cluster_labels):
    """
    One-hot encode cluster labels.
    """
    cluster_df = pd.DataFrame({"cluster": pd.Series(cluster_labels).astype(str)})

    try:
        encoder = OneHotEncoder(
            sparse_output=False,
            drop=None,
            handle_unknown="ignore"
        )
    except TypeError:
        encoder = OneHotEncoder(
            sparse=False,
            drop=None,
            handle_unknown="ignore"
        )

    X_clusters = encoder.fit_transform(cluster_df[["cluster"]])

    try:
        feature_names = encoder.get_feature_names_out(["cluster"]).tolist()
    except Exception:
        feature_names = [f"cluster_{i}" for i in range(X_clusters.shape[1])]

    return X_clusters, feature_names


def _prepare_numeric_variables(data, variable_cols):
    """
    Keep numeric dependent variables only, then impute + standardize.
    """
    df = data[variable_cols].copy()

    kept_cols = []
    for col in variable_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            kept_cols.append(col)

    if not kept_cols:
        raise ValueError("No numeric dependent variables found for CCA.")

    Y = df[kept_cols].to_numpy()

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    Y_imputed = imputer.fit_transform(Y)
    Y_scaled = scaler.fit_transform(Y_imputed)

    return Y_scaled, kept_cols


# ============================================================
# 2. Run in-sample CCA
# ============================================================

def run_cluster_behavior_cca(
    data: pd.DataFrame,
    cluster_col: str,
    variable_cols: list[str],
    n_components: int = 1,
    max_iter: int = 1000,
    tol: float = 1e-6,
):
    """
    Run CCA between:
      X = one-hot encoded cluster labels
      Y = numeric behavioral variables

    Returns a detailed results dict.
    """
    df = data[[cluster_col] + variable_cols].copy()
    df = df.dropna(subset=[cluster_col]).reset_index(drop=True)

    if df.empty:
        raise ValueError("No rows left after removing missing cluster labels.")

    X_clusters, cluster_feature_names = _prepare_cluster_matrix(df[cluster_col])
    Y_scaled, kept_variable_cols = _prepare_numeric_variables(df, variable_cols)

    max_valid_components = min(
        X_clusters.shape[0],
        X_clusters.shape[1],
        Y_scaled.shape[1],
    )

    if max_valid_components < 1:
        raise ValueError("CCA needs at least 1 valid component.")

    n_components = min(n_components, max_valid_components)

    cca = CCA(
        n_components=n_components,
        scale=False,
        max_iter=max_iter,
        tol=tol,
    )

    X_c, Y_c = cca.fit_transform(X_clusters, Y_scaled)

    canonical_correlations = []
    for i in range(n_components):
        canonical_correlations.append(_safe_corr(X_c[:, i], Y_c[:, i]))

    canonical_correlations = np.array(canonical_correlations)

    scores_df = pd.DataFrame(
        np.hstack([X_c, Y_c]),
        columns=[f"X_CC{i+1}" for i in range(n_components)] +
                [f"Y_CC{i+1}" for i in range(n_components)],
    )

    return {
        "n_samples": len(df),
        "cluster_features": cluster_feature_names,
        "dependent_variables_used": kept_variable_cols,
        "n_components": n_components,
        "canonical_correlations": canonical_correlations,
        "scores": scores_df,
        "model": cca,
    }


# ============================================================
# 3. Cross-validated CCA
# ============================================================

def cross_validated_cca_correlations(
    data: pd.DataFrame,
    cluster_col: str,
    variable_cols: list[str],
    n_components: int = 1,
    n_splits: int = 5,
    random_state: int = 42,
):
    """
    Estimate out-of-sample canonical correlation with K-fold CV.
    """
    df = data[[cluster_col] + variable_cols].copy()
    df = df.dropna(subset=[cluster_col]).reset_index(drop=True)

    if len(df) < n_splits:
        raise ValueError("Not enough samples for the requested number of CV folds.")

    kept_cols = [c for c in variable_cols if pd.api.types.is_numeric_dtype(df[c])]
    if not kept_cols:
        raise ValueError("No numeric dependent variables found for CCA.")
    
    class_counts = df[cluster_col].value_counts()
    min_class_count = class_counts.min()
    if min_class_count < n_splits:
        print(f"Warning: Reducing n_splits from {n_splits} to {max(min_class_count, 3)} due to class imbalance.")
        n_splits = max(min_class_count, 3)  # Ensure at least 3  splits

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(df), start=1):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        train_clusters = pd.DataFrame({"cluster": train_df[cluster_col].astype(str)})
        test_clusters = pd.DataFrame({"cluster": test_df[cluster_col].astype(str)})

        try:
            encoder = OneHotEncoder(
                sparse_output=False,
                drop=None,
                handle_unknown="ignore"
            )
        except TypeError:
            encoder = OneHotEncoder(
                sparse=False,
                drop=None,
                handle_unknown="ignore"
            )

        X_train = encoder.fit_transform(train_clusters[["cluster"]])
        X_test = encoder.transform(test_clusters[["cluster"]])

        train_Y = train_df[kept_cols].to_numpy()
        test_Y = test_df[kept_cols].to_numpy()

        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()

        train_Y = scaler.fit_transform(imputer.fit_transform(train_Y))
        test_Y = scaler.transform(imputer.transform(test_Y))

        max_valid = min(X_train.shape[0], X_train.shape[1], train_Y.shape[1])
        n_comp_fold = min(n_components, max_valid)

        if n_comp_fold < 1:
            raise ValueError("No valid CCA component in this fold.")

        cca = CCA(
            n_components=n_comp_fold,
            scale=False,
            max_iter=1000,
            tol=1e-6,
        )
        cca.fit(X_train, train_Y)

        X_test_c, Y_test_c = cca.transform(X_test, test_Y)
        corrs = [_safe_corr(X_test_c[:, i], Y_test_c[:, i]) for i in range(n_comp_fold)]

        fold_results.append({
            "fold": fold_idx,
            "canonical_correlations": corrs,
        })

    max_components_seen = max(len(fr["canonical_correlations"]) for fr in fold_results)
    mean_corrs = []
    std_corrs = []

    for i in range(max_components_seen):
        vals = [
            fr["canonical_correlations"][i]
            for fr in fold_results
            if len(fr["canonical_correlations"]) > i and not np.isnan(fr["canonical_correlations"][i])
        ]
        mean_corrs.append(float(np.mean(vals)) if vals else np.nan)
        std_corrs.append(float(np.std(vals)) if vals else np.nan)

    return {
        "fold_results": fold_results,
        "mean_test_canonical_correlations": mean_corrs,
        "std_test_canonical_correlations": std_corrs,
    }


# ============================================================
# 4. Common evaluator: in-sample CC + CV CCA
# ============================================================

def evaluate_cca_cv(
    data: pd.DataFrame,
    cluster_col: str,
    variable_cols: list[str],
    cv_splits: int = 5,
    random_state: int = 42,
):
    """
    Returns:
        dict with:
            cc
            cv_mean_cc
            cv_std_cc
            n_samples
            variables_used
    """
    cca_result = run_cluster_behavior_cca(
        data=data,
        cluster_col=cluster_col,
        variable_cols=variable_cols,
        n_components=1,
    )

    cv_result = cross_validated_cca_correlations(
        data=data,
        cluster_col=cluster_col,
        variable_cols=variable_cols,
        n_components=1,
        n_splits=cv_splits,
        random_state=random_state,
    )

    return {
        "cc": float(cca_result["canonical_correlations"][0]),
        "cv_mean_cc": float(cv_result["mean_test_canonical_correlations"][0]),
        "cv_std_cc": float(cv_result["std_test_canonical_correlations"][0]),
        "n_samples": int(cca_result["n_samples"]),
        "variables_used": cca_result["dependent_variables_used"],
    }


# ============================================================
# 5. Step 1: CCA on all variables
# ============================================================

def run_full_cca(
    data: pd.DataFrame,
    cluster_col: str,
    variable_cols: list[str],
    cv_splits: int = 5,
    random_state: int = 42,
    print_results: bool = True,
):
    results = evaluate_cca_cv(
        data=data,
        cluster_col=cluster_col,
        variable_cols=variable_cols,
        cv_splits=cv_splits,
        random_state=random_state,
    )
    if print_results:
        print("\n=== Step 1: CCA on all variables ===")
        print(f"n_samples: {results['n_samples']}")
        print(f"CC1:          {results['cc']:.4f}")
        print(f"Mean CV CC1:  {results['cv_mean_cc']:.4f}")
        print(f"Std CV CC1:   {results['cv_std_cc']:.4f}")

    return {
        "cc": results["cc"],
        "cv_mean_cc": results["cv_mean_cc"],
        "cv_std_cc": results["cv_std_cc"],
    }


# ============================================================
# 6. Step 2: Add-one-in forward selection for CCA
# ============================================================

def forward_select_variables_cca(
    data: pd.DataFrame,
    cluster_col: str,
    candidate_variables: list[str],
    cv_splits: int = 5,
    random_state: int = 42,
    max_variables: int | None = None,
    min_improvement: float = 0.0,
    print_results: bool = True,
):
    """
    Greedy forward selection using CV mean CC1 as main score,
    and lower CV std as tie-breaker.
    """
    if max_variables is None:
        max_variables = len(candidate_variables)

    selected = []
    remaining = candidate_variables.copy()
    history = []

    best_score = -np.inf
    best_std = np.nan

    if print_results:
        print("\n=== Step 2: Forward add-one-in variable selection (CCA) ===")

    step = 1
    while remaining and len(selected) < max_variables:
        round_results = []

        for var in remaining:
            vars_try = selected + [var]

            try:
                res = evaluate_cca_cv(
                    data=data,
                    cluster_col=cluster_col,
                    variable_cols=vars_try,
                    cv_splits=cv_splits,
                    random_state=random_state,
                )
            except Exception:
                continue

            round_results.append({
                "step": step,
                "candidate_variable": var,
                "variables_if_added": vars_try.copy(),
                "cc": res["cc"],
                "cv_mean_cc": res["cv_mean_cc"],
                "cv_std_cc": res["cv_std_cc"],
            })

        if not round_results:
            print("Stopping: no valid candidate variables.")
            break

        round_df = pd.DataFrame(round_results)
        round_df = round_df.sort_values(
            by=["cv_mean_cc", "cv_std_cc"],
            ascending=[False, True]
        ).reset_index(drop=True)

        best_row = round_df.iloc[0]
        improvement = best_row["cv_mean_cc"] - best_score

        if print_results:
            print(
                f"Step {step}: best add = {best_row['candidate_variable']}, "
                f"CC = {best_row['cc']:.4f}, "
                f"CV CC = {best_row['cv_mean_cc']:.4f}, "
                f"CV std = {best_row['cv_std_cc']:.4f}, "
                f"improvement = {improvement:.4f}"
            )

        if improvement > min_improvement:
            selected.append(best_row["candidate_variable"])
            remaining.remove(best_row["candidate_variable"])
            best_score = float(best_row["cv_mean_cc"])
            best_std = float(best_row["cv_std_cc"])

            history.append({
                "step": step,
                "selected_variable": best_row["candidate_variable"],
                "selected_variables_so_far": selected.copy(),
                "cc": float(best_row["cc"]),
                "cv_mean_cc": best_score,
                "cv_std_cc": best_std,
                "improvement": improvement,
            })
            step += 1
        else:
            if print_results:
                print("Stopping: no further improvement above threshold.")
            break

    history_df = pd.DataFrame(history)

    if print_results:
        print("\nSelected variables:")
        print(selected)

    return selected, history_df


# ============================================================
# 7. Step 3: CCA on selected variables
# ============================================================

def run_selected_variable_cca(
    data: pd.DataFrame,
    cluster_col: str,
    selected_variables: list[str],
    cv_splits: int = 5,
    random_state: int = 42,
    print_results: bool = True,
):
    results = evaluate_cca_cv(
        data=data,
        cluster_col=cluster_col,
        variable_cols=selected_variables,
        cv_splits=cv_splits,
        random_state=random_state,
    )
    if print_results:
        print("\n=== Step 3: CCA on selected variables ===")
        print(f"Selected variables: {selected_variables}")
        print(f"n_samples: {results['n_samples']}")
        print(f"CC1:          {results['cc']:.4f}")
        print(f"Mean CV CC1:  {results['cv_mean_cc']:.4f}")
        print(f"Std CV CC1:   {results['cv_std_cc']:.4f}")

    return {
        "cc": results["cc"],
        "cv_mean_cc": results["cv_mean_cc"],
        "cv_std_cc": results["cv_std_cc"],
    }


# ============================================================
# 8. Step 4/5: Sequential subject removal for CCA
# ============================================================

def sequential_subject_removal_cca(
    data: pd.DataFrame,
    cluster_col: str,
    selected_variables: list[str],
    cv_splits: int = 5,
    random_state: int = 42,
    subject_id_col: str | None = None,
    max_removals: int | None = None,
    min_improvement: float = 0.0,
    lambda_std: float = 0.5,
    print_results: bool = True,
    ):
    """
    Sequentially remove one subject at a time, worst first,
    stopping when CV mean CC1 no longer improves.

    Returns:
        dict with:
            cca_baseline_result
            cca_removal_history
            cca_removed_subjects_stepwise
            cca_removed_subjects_result
            cca_best_removed_subjects
            cca_best_data
    """
    current_data = data.copy()

    baseline_result = evaluate_cca_cv(
        data=current_data,
        cluster_col=cluster_col,
        variable_cols=selected_variables,
        cv_splits=cv_splits,
        random_state=random_state,
    )

    current_cv_mean = baseline_result["cv_mean_cc"]
    current_cv_std = baseline_result["cv_std_cc"]

    best_score = current_cv_mean - lambda_std * current_cv_std
    best_cv_mean = current_cv_mean
    best_cv_std = current_cv_std
    best_data = current_data.copy()
    best_removed_subjects = []

    if max_removals is None:
        max_removals = len(current_data) - (cv_splits + 2)

    history = []
    removed_subjects = []
    removed_subjects_stepwise = []

    if print_results:
        print("\n=== Step 4: Sequential subject removal with automatic stopping (CCA) ===")
        print(
            f"Start: n={len(current_data)}, "
            f"CC={baseline_result['cc']:.4f}, "
            f"CV CC={current_cv_mean:.4f}, "
            f"CV std={current_cv_std:.4f}"
        )

    for step in range(1, max_removals + 1):
        if len(current_data) <= cv_splits + 2:
            print("Stopping: too few subjects left.")
            break
        
        smallest_cluster_size = current_data[cluster_col].value_counts().min()
        if smallest_cluster_size < 2:
            print(f"Stopping: cluster sizes too small for CV (smallest cluster size={smallest_cluster_size}).")
            break

        candidate_rows = []

        for i in range(len(current_data)):
            reduced = current_data.drop(index=current_data.index[i])

            try:
                res = evaluate_cca_cv(
                    data=reduced,
                    cluster_col=cluster_col,
                    variable_cols=selected_variables,
                    cv_splits=cv_splits,
                    random_state=random_state,
                )
            except Exception:
                continue

            if subject_id_col is None:
                subject_id = current_data.index[i]
            else:
                subject_id = current_data.iloc[i][subject_id_col]

            if hasattr(subject_id, "item"):
                subject_id = subject_id.item()
            if isinstance(subject_id, float) and subject_id.is_integer():
                subject_id = int(subject_id)

            candidate_score = res["cv_mean_cc"] - lambda_std * res["cv_std_cc"]
            current_score = current_cv_mean - lambda_std * current_cv_std

            candidate_rows.append({
                "row_index": current_data.index[i],
                "subject_id": subject_id,
                "cc": res["cc"],
                "n_remaining": len(reduced),
                "cv_mean_cc": res["cv_mean_cc"],
                "cv_std_cc": res["cv_std_cc"],
                "score": candidate_score,
                "delta_score": candidate_score - current_score,
            })

        if not candidate_rows:
            print("Stopping: no valid candidates.")
            break

        candidates_df = pd.DataFrame(candidate_rows)
        candidates_df = candidates_df.sort_values(
            by=["cv_mean_cc", "cv_std_cc"],
            ascending=[False, True]
        ).reset_index(drop=True)

        best_candidate = candidates_df.iloc[0]

        if best_candidate["delta_score"] <= min_improvement:
            if print_results:
                print(
                    f"Stopping at step {step}: no further improvement. "
                    f"Best candidate delta_score = {best_candidate['delta_score']:.4f}"
                )
            break

        sid = best_candidate["subject_id"]
        if hasattr(sid, "item"):
            sid = sid.item()
        if isinstance(sid, float) and sid.is_integer():
            sid = int(sid)

        removed_subjects.append(sid)
        removed_subjects_stepwise.append(removed_subjects.copy())

        if print_results:
            print(
                f"Step {step}: remove {sid} | "
                f"n={best_candidate['n_remaining']} | "
                f"CC={best_candidate['cc']:.4f} | "
                f"CV CC={best_candidate['cv_mean_cc']:.4f} | "
                f"CV std={best_candidate['cv_std_cc']:.4f} | "
                f"delta_score={best_candidate['delta_score']:.4f}"
            )

        history.append({
            "step": step,
            "removed_subject": sid,
            "remaining_n": int(best_candidate["n_remaining"]),
            "cc": float(best_candidate["cc"]),
            "cv_mean_cc": float(best_candidate["cv_mean_cc"]),
            "cv_std_cc": float(best_candidate["cv_std_cc"]),
            "delta_score": float(best_candidate["delta_score"]),
            "removed_subjects_so_far": removed_subjects.copy(),
        })

        current_data = current_data.drop(index=best_candidate["row_index"]).copy()
        current_cv_mean = float(best_candidate["cv_mean_cc"])
        current_cv_std = float(best_candidate["cv_std_cc"])

        if current_score > best_score:
            best_score = current_score
            best_cv_mean = current_cv_mean
            best_cv_std = current_cv_std
            best_data = current_data.copy()
            best_removed_subjects = removed_subjects.copy()

    history_df = pd.DataFrame(history)

    best_result = evaluate_cca_cv(
        data=best_data,
        cluster_col=cluster_col,
        variable_cols=selected_variables,
        cv_splits=cv_splits,
        random_state=random_state,
    )

    if print_results:
        print("\n=== Step 5: CCA on removed subjects ===")
        print(f"Removed subjects: {best_removed_subjects}")
        print(f"Best n={len(best_data)}")
        print(f"Best CC={best_result['cc']:.4f}")
        print(f"Best CV CC={best_result['cv_mean_cc']:.4f}")
        print(f"Best CV std={best_result['cv_std_cc']:.4f}")

    return {
        "cca_baseline_result": {
            "cc": baseline_result["cc"],
            "cv_mean_cc": baseline_result["cv_mean_cc"],
            "cv_std_cc": baseline_result["cv_std_cc"],
        },
        "cca_removal_history": history_df,
        "cca_removed_subjects_stepwise": removed_subjects_stepwise,
        "cca_removed_subjects_result": {
            "cc": best_result["cc"],
            "cv_mean_cc": best_result["cv_mean_cc"],
            "cv_std_cc": best_result["cv_std_cc"],
        },
        "cca_best_removed_subjects": best_removed_subjects,
        "cca_best_data": best_data.reset_index(drop=True),
    }


# ============================================================
# 9. Full wrapper to mirror the MLR output structure
# ============================================================

def full_cca_selection_subject_pipeline(
    data: pd.DataFrame,
    cluster_col: str,
    candidate_variables: list[str],
    cv_splits: int = 5,
    random_state: int = 42,
    max_variables: int | None = None,
    min_improvement: float = 0.0,
    subject_id_col: str | None = None,
    print_results: bool = True,
):
    """
    Full ordered workflow:
    1. CCA on all variables
    2. Forward add-one-in variable selection
    3. CCA on selected variables
    4. Sequential subject removal with automatic stopping

    Returns:
        dict:
            cca_all_variables_results
            cca_selected_variables_results
            cca_removed_subjects_results
            cca_selected_variables
            cca_removed_subjects
    """
    all_var_results = run_full_cca(
        data=data,
        cluster_col=cluster_col,
        variable_cols=candidate_variables,
        cv_splits=cv_splits,
        random_state=random_state,
        print_results=print_results
    )

    selected_variables, selection_history = forward_select_variables_cca(
        data=data,
        cluster_col=cluster_col,
        candidate_variables=candidate_variables,
        cv_splits=cv_splits,
        random_state=random_state,
        max_variables=max_variables,
        min_improvement=min_improvement,
        print_results=print_results,
    )

    selected_var_results = run_selected_variable_cca(
        data=data,
        cluster_col=cluster_col,
        selected_variables=selected_variables,
        cv_splits=cv_splits,
        random_state=random_state,
        print_results=print_results,
    )

    removed_subjects_dict = sequential_subject_removal_cca(
        data=data,
        cluster_col=cluster_col,
        selected_variables=selected_variables,
        cv_splits=cv_splits,
        random_state=random_state,
        subject_id_col=subject_id_col,
        min_improvement=min_improvement,
        print_results=print_results,
    )

    removed_subjects_results = removed_subjects_dict["cca_removed_subjects_result"]
    removed_subjects = removed_subjects_dict["cca_best_removed_subjects"]

    return {
        "cca_all_variables_results": all_var_results,
        "cca_selected_variables_results": selected_var_results,
        "cca_removed_subjects_results": removed_subjects_results,
        "cca_selected_variables": selected_variables,
        "cca_removed_subjects": removed_subjects,
    }
    
    

# ============================================================
# 10. CCA using MLR-selected variables from an existing JSON
# ============================================================

def create_cca_from_mlr_selected_variables_json(
    data: pd.DataFrame,
    input_json_path: str | Path,
    output_json_path: str | Path,
    cluster_col_template: str = "Cluster_{k}",
    cv_splits: int = 5,
    random_state: int = 42,
    selected_variables_key: str = "mlr_selected_variables",
    print_results: bool = True,
):
    """
    Read MLR-selected variables from an existing results JSON file and run CCA
    using only those variables.

    This function creates a new JSON file. It does not modify, append to, or
    overwrite the input JSON file.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the cluster label columns and behavioural variables.

    input_json_path : str | Path
        Existing JSON file containing mlr_selected_variables for each cluster.

    output_json_path : str | Path
        New JSON file to save the CCA results to.

    cluster_col_template : str
        Template for cluster columns in `data`.

        Default assumes columns named:
            Cluster_2, Cluster_3, ..., Cluster_10

        Example alternatives:
            "cluster_{k}"
            "DCEC_{k}_labels"

    cv_splits : int
        Number of folds for cross-validated CCA.

    random_state : int
        Random seed for cross-validation.

    selected_variables_key : str
        JSON key containing the MLR-selected variables.

    print_results : bool
        Whether to print progress.

    Returns
    -------
    new_results : dict
        New JSON-compatible results dictionary.

    summary_df : pd.DataFrame
        Summary table with one row per cluster.
    """

    input_json_path = Path(input_json_path)
    output_json_path = Path(output_json_path)

    if input_json_path.resolve() == output_json_path.resolve():
        raise ValueError(
            "input_json_path and output_json_path are the same. "
            "Choose a different output path to avoid overwriting the original JSON."
        )

    with input_json_path.open("r", encoding="utf-8") as f:
        old_results = json.load(f)

    def _cluster_number_from_key(key: str) -> int | None:
        match = re.search(r"Cluster_(\d+)_results", key)
        if match is None:
            return None
        return int(match.group(1))

    def _json_safe(obj):
        """
        Convert NumPy/Pandas values to JSON-safe Python types.
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

    new_results = {
        "source_json": str(input_json_path),
        "analysis": "CCA performed using MLR-selected variables",
        "cv_splits": cv_splits,
        "random_state": random_state,
        "cluster_col_template": cluster_col_template,
        "results": {}
    }

    
    
    summary_rows = []

    cluster_items = []
    for cluster_key, cluster_result in old_results.items():
        k = _cluster_number_from_key(cluster_key)
        if k is not None:
            cluster_items.append((k, cluster_key, cluster_result))

    cluster_items = sorted(cluster_items, key=lambda x: x[0])

    for k, old_cluster_key, old_cluster_result in cluster_items:
        new_cluster_key = f"Cluster_{k}_results"
        cluster_col = cluster_col_template.format(k=k)

        mlr_selected_variables = old_cluster_result.get(selected_variables_key, [])
        mlr_selected_variables = list(dict.fromkeys(mlr_selected_variables))

        missing_variables = [
            var for var in mlr_selected_variables
            if var not in data.columns
        ]

        available_variables = [
            var for var in mlr_selected_variables
            if var in data.columns
        ]

        non_numeric_variables = [
            var for var in available_variables
            if not pd.api.types.is_numeric_dtype(data[var])
        ]

        if print_results:
            print(f"\n=== Cluster {k}: CCA using MLR-selected variables ===")
            print(f"Cluster column: {cluster_col}")
            print(f"MLR-selected variables: {mlr_selected_variables}")

            if missing_variables:
                print(f"Missing variables skipped: {missing_variables}")

            if non_numeric_variables:
                print(f"Non-numeric variables ignored by CCA: {non_numeric_variables}")

        cluster_output = {
            "cluster_number": k,
            "cluster_col": cluster_col,
            "mlr_selected_variables": mlr_selected_variables,
            "missing_variables": missing_variables,
            "non_numeric_variables": non_numeric_variables,
        }

        if cluster_col not in data.columns:
            error_msg = f"Cluster column '{cluster_col}' not found in data."

            cluster_output["cca_mlr_selected_variables_results"] = {
                "error": error_msg,
                "cc": None,
                "cv_mean_cc": None,
                "cv_std_cc": None,
                "n_samples": None,
                "variables_used": [],
            }

            if print_results:
                print(f"Skipping: {error_msg}")

        elif not available_variables:
            error_msg = "None of the MLR-selected variables were found in data."

            cluster_output["cca_mlr_selected_variables_results"] = {
                "error": error_msg,
                "cc": None,
                "cv_mean_cc": None,
                "cv_std_cc": None,
                "n_samples": None,
                "variables_used": [],
            }

            if print_results:
                print(f"Skipping: {error_msg}")

        else:
            try:
                cca_result = evaluate_cca_cv(
                    data=data,
                    cluster_col=cluster_col,
                    variable_cols=available_variables,
                    cv_splits=cv_splits,
                    random_state=random_state,
                )

                cluster_output["cca_mlr_selected_variables_results"] = {
                    "cc": cca_result["cc"],
                    "cv_mean_cc": cca_result["cv_mean_cc"],
                    "cv_std_cc": cca_result["cv_std_cc"],
                    "n_samples": cca_result["n_samples"],
                    "variables_used": cca_result["variables_used"],
                    "error": None,
                }

                if print_results:
                    print(f"Variables used: {cca_result['variables_used']}")
                    print(f"n_samples:     {cca_result['n_samples']}")
                    print(f"CC1:           {cca_result['cc']:.4f}")
                    print(f"Mean CV CC1:   {cca_result['cv_mean_cc']:.4f}")
                    print(f"Std CV CC1:    {cca_result['cv_std_cc']:.4f}")

            except Exception as e:
                error_msg = str(e)

                cluster_output["cca_mlr_selected_variables_results"] = {
                    "error": error_msg,
                    "cc": None,
                    "cv_mean_cc": None,
                    "cv_std_cc": None,
                    "n_samples": None,
                    "variables_used": [],
                }

                if print_results:
                    print(f"CCA failed: {error_msg}")

        new_results["results"][new_cluster_key] = cluster_output

        cca_output = cluster_output["cca_mlr_selected_variables_results"]

        summary_rows.append({
            "cluster_number": k,
            "cluster_col": cluster_col,
            "cc": cca_output["cc"],
            "cv_mean_cc": cca_output["cv_mean_cc"],
            "cv_std_cc": cca_output["cv_std_cc"],
            "n_samples": cca_output["n_samples"],
            "mlr_selected_variables": mlr_selected_variables,
            "variables_used": cca_output["variables_used"],
            "missing_variables": missing_variables,
            "non_numeric_variables": non_numeric_variables,
            "error": cca_output["error"],
        })

    summary_df = pd.DataFrame(summary_rows)

    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(new_results), f, indent=4)

    if print_results:
        print(f"\nSaved new CCA results JSON to: {output_json_path}")

    return new_results, summary_df


def permutation_test_cca_cv(
    data: pd.DataFrame,
    cluster_col: str,
    variable_cols: list[str],
    cv_splits: int = 5,
    random_state: int = 42,
    n_permutations: int = 1000,
    score_key: str = "cv_mean_cc",
    alternative: str = "greater",
    print_results: bool = True,
):
    """
    Permutation test for CCA.

    Tests whether the observed CCA score is higher than expected when the
    cluster labels are randomly shuffled.

    Parameters
    ----------
    score_key : str
        Which CCA score to test.
        Recommended:
            "cv_mean_cc"

        Alternative:
            "cc"

    alternative : str
        Usually "greater", because stronger CCA association means higher score.

    Returns
    -------
    dict with observed score, null distribution and p-value.
    """

    def permutation_p_value(
        observed_score: float,
        null_scores: list[float] | np.ndarray,
        alternative: str = "greater",
    ):
        """
        Compute permutation p-value.

        alternative:
            "greater"  -> p = P(null >= observed)
            "less"     -> p = P(null <= observed)
            "two-sided" -> p = P(abs(null) >= abs(observed))
        """
        null_scores = np.asarray(null_scores, dtype=float)
        null_scores = null_scores[~np.isnan(null_scores)]

        if len(null_scores) == 0:
            return np.nan

        if alternative == "greater":
            count = np.sum(null_scores >= observed_score)
        elif alternative == "less":
            count = np.sum(null_scores <= observed_score)
        elif alternative == "two-sided":
            count = np.sum(np.abs(null_scores) >= abs(observed_score))
        else:
            raise ValueError("alternative must be 'greater', 'less', or 'two-sided'.")

        return float((count + 1) / (len(null_scores) + 1))
        
    observed_result = evaluate_cca_cv(
        data=data,
        cluster_col=cluster_col,
        variable_cols=variable_cols,
        cv_splits=cv_splits,
        random_state=random_state,
    )

    observed_score = observed_result[score_key]

    rng = np.random.default_rng(random_state)
    null_scores = []

    valid_mask = data[cluster_col].notna()

    for perm_idx in range(n_permutations):
        permuted_data = data.copy()

        original_labels = permuted_data.loc[valid_mask, cluster_col].to_numpy()
        permuted_labels = rng.permutation(original_labels)

        permuted_data.loc[valid_mask, cluster_col] = permuted_labels

        try:
            perm_result = evaluate_cca_cv(
                data=permuted_data,
                cluster_col=cluster_col,
                variable_cols=variable_cols,
                cv_splits=cv_splits,
                random_state=random_state,
            )

            null_scores.append(perm_result[score_key])

        except Exception:
            null_scores.append(np.nan)

    null_scores = np.asarray(null_scores, dtype=float)

    p_value = permutation_p_value(
        observed_score=observed_score,
        null_scores=null_scores,
        alternative=alternative,
    )

    result = {
        "observed_result": observed_result,
        "tested_score": score_key,
        "observed_score": float(observed_score),
        "p_value": float(p_value) if not np.isnan(p_value) else None,
        "n_permutations": int(n_permutations),
        "n_valid_permutations": int(np.sum(~np.isnan(null_scores))),
        "null_mean": float(np.nanmean(null_scores)),
        "null_std": float(np.nanstd(null_scores)),
        "null_scores": null_scores.tolist(),
        "alternative": alternative,
    }

    if print_results:
        print(f"\n=== Permutation test CCA: {cluster_col} ===")
        print(f"Variables: {variable_cols}")
        print(f"Tested score: {score_key}")
        print(f"Observed score: {observed_score:.4f}")
        print(f"Null mean: {result['null_mean']:.4f}")
        print(f"Null std: {result['null_std']:.4f}")
        print(f"p-value: {p_value:.4f}")

    return result