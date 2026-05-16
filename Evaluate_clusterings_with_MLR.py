import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
import json
import re
from pathlib import Path
from sklearn.metrics import accuracy_score

# ============================================================
# 1. Core helper: build preprocessing + multinomial logistic regression
# ============================================================

def build_mlr_pipeline(X: pd.DataFrame):
    """
    Build a pipeline with:
    - median imputation + scaling for numeric variables
    - most-frequent imputation + one-hot encoding for categorical variables
    - multinomial logistic regression

    Works across sklearn versions by avoiding deprecated multi_class arg.
    """
    numeric_features = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    try:
        categorical_onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        categorical_onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", categorical_onehot)
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    model = LogisticRegression(
        max_iter=5000,
        solver="lbfgs"
    )

    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])

    return pipe


# ============================================================
# 2. Evaluate one variable set with CV accuracy
# ============================================================

def evaluate_mlr_cv(
    data: pd.DataFrame,
    cluster_col: str,
    variable_cols: list[str],
    cv_splits: int = 5,
    random_state: int = 42,
):
    """
    Cross-validated multinomial logistic regression accuracy.

    Returns:
        dict with mean accuracy, std, fold scores, chance accuracy
    """
    df = data[[cluster_col] + variable_cols].copy()
    df = df.dropna(subset=[cluster_col])

    if df.empty:
        raise ValueError("No valid rows left after dropping missing cluster labels.")

    X = df[variable_cols].copy()
    y = df[cluster_col].astype(str).copy()


    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    n_classes = len(np.unique(y_encoded))
    class_counts = np.bincount(y_encoded)
    majority_class_accuracy = class_counts.max() / class_counts.sum()

    pipe = build_mlr_pipeline(X)
    
    min_class_count = class_counts.min()
    if min_class_count < cv_splits:
        print(f"Warning: Reducing cv_splits from {cv_splits} to {max(min_class_count, 3)} due to class imbalance.")
        cv_splits = max(min_class_count, 3)

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(pipe, X, y_encoded, cv=cv, scoring="accuracy")

    return {
        "mean_accuracy": float(np.mean(scores)),
        "std_accuracy": float(np.std(scores)),
        "fold_scores": scores,
        "chance_accuracy_majority_class": float(majority_class_accuracy),
        "n_classes": n_classes,
        "class_labels": list(label_encoder.classes_),
        "n_samples": len(df),
    }


# ============================================================
# 3. Step 1: Perform MLR on all variables
# ============================================================

def run_full_mlr(
    data: pd.DataFrame,
    cluster_col: str,
    variable_cols: list[str],
    cv_splits: int = 5,
    random_state: int = 42,
    print_results: bool = True,
):
    results = evaluate_mlr_cv(
        data=data,
        cluster_col=cluster_col,
        variable_cols=variable_cols,
        cv_splits=cv_splits,
        random_state=random_state,
    )

    if print_results:
        print("\n=== Step 1: MLR on all variables ===")
        print(f"n_samples: {results['n_samples']}")
        print(f"Mean CV accuracy: {results['mean_accuracy']:.4f}")
        print(f"Std CV accuracy:  {results['std_accuracy']:.4f}")
        print(f"Chance accuracy:  {results['chance_accuracy_majority_class']:.4f}")

    # Only return the results mean_accuracy, std_accuracy, chance_accuracy_majority_class for the all-variable model
    return {
        "mean_accuracy": results["mean_accuracy"],
        "std_accuracy": results["std_accuracy"],
        "chance_accuracy_majority_class": results["chance_accuracy_majority_class"]
    }


# ============================================================
# 4. Step 2: Add-one-in forward selection
# ============================================================

def forward_select_variables_mlr(
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
    Greedy forward selection:
    - Start with no variables
    - At each step, add the variable that gives the highest CV accuracy
    - Stop when improvement <= min_improvement or max_variables reached

    Returns:
        selected_variables, history_df
    """
    if max_variables is None:
        max_variables = len(candidate_variables)

    selected = []
    remaining = candidate_variables.copy()
    history = []

    best_score = 0.0
    best_std = np.nan

    if print_results:
        print("\n=== Step 2: Forward add-one-in variable selection ===")

    step = 1
    while remaining and len(selected) < max_variables:
        round_results = []

        for var in remaining:
            vars_try = selected + [var]

            res = evaluate_mlr_cv(
                data=data,
                cluster_col=cluster_col,
                variable_cols=vars_try,
                cv_splits=cv_splits,
                random_state=random_state,
            )

            round_results.append({
                "step": step,
                "candidate_variable": var,
                "variables_if_added": vars_try.copy(),
                "mean_accuracy": res["mean_accuracy"],
                "std_accuracy": res["std_accuracy"],
            })

        round_df = pd.DataFrame(round_results)
        round_df = round_df.sort_values(
            by=["mean_accuracy", "std_accuracy"],
            ascending=[False, True]
        ).reset_index(drop=True)

        best_row = round_df.iloc[0]
        improvement = best_row["mean_accuracy"] - best_score

        if print_results:
            print(
                f"Step {step}: best add = {best_row['candidate_variable']}, "
                f"CV acc = {best_row['mean_accuracy']:.4f}, "
                f"std = {best_row['std_accuracy']:.4f}, "
                f"improvement = {improvement:.4f}"
            )

        if improvement > min_improvement:
            selected.append(best_row["candidate_variable"])
            remaining.remove(best_row["candidate_variable"])
            best_score = float(best_row["mean_accuracy"])
            best_std = float(best_row["std_accuracy"])

            history.append({
                "step": step,
                "selected_variable": best_row["candidate_variable"],
                "selected_variables_so_far": selected.copy(),
                "mean_accuracy": best_score,
                "std_accuracy": best_std,
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
# 5. Step 3: Perform MLR on selected variables
# ============================================================

def run_selected_variable_mlr(
    data: pd.DataFrame,
    cluster_col: str,
    selected_variables: list[str],
    cv_splits: int = 5,
    random_state: int = 42,
    print_results: bool = True,
):
    results = evaluate_mlr_cv(
        data=data,
        cluster_col=cluster_col,
        variable_cols=selected_variables,
        cv_splits=cv_splits,
        random_state=random_state,
    )

    if print_results:
        print("\n=== Step 3: MLR on selected variables ===")
        print(f"Selected variables: {selected_variables}")
        print(f"n_samples: {results['n_samples']}")
        print(f"Mean CV accuracy: {results['mean_accuracy']:.4f}")
        print(f"Std CV accuracy:  {results['std_accuracy']:.4f}")
        print(f"Chance accuracy:  {results['chance_accuracy_majority_class']:.4f}")

    # Only return the results mean_accuracy, std_accuracy, chance_accuracy_majority_class for the selected-variable model
    return {
        "mean_accuracy": results["mean_accuracy"],
        "std_accuracy": results["std_accuracy"],
        "chance_accuracy_majority_class": results["chance_accuracy_majority_class"]
    }


# ============================================================
# 6. Step 4: Leave-one-out influence to find worst subjects
# ============================================================

def leave_one_out_subject_influence_mlr(
    data: pd.DataFrame,
    cluster_col: str,
    selected_variables: list[str],
    cv_splits: int = 5,
    random_state: int = 42,
    subject_id_col: str | None = None,
    print_results: bool = True,
):
    """
    For each subject:
    - remove the subject
    - rerun CV MLR on selected variables
    - compare accuracy and std to the baseline

    Positive delta_accuracy means removing the subject improves performance.
    Negative delta_std means removing the subject reduces unnecessary variance.

    Returns:
        baseline_results, influence_df
    """
    baseline = evaluate_mlr_cv(
        data=data,
        cluster_col=cluster_col,
        variable_cols=selected_variables,
        cv_splits=cv_splits,
        random_state=random_state,
    )

    rows = []

    if print_results:
        print("\n=== Step 4: Leave-one-out subject influence ===")
        print(
            f"Baseline selected-variable model: "
            f"acc = {baseline['mean_accuracy']:.4f}, std = {baseline['std_accuracy']:.4f}"
        )

    for i in range(len(data)):
        reduced = data.drop(index=data.index[i])

        res = evaluate_mlr_cv(
            data=reduced,
            cluster_col=cluster_col,
            variable_cols=selected_variables,
            cv_splits=cv_splits,
            random_state=random_state,
        )

        if subject_id_col is None:
            subject_id = data.index[i]
        else:
            subject_id = data.iloc[i][subject_id_col]

        rows.append({
            "row_index": data.index[i],
            "subject_id": subject_id,
            "loo_mean_accuracy": res["mean_accuracy"],
            "loo_std_accuracy": res["std_accuracy"],
            "delta_accuracy": res["mean_accuracy"] - baseline["mean_accuracy"],
            "delta_std": res["std_accuracy"] - baseline["std_accuracy"],
        })

    influence_df = pd.DataFrame(rows)

    # Combined ranking:
    # worst subjects are those whose removal improves accuracy and reduces std
    influence_df["influence_score"] = (
        influence_df["delta_accuracy"] - influence_df["delta_std"]
    )

    influence_df = influence_df.sort_values(
        by=["influence_score", "delta_accuracy"],
        ascending=[False, False]
    ).reset_index(drop=True)

    if print_results:
        print("\nTop subjects whose removal improves the model most:")
        print(influence_df.head(10))

    return baseline, influence_df


# ============================================================
# 7. Step 5: Perform MLR on selected variables and reduced subjects
# ============================================================

def run_reduced_subject_mlr(
    data: pd.DataFrame,
    cluster_col: str,
    selected_variables: list[str],
    subjects_to_remove: list,
    cv_splits: int = 5,
    random_state: int = 42,
    subject_id_col: str | None = None,
    print_results: bool = True,
):
    """
    Remove manually chosen subjects, then rerun CV MLR on selected variables.

    subjects_to_remove:
    - if subject_id_col is None: interpreted as dataframe row indices
    - otherwise interpreted as values in subject_id_col
    """
    if subject_id_col is None:
        reduced_data = data.drop(index=subjects_to_remove).copy()
    else:
        reduced_data = data.loc[~data[subject_id_col].isin(subjects_to_remove)].copy()

    reduced_data = reduced_data.reset_index(drop=True)

    results = evaluate_mlr_cv(
        data=reduced_data,
        cluster_col=cluster_col,
        variable_cols=selected_variables,
        cv_splits=cv_splits,
        random_state=random_state,
    )

    if print_results:
        print("\n=== Step 5: MLR on selected variables and reduced subjects ===")
        print(f"Removed subjects: {subjects_to_remove}")
        print(f"Remaining n_samples: {results['n_samples']}")
        print(f"Mean CV accuracy: {results['mean_accuracy']:.4f}")
        print(f"Std CV accuracy:  {results['std_accuracy']:.4f}")
        print(f"Chance accuracy:  {results['chance_accuracy_majority_class']:.4f}")

    return reduced_data, results


def sequential_subject_removal_mlr(
    data: pd.DataFrame,
    cluster_col: str,
    selected_variables: list[str],
    cv_splits: int = 5,
    random_state: int = 42,
    subject_id_col: str | None = None,
    max_removals: int | None = None,
    min_improvement: float = 0.0,
    print_results: bool = True,
    lambda_std: float = 0.5
):
    """
    Sequentially remove one subject at a time, always removing the current
    worst subject first, and stop automatically when CV accuracy does not
    improve anymore.

    "Worst" subject at each step = the subject whose removal gives the
    highest cross-validated MLR accuracy on the remaining dataset.
    If two removals give the same accuracy, the one with lower CV std wins.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe
    cluster_col : str
        Column with cluster labels
    selected_variables : list[str]
        Variables used in the MLR model
    cv_splits : int
        Number of CV folds
    random_state : int
        Random state for CV
    subject_id_col : str | None
        Optional column with subject IDs. If None, dataframe index is used.
    max_removals : int | None
        Maximum number of subjects to remove
    min_improvement : float
        Minimum required increase in mean CV accuracy to accept a removal

    Returns
    -------
    dict with:
        baseline_result
        removal_history
        removed_subjects_stepwise
        removed_subjects_final
        best_result
        best_removed_subjects
        best_data
        final_data
    """
    current_data = data.copy()

    baseline_result = evaluate_mlr_cv(
        data=current_data,
        cluster_col=cluster_col,
        variable_cols=selected_variables,
        cv_splits=cv_splits,
        random_state=random_state,
    )

    current_acc = baseline_result["mean_accuracy"]
    current_std = baseline_result["std_accuracy"]

    best_acc = current_acc
    best_std = current_std
    best_score = current_acc - current_std * lambda_std
    best_data = current_data.copy()
    best_removed_subjects = []

    if max_removals is None:
        max_removals = len(current_data) - (cv_splits + 2)

    history = []
    removed_subjects = []
    removed_subjects_stepwise = []

    if print_results:
        print("\n=== Step 4: Sequential subject removal with automatic stopping ===")
        print(
            f"Start: n={len(current_data)}, "
            f"acc={current_acc:.4f}, std={current_std:.4f}"
        )

    for step in range(1, max_removals + 1):

        if len(current_data) <= cv_splits + 2:
            print("Stopping: too few subjects left.")
            break

        candidate_rows = []

        for i in range(len(current_data)):
            reduced = current_data.drop(index=current_data.index[i])

            try:
                res = evaluate_mlr_cv(
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
                subject_id = int(subject_id)

            # Convert numpy types to native Python types
            if hasattr(subject_id, "item"):
                subject_id = subject_id.item()

            current_score = float(res["mean_accuracy"]) - float(res["std_accuracy"]) * lambda_std
            delta_score = current_score - float(baseline_result["mean_accuracy"])

            candidate_rows.append({
                "row_index": current_data.index[i],
                "subject_id": subject_id,
                "mean_accuracy": res["mean_accuracy"],
                "std_accuracy": res["std_accuracy"],
                "n_remaining": len(reduced),
                "delta_accuracy": res["mean_accuracy"] - current_acc,
                "delta_std": res["std_accuracy"] - current_std,
                "delta_score": delta_score,
            })

        if not candidate_rows:
            print("Stopping: no valid candidates.")
            break

        candidates_df = pd.DataFrame(candidate_rows)

        # Pick best candidate: highest accuracy, then lowest std
        candidates_df = candidates_df.sort_values(
            by=["mean_accuracy", "std_accuracy"],
            ascending=[False, True]
        ).reset_index(drop=True)

        #best_idx = candidates_df.index[0]
        #best_candidate = candidates_df.loc[best_idx]
        best_candidate = candidates_df.iloc[0]

        # Stop if no improvement
        if best_candidate["delta_score"] <= min_improvement:
            if print_results:
                print(
                    f"Stopping at step {step}: no further improvement. "
                    f"Best candidate delta_acc = {best_candidate['delta_accuracy']:.4f}"
                    f"delta_score = {best_candidate['delta_score']:.4f}"
                )
            break

        # Accept removal
        sid = best_candidate["subject_id"]
        sid = sid.item() if hasattr(sid, "item") else sid
        sid = int(sid)
        removed_subjects.append(sid)
        removed_subjects_stepwise.append(removed_subjects.copy())

        if print_results:
            print(
                f"Step {step}: remove {sid} | "
                f"n={best_candidate['n_remaining']} | "
                f"acc={best_candidate['mean_accuracy']:.4f} | "
                f"std={best_candidate['std_accuracy']:.4f} | "
                f"delta_acc={best_candidate['delta_accuracy']:.4f}"
                f"delta_score={best_candidate['delta_score']:.4f}"
            )

        history.append({
            "step": step,
            "removed_subject": sid,
            "remaining_n": best_candidate["n_remaining"],
            "mean_accuracy": best_candidate["mean_accuracy"],
            "std_accuracy": best_candidate["std_accuracy"],
            "delta_accuracy": best_candidate["delta_accuracy"],
            "delta_std": best_candidate["delta_std"],
            "removed_subjects_so_far": removed_subjects.copy(),
            "delta_score": best_candidate["delta_score"],
        })

        # Permanently remove from current dataset
        current_data = current_data.drop(index=best_candidate["row_index"]).copy()
        current_acc = float(best_candidate["mean_accuracy"])
        current_std = float(best_candidate["std_accuracy"])
        current_score = float(best_candidate["delta_score"])

        # Update best dataset
        if current_score > best_score:
            best_score = current_score
            best_acc = current_acc
            best_std = current_std
            best_data = current_data.copy()
            best_removed_subjects = removed_subjects.copy()

    history_df = pd.DataFrame(history)

    best_result = evaluate_mlr_cv(
        data=best_data,
        cluster_col=cluster_col,
        variable_cols=selected_variables,
        cv_splits=cv_splits,
        random_state=random_state,
    )

    if print_results:
        print("\n=== Step 5: MLR on removed subjects ===")
        print(f"Removed subjects: {best_removed_subjects}")
        print(f"Best n={len(best_data)}")
        print(f"Best acc={best_result['mean_accuracy']:.4f}")
        print(f"Best std={best_result['std_accuracy']:.4f}")

    return {
        "mlr_baseline_result": {"mean_accuracy": baseline_result["mean_accuracy"], "std_accuracy": baseline_result["std_accuracy"], "chance_accuracy_majority_class": baseline_result["chance_accuracy_majority_class"]},
        "mlr_removal_history": history_df,
        "mlr_removed_subjects_stepwise": removed_subjects_stepwise,
        "mlr_removed_subjects_result": {"mean_accuracy": best_result["mean_accuracy"], "std_accuracy": best_result["std_accuracy"], "chance_accuracy_majority_class": best_result["chance_accuracy_majority_class"]},
        "mlr_best_removed_subjects": best_removed_subjects,
        "mlr_best_data": best_data.reset_index(drop=True)
    }


# ============================================================
# 8. One wrapper to run the whole pipeline
# ============================================================

def full_mlr_selection_subject_pipeline(
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
    1. MLR on all variables
    2. Forward add-one-in selection
    3. MLR on selected variables
    4. Leave-one-out subject influence
    5. MLR on selected variables and reduced subjects (if subjects_to_remove is given)

    Returns:
        dict:
            mlr_all_variables_results
            mlr_selected_variables_results
            mlr_removed_subjects_results
            mlr_selected_variables
            mlr_removed_subjects
    """

    all_var_results = run_full_mlr(
        data=data,
        cluster_col=cluster_col,
        variable_cols=candidate_variables,
        cv_splits=cv_splits,
        random_state=random_state,
        print_results=print_results
    )

    selected_variables, selection_history = forward_select_variables_mlr(
        data=data,
        cluster_col=cluster_col,
        candidate_variables=candidate_variables,
        cv_splits=cv_splits,
        random_state=random_state,
        max_variables=max_variables,
        min_improvement=min_improvement,
        print_results=print_results 
    )

    selected_var_results = run_selected_variable_mlr(
        data=data,
        cluster_col=cluster_col,
        selected_variables=selected_variables,
        cv_splits=cv_splits,
        random_state=random_state,
        print_results=print_results
    )

    """ baseline_selected, influence_df = leave_one_out_subject_influence_mlr(
        data=data,
        cluster_col=cluster_col,
        selected_variables=selected_variables,
        cv_splits=cv_splits,
        random_state=random_state,
        subject_id_col=subject_id_col,
    ) """



    removed_subjects_dict = sequential_subject_removal_mlr(
        data=data,
        cluster_col=cluster_col,
        selected_variables=selected_variables,
        cv_splits=cv_splits,
        random_state=random_state,
        subject_id_col=subject_id_col,
        print_results=print_results
    )

    reduced_results = removed_subjects_dict["mlr_removed_subjects_result"]
    removed_subjects = removed_subjects_dict["mlr_best_removed_subjects"]

    return {
        "mlr_all_variables_results": all_var_results,
        #"mlr_selection_history": selection_history,
        "mlr_selected_variables_results": selected_var_results,
        #"mlr_baseline_selected_results": baseline_selected,
        #"mlr_subject_influence": influence_df,
        #"mlr_reduced_data": reduced_data,
        "mlr_removed_subjects_results": reduced_results,
        "mlr_selected_variables": selected_variables,
        "mlr_removed_subjects": removed_subjects
    }


# ============================================================
# MLR using CCA-selected variables from existing JSON
# ============================================================

def _evaluate_mlr_cv_basic(
    data: pd.DataFrame,
    cluster_col: str,
    variable_cols: list[str],
    cv_splits: int = 5,
    random_state: int = 42,
    max_iter: int = 1000,
):
    """
    Basic cross-validated multinomial logistic regression.

    Predicts cluster labels from behavioural variables.

    Returns:
        dict with:
            mean_accuracy
            std_accuracy
            chance_accuracy_majority_class
            n_samples
            variables_used
            fold_accuracies
    """

    df = data[[cluster_col] + variable_cols].copy()
    df = df.dropna(subset=[cluster_col]).reset_index(drop=True)

    if df.empty:
        raise ValueError("No rows left after removing missing cluster labels.")

    if df[cluster_col].nunique() < 2:
        raise ValueError("MLR needs at least two unique cluster labels.")

    kept_cols = [
        col for col in variable_cols
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
    ]

    if not kept_cols:
        raise ValueError("No numeric independent variables found for MLR.")

    X = df[kept_cols].to_numpy()
    y = df[cluster_col].astype(str).to_numpy()

    class_counts = pd.Series(y).value_counts()
    min_class_count = int(class_counts.min())

    if min_class_count < 2:
        raise ValueError(
            f"Smallest cluster has only {min_class_count} subject. "
            "Stratified CV needs at least 2 subjects per class."
        )

    n_splits = min(cv_splits, min_class_count)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    fold_accuracies = []

    for train_idx, test_idx in skf.split(X, y):
        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        X_train = scaler.fit_transform(imputer.fit_transform(X_train))
        X_test = scaler.transform(imputer.transform(X_test))

        model = LogisticRegression(
            max_iter=max_iter,
            solver="lbfgs",
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        fold_accuracies.append(float(accuracy_score(y_test, y_pred)))

    chance_accuracy = float(class_counts.max() / class_counts.sum())

    return {
        "mean_accuracy": float(np.mean(fold_accuracies)),
        "std_accuracy": float(np.std(fold_accuracies)),
        "chance_accuracy_majority_class": chance_accuracy,
        "n_samples": int(len(df)),
        "variables_used": kept_cols,
        "fold_accuracies": fold_accuracies,
        "n_splits": int(n_splits),
    }


def create_mlr_from_cca_selected_variables_json(
    data: pd.DataFrame,
    input_json_path: str | Path,
    output_json_path: str | Path,
    cluster_col_template: str = "Cluster_{k}",
    cv_splits: int = 5,
    random_state: int = 42,
    selected_variables_key: str = "cca_selected_variables",
    print_results: bool = True,
):
    """
    Read CCA-selected variables from an existing results JSON file and run MLR
    using only those variables.

    This function creates a new JSON file. It does not modify, append to, or
    overwrite the input JSON file.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing cluster label columns and behavioural variables.

    input_json_path : str | Path
        Existing JSON file containing cca_selected_variables for each cluster.

    output_json_path : str | Path
        New JSON file to save the MLR results to.

    cluster_col_template : str
        Template for cluster columns in `data`.

        Default assumes columns named:
            Cluster_2, Cluster_3, ..., Cluster_10

    cv_splits : int
        Number of folds for cross-validated MLR.

    random_state : int
        Random seed for cross-validation.

    selected_variables_key : str
        JSON key containing the CCA-selected variables.

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
        "analysis": "MLR performed using CCA-selected variables",
        "cv_splits": cv_splits,
        "random_state": random_state,
        "cluster_col_template": cluster_col_template,
        "results": {},
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

        cca_selected_variables = old_cluster_result.get(selected_variables_key, [])
        cca_selected_variables = list(dict.fromkeys(cca_selected_variables))

        missing_variables = [
            var for var in cca_selected_variables
            if var not in data.columns
        ]

        available_variables = [
            var for var in cca_selected_variables
            if var in data.columns
        ]

        non_numeric_variables = [
            var for var in available_variables
            if not pd.api.types.is_numeric_dtype(data[var])
        ]

        numeric_available_variables = [
            var for var in available_variables
            if pd.api.types.is_numeric_dtype(data[var])
        ]

        if print_results:
            print(f"\n=== Cluster {k}: MLR using CCA-selected variables ===")
            print(f"Cluster column: {cluster_col}")
            print(f"CCA-selected variables: {cca_selected_variables}")

            if missing_variables:
                print(f"Missing variables skipped: {missing_variables}")

            if non_numeric_variables:
                print(f"Non-numeric variables skipped: {non_numeric_variables}")

        cluster_output = {
            "cluster_number": k,
            "cluster_col": cluster_col,
            "cca_selected_variables": cca_selected_variables,
            "missing_variables": missing_variables,
            "non_numeric_variables": non_numeric_variables,
        }

        if cluster_col not in data.columns:
            error_msg = f"Cluster column '{cluster_col}' not found in data."

            cluster_output["mlr_cca_selected_variables_results"] = {
                "mean_accuracy": None,
                "std_accuracy": None,
                "chance_accuracy_majority_class": None,
                "n_samples": None,
                "variables_used": [],
                "fold_accuracies": [],
                "n_splits": None,
                "error": error_msg,
            }

            if print_results:
                print(f"Skipping: {error_msg}")

        elif not numeric_available_variables:
            error_msg = "No usable numeric CCA-selected variables found in data."

            cluster_output["mlr_cca_selected_variables_results"] = {
                "mean_accuracy": None,
                "std_accuracy": None,
                "chance_accuracy_majority_class": None,
                "n_samples": None,
                "variables_used": [],
                "fold_accuracies": [],
                "n_splits": None,
                "error": error_msg,
            }

            if print_results:
                print(f"Skipping: {error_msg}")

        else:
            try:
                mlr_result = _evaluate_mlr_cv_basic(
                    data=data,
                    cluster_col=cluster_col,
                    variable_cols=numeric_available_variables,
                    cv_splits=cv_splits,
                    random_state=random_state,
                )

                cluster_output["mlr_cca_selected_variables_results"] = {
                    "mean_accuracy": mlr_result["mean_accuracy"],
                    "std_accuracy": mlr_result["std_accuracy"],
                    "chance_accuracy_majority_class": mlr_result["chance_accuracy_majority_class"],
                    "n_samples": mlr_result["n_samples"],
                    "variables_used": mlr_result["variables_used"],
                    "fold_accuracies": mlr_result["fold_accuracies"],
                    "n_splits": mlr_result["n_splits"],
                    "error": None,
                }

                if print_results:
                    print(f"Variables used: {mlr_result['variables_used']}")
                    print(f"n_samples:     {mlr_result['n_samples']}")
                    print(f"Mean accuracy: {mlr_result['mean_accuracy']:.4f}")
                    print(f"Std accuracy:  {mlr_result['std_accuracy']:.4f}")
                    print(f"Chance acc.:   {mlr_result['chance_accuracy_majority_class']:.4f}")

            except Exception as e:
                error_msg = str(e)

                cluster_output["mlr_cca_selected_variables_results"] = {
                    "mean_accuracy": None,
                    "std_accuracy": None,
                    "chance_accuracy_majority_class": None,
                    "n_samples": None,
                    "variables_used": [],
                    "fold_accuracies": [],
                    "n_splits": None,
                    "error": error_msg,
                }

                if print_results:
                    print(f"MLR failed: {error_msg}")

        new_results["results"][new_cluster_key] = cluster_output

        mlr_output = cluster_output["mlr_cca_selected_variables_results"]

        summary_rows.append({
            "cluster_number": k,
            "cluster_col": cluster_col,
            "mean_accuracy": mlr_output["mean_accuracy"],
            "std_accuracy": mlr_output["std_accuracy"],
            "chance_accuracy_majority_class": mlr_output["chance_accuracy_majority_class"],
            "n_samples": mlr_output["n_samples"],
            "n_splits": mlr_output["n_splits"],
            "cca_selected_variables": cca_selected_variables,
            "variables_used": mlr_output["variables_used"],
            "missing_variables": missing_variables,
            "non_numeric_variables": non_numeric_variables,
            "error": mlr_output["error"],
        })

    summary_df = pd.DataFrame(summary_rows)

    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(new_results), f, indent=4)

    if print_results:
        print(f"\nSaved new MLR results JSON to: {output_json_path}")

    return new_results, summary_df


def permutation_test_mlr_cv(
    data: pd.DataFrame,
    cluster_col: str,
    variable_cols: list[str],
    cv_splits: int = 5,
    random_state: int = 42,
    n_permutations: int = 1000,
    score_key: str = "mean_accuracy",
    alternative: str = "greater",
    print_results: bool = True,
):
    """
    Permutation test for multinomial logistic regression.

    Tests whether the observed prediction accuracy is higher than expected
    when cluster labels are randomly shuffled.
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
       

    observed_result = _evaluate_mlr_cv_basic(
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
            perm_result = _evaluate_mlr_cv_basic(
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
        print(f"\n=== Permutation test MLR: {cluster_col} ===")
        print(f"Variables: {variable_cols}")
        print(f"Tested score: {score_key}")
        print(f"Observed score: {observed_score:.4f}")
        print(f"Null mean: {result['null_mean']:.4f}")
        print(f"Null std: {result['null_std']:.4f}")
        print(f"p-value: {p_value:.4f}")

    return result