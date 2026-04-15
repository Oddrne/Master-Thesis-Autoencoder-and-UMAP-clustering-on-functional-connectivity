import re
from pathlib import Path
from scipy.io import savemat
import h5py
import numpy as np


def extract_info(filename):
    """
    Extract:
    - input_name (everything before '_run')
    - run_id (between 'run_' and 'ROIS')
    """
    filename = Path(filename).stem
    
    input_name = filename.split("_run")[0]
    subject_id = re.search(r"sub-(.*?)_task", filename)
    run_match = re.search(r"run-(.*?)__ROIS", filename)
    if run_match is None:
        raise ValueError(f"Could not extract run_id from {filename}")

    run_id = run_match.group(1)

    return input_name, run_id, subject_id.group(1)


def load_roisignal(mat_path):
    with h5py.File(mat_path, "r") as f:
        ts = np.array(f["ROISignal"])
        ts = np.squeeze(ts).T

    if ts.shape != (1000, 240):
        raise ValueError(f"Unexpected shape {ts.shape} in {mat_path}")

    return ts


def compute_fc(ts):
    fc = np.corrcoef(ts)
    fc = np.nan_to_num(fc)
    np.fill_diagonal(fc, 1.0)
    return fc


def compute_fc_directory(input_dir):
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob("*.mat"))

    results = []

    for file in files:
        input_name, run_id, subject_id = extract_info(file.name)

        ts = load_roisignal(file)
        fc = compute_fc(ts)

        results.append({
            "subject_id": subject_id,
            "fc_matrix": fc
        })

        #print(f"{file.name} -> run={run_id} | FC={fc.shape}")

    return results, run_id

def save_fc_results(results, output_file="fc_results.mat", run_id=1):
    """
    Save each FC matrix under a key:
    '{input_name}_run_{run_id}'
    """

    filtered = [r for r in results]

    subject_names = np.stack([r["subject_id"] for r in filtered], axis=0)
    fc_stack = np.stack([r["fc_matrix"] for r in filtered], axis=0)

    run_name = f"run{run_id}_data"

    save_dict = {
        "subject_names": np.array(subject_names, dtype=object),
        run_name: fc_stack
    }

    savemat(output_file, save_dict)

    print(f"\nSaved {len(filtered)} FC matrices to {output_file}")