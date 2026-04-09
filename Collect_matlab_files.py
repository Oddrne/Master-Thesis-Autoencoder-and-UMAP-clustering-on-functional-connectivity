import os
import scipy.io as sio
import numpy as np

folder_path = "C:\Mats og Odd Arne\Prosjektoppgave\sch407\OA\zFCmat"
output_file = "OldAge_combined_run1_400_noZ.mat"

# The filenames with list names are:
# YoungAge_combined_run1_400 - run1_data, subject_names
# YoungAge_combined_run2_400 - run2_data, subject_names
# OldAge_combined_run1_400 - run1_data, subject_names
# OldAge_combined_run2_400 - run2_data, subject_names


matrices = []
names = []

for filename in sorted(os.listdir(folder_path)):
    if filename.endswith(".mat") and "run-1_" in filename:
        file_path = os.path.join(folder_path, filename)

        mat_data = sio.loadmat(file_path)
        mat_data = {k: v for k, v in mat_data.items() if not k.startswith("__")}

        # Assume one main variable per file
        key = list(mat_data.keys())[0]
        matrix = mat_data[key]

        matrix = np.tanh(matrix)  # Apply tanh to the matrix to retrieve original values in range [-1, 1]

        # Sanity check
        if matrix.shape != (454, 454):
            print(f"Skipping {filename}, unexpected shape: {matrix.shape}")
            continue

        # ✂️ Crop to 400x400
        matrix_400 = matrix[:400, :400]

        matrices.append(matrix_400)
        names.append(os.path.splitext(filename)[0])

# Stack into (N, 400, 400)
stacked = np.stack(matrices, axis=0)

# Save
sio.savemat(output_file, {
    "run1_data": stacked,
    "subject_names": names
})

print(f"Saved {len(matrices)} matrices of shape {stacked.shape} to {output_file}")