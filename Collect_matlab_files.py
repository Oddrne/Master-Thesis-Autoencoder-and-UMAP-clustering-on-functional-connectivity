import os
import scipy.io as sio
import numpy as np

folder_path = "C:\Mats og Odd Arne\Prosjektoppgave\sch407\OA\zFCmat"
output_file = "OldAge_combined_run2_400.mat"

matrices = []
names = []

for filename in sorted(os.listdir(folder_path)):
    if filename.endswith(".mat") and "run-2_" in filename:
        file_path = os.path.join(folder_path, filename)

        mat_data = sio.loadmat(file_path)
        mat_data = {k: v for k, v in mat_data.items() if not k.startswith("__")}

        # Assume one main variable per file
        key = list(mat_data.keys())[0]
        matrix = mat_data[key]

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
    "run2_data": stacked,
    "subject_names": names
})

print(f"Saved {len(matrices)} matrices of shape {stacked.shape} to {output_file}")