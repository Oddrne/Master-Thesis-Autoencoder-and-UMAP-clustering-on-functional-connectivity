import pandas as pd
import numpy as np

# Input and output files
input_file = "C:\Mats og Odd Arne\Prosjektoppgave\ISC_data\Beh.csv"
output_file = "Dummy Test Data\Dummy_Beh.csv"

# Number of synthetic rows to generate
n_samples = 72  # change as needed

# Load data
df = pd.read_csv(input_file)

# Prepare new dataframe
random_data = {}

for col in df.columns:
    # Only process numeric columns
    if pd.api.types.is_numeric_dtype(df[col]):
        col_min = df[col].min()
        col_max = df[col].max()

        # Generate random values in range
        random_values = np.random.uniform(col_min, col_max, size=n_samples)

        # If original column is integer, round
        if pd.api.types.is_integer_dtype(df[col]):
            random_values = np.round(random_values).astype(int)

        random_data[col] = random_values
    else:
        # For non-numeric columns, just copy random choices
        random_data[col] = np.random.choice(df[col].dropna(), size=n_samples)

# Create new dataframe
df_random = pd.DataFrame(random_data)

# Save to CSV
df_random.to_csv(output_file, index=False)

print(f"Random dataset saved to {output_file}")