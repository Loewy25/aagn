import os
import re
import glob
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

# Path to the original OASIS Excel file
excel_path = "/home/l.peiwang/aagn/Reference.xlsx"

# Read the Excel file into a DataFrame
df = pd.read_excel(excel_path)

# Directory where all OASIS NIfTI files are located
base_dir = "/scratch/l.peiwang/OASIS_1"

def get_label(cdr_value):
    """Return 'CN' if CDR=0, otherwise 'AD'."""
    return "CN" if cdr_value == 0 else "AD"

all_data = []

for _, row in df.iterrows():
    subject_id = row["ID"]  # e.g., "OAS1_0013_MR1"
    cdr_value = row["CDR"]
    label = get_label(cdr_value)

    # Extract the 4-digit number (e.g. "0013") from something like "OAS1_0013_MR1"
    match = re.search(r"OAS1_(\d{4})_", subject_id)
    if not match:
        # If we can't parse the ID, skip this row
        continue
    digits = match.group(1)  # e.g. "0013"

    # For each possible mpr index, search for all matching files
    for mpr_idx in [1, 2, 3]:
        pattern = f"OAS1_{digits}_MR*_mpr-{mpr_idx}_anon.nii.gz"
        matches = glob.glob(os.path.join(base_dir, pattern))
        for filename in matches:
            all_data.append((filename, label))

# Create a DataFrame with columns ["filename", "DX"]
df_all = pd.DataFrame(all_data, columns=["filename", "DX"])

# -------------------------------------------------------------------------
# Stratified splitting into train/val/test (70% / 15% / 15%)
# We'll do two splits using scikit-learn's train_test_split with 'stratify'
# -------------------------------------------------------------------------

# 1) First, split off the test set (15%).
#    If total is 100%, we want 15% test, so test_size=0.15.
df_train_val, df_test = train_test_split(
    df_all,
    test_size=0.15,
    stratify=df_all["DX"],
    random_state=42
)

# 2) Now, we have 85% data left in df_train_val, and we want 15% of the original total
#    to be our val set. 15% out of 85% is 0.15/0.85 ~ 0.1765
df_train, df_val = train_test_split(
    df_train_val,
    test_size=0.1764705882,  # ~ 15% of total
    stratify=df_train_val["DX"],
    random_state=42
)

# Check the sizes
n_train = len(df_train)
n_val = len(df_val)
n_test = len(df_test)
n_total = len(df_all)
print(f"Total samples: {n_total}")
print(f"Train samples: {n_train}  ({n_train / n_total:.2%})")
print(f"Val samples:   {n_val}    ({n_val / n_total:.2%})")
print(f"Test samples:  {n_test}   ({n_test / n_total:.2%})")

# -------------------------------------------------------------------------
# Save to CSV
# -------------------------------------------------------------------------
out_dir = "/home/l.peiwang/aagn"
df_train.to_csv(os.path.join(out_dir, "train.csv"), index=False)
df_val.to_csv(os.path.join(out_dir, "val.csv"), index=False)
df_test.to_csv(os.path.join(out_dir, "test.csv"), index=False)

print("CSV files created (with stratified splitting):")
print(f"  - {os.path.join(out_dir, 'train.csv')}")
print(f"  - {os.path.join(out_dir, 'val.csv')}")
print(f"  - {os.path.join(out_dir, 'test.csv')}")


