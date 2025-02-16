import nibabel as nib
import pandas as pd
import torch

from aagn import AAGN  # Import your model

# ----- Step 1: Combine CSVs into one DataFrame -----
csv_paths = ['data/train.csv', 'data/val.csv', 'data/test.csv']
dfs = [pd.read_csv(csv_file) for csv_file in csv_paths]
combined_df = pd.concat(dfs, ignore_index=True)
print(f"Total samples in combined dataset: {len(combined_df)}")

# ----- Step 2: Initialize your model for inference -----
model = AAGN()
model.load_from_checkpoint("logs/aagn/version_9/checkpoints/aagn.ckpt")
model.eval()  # Set the model to evaluation mode

# This list will hold the results, which will eventually become a DataFrame
results = []

# ----- Step 3: Loop over each file and run inference -----
for index, row in combined_df.iterrows():
    file_path = row['filename']  # Assumes CSVs have a column named 'filename'
    print(f"\nProcessing file: {file_path}")

    # Load the MRI using nibabel (assuming a NIfTI file)
    try:
        nifti_img = nib.load(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        # Record the error in the results, skip further processing
        results.append({
            'filename': file_path,
            'pred': None,
            'error': str(e)
        })
        continue

    # Get the data as a NumPy array (expected shape, e.g., (91, 109, 91))
    mri_data = nifti_img.get_fdata()

    # Convert to a PyTorch tensor and add two dimensions (batch & channel)
    # Shape goes from (D, H, W) -> (1, 1, D, H, W)
    tensor_data = torch.from_numpy(mri_data).float().unsqueeze(0).unsqueeze(0)

    # Run inference (no gradient tracking)
    with torch.no_grad():
        pred, rois = model(tensor_data, return_roi=True)

    # Convert prediction to a Python-friendly format
    # e.g., if pred is shape (1,) or scalar, .item() might also work
    pred_value = pred.cpu().numpy().tolist()

    # We'll create a dictionary (row_dict) that represents one row in the CSV
    row_dict = {
        'filename': file_path,
        'pred': pred_value
    }

    # 'rois' is a dictionary; let's store each key as its own CSV column
    for k, v in rois.items():
        if isinstance(v, torch.Tensor):
            # If v is a single value (shape [] or [1]), we can do v.item()
            # Otherwise, consider summarizing or flattening:
            # e.g. v.cpu().numpy().mean(), v.cpu().numpy().sum(), etc.
            row_dict[f"roi_{k}"] = v.cpu().numpy().item()
        else:
            # If it's not a tensor (maybe a scalar or string), just store directly
            row_dict[f"roi_{k}"] = v

    # Add this row to our results list
    results.append(row_dict)

# ----- Step 4: Convert results to a DataFrame and save to CSV -----
results_df = pd.DataFrame(results)
results_df.to_csv('inference_results.csv', index=False)
print("Saved inference results to inference_results.csv")

