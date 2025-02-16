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

# This list will hold results for each file
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
        # You can optionally store an error entry or skip
        results.append({
            'filename': file_path,
            'pred': None,
            'rois': None,
            'error': str(e)
        })
        continue

    # Get the data as a NumPy array (expected shape: e.g., (91, 109, 91))
    mri_data = nifti_img.get_fdata()

    # Convert to a PyTorch tensor and add dimensions:
    # - channel dimension (1)
    # - batch dimension (1)
    tensor_data = torch.from_numpy(mri_data).float()  # shape (91, 109, 91)
    tensor_data = tensor_data.unsqueeze(0).unsqueeze(0)  # shape (1, 1, 91, 109, 91)

    # Run inference without tracking gradients
    with torch.no_grad():
        pred, rois = model(tensor_data, return_roi=True)

    # Convert pred and rois to a Python-friendly format
    # (For demonstration, we flatten rois into a list. Adjust as needed!)
    pred_value = pred.cpu().numpy()  # e.g., shape (1,) if a single prediction
    rois_value = rois.cpu().numpy()  # could be 3D or 4D, etc.

    # Store the results (adjust how you represent rois if large)
    # Here we store them as list, but large arrays might be impractical in CSV.
    results.append({
        'filename': file_path,
        'pred': pred_value.tolist(),
        'rois': rois_value.flatten().tolist()  # or some summary/mean, etc.
    })

# ----- Step 4: Save results to CSV -----
# Convert results list to a DataFrame
results_df = pd.DataFrame(results)
results_df.to_csv('inference_results.csv', index=False)
print("Saved inference results to inference_results.csv")
