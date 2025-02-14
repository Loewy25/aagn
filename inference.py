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

# ----- Step 3: Loop over each file and run inference -----
for index, row in combined_df.iterrows():
    file_path = row['filename']  # Assumes CSVs have a column named 'filename'
    print(f"\nProcessing file: {file_path}")

    # Load the MRI using nibabel (assuming a NIfTI file)
    try:
        nifti_img = nib.load(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        continue

    # Get the data as a NumPy array (expected shape: e.g., (91, 109, 91))
    mri_data = nifti_img.get_fdata()

    # Convert to a PyTorch tensor and add two dimensions:
    # 1) A channel dimension
    # 2) A batch dimension
    # Final shape becomes: (1, 1, depth, height, width)
    tensor_data = torch.from_numpy(mri_data).float()  # e.g., (91, 109, 91)
    tensor_data = tensor_data.unsqueeze(0).unsqueeze(0)  # becomes (1, 1, 91, 109, 91)

    # Run inference without tracking gradients
    with torch.no_grad():
        pred, rois = model(tensor_data, return_roi=True)

    # Print or process the output
    print("Prediction:", pred)
    print("Regions of Interest (ROIs):", rois)
