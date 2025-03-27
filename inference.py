import nibabel as nib
import pandas as pd
import torch

from aagn import AAGN  # Import your model

csv_paths = [ 'data/test (7).csv','data/train (3).csv','data/val (1).csv']
#csv_paths = [ 'data/test.csv']
dfs = [pd.read_csv(csv_file) for csv_file in csv_paths]
combined_df = pd.concat(dfs, ignore_index=True)
print(f"Total samples in combined dataset: {len(combined_df)}")

model = AAGN.load_from_checkpoint("logs/aagn/version_17/checkpoints/aagn.ckpt")
model.eval()


results = []

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

    mri_data = nifti_img.get_fdata()

    tensor_data = torch.from_numpy(mri_data).float().unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        pred, rois = model(tensor_data, return_roi=True)

    pred_value = pred.cpu().numpy().tolist()

    row_dict = {
        'filename': file_path,
        'pred': pred_value
    }

    for k, v in rois.items():
        if isinstance(v, torch.Tensor):
            # If v is a single value (shape [] or [1]), we can do v.item()
            # Otherwise, consider summarizing or flattening:
            # e.g. v.cpu().numpy().mean(), v.cpu().numpy().sum(), etc.
            row_dict[f"roi_{k}"] = v.cpu().numpy().item()
        else:
            # If it's not a tensor (maybe a scalar or string), just store directly
            row_dict[f"roi_{k}"] = v

    results.append(row_dict)

results_df = pd.DataFrame(results)
results_df.to_csv('inference_results_ADNI_finetunning_inverse_all_v1.csv', index=False)
print("Saved inference results to inference_results.csv")


roi_columns = [col for col in results_df.columns if col.startswith("roi_")]

roi_sums = results_df[roi_columns].sum()

min_val = roi_sums.min()
max_val = roi_sums.max()
denominator = max_val - min_val

if denominator == 0:
    # If all sums are identical, set them all to 0 (or any constant)
    roi_normalized = roi_sums.apply(lambda x: 0)
else:
    roi_normalized = (roi_sums - min_val) / denominator

heatmap_row_dict = {
    'filename': 'HEATMAP_ROW',  # special label for this row
    'pred': None                # we don't normalize predictions
}

for roi_col in roi_columns:
    heatmap_row_dict[roi_col] = roi_normalized[roi_col]

heatmap_row_df = pd.DataFrame([heatmap_row_dict])

results_df = pd.concat([results_df, heatmap_row_df], ignore_index=True)

results_df.to_csv('inference_results_with_heatmap_ADNI_finetunning_inverse_all_v1.csv', index=False)
print("Saved results with heatmap row to inference_results_with_heatmap.csv")

