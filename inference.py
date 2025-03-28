import nibabel as nib
import pandas as pd
import torch
from torchvision.transforms import Compose
from transform import MinMaxInstance
from aagn import AAGN  # Import your model
from sklearn.metrics import classification_report, roc_auc_score

# CSV paths for your datasets
csv_paths = ['data/test (7).csv']
dfs = [pd.read_csv(csv_file) for csv_file in csv_paths]
combined_df = pd.concat(dfs, ignore_index=True)
print(f"Total samples in combined dataset: {len(combined_df)}")

# Load your trained model from checkpoint
model = AAGN.load_from_checkpoint("logs/aagn/version_21/checkpoints/aagn.ckpt")
model.eval()

# Apply exactly the transformations used during testing/validation
transforms = Compose([MinMaxInstance()])

results = []
predictions, targets, scores = [], [], []

for index, row in combined_df.iterrows():
    file_path = row['filename']  # Ensure your CSV has 'filename' column
    true_label = 1 if row['DX'] == 'AD' else 0
    targets.append(true_label)

    print(f"\nProcessing file: {file_path}")

    try:
        nifti_img = nib.load(file_path)
        mri_data = nifti_img.get_fdata()
        tensor_data = torch.from_numpy(mri_data).float().unsqueeze(0).unsqueeze(0)
        tensor_data = transforms(tensor_data)

        with torch.no_grad():
            logits, rois = model(tensor_data, return_roi=True)
            probabilities = logits.softmax(dim=1)

        pred_class = probabilities.argmax(dim=1).item()
        pred_probs = probabilities.cpu().numpy().tolist()

        predictions.append(pred_class)
        scores.append(probabilities[0, 1].item())

        row_dict = {
            'filename': file_path,
            'pred_class': pred_class,
            'pred_probabilities': pred_probs
        }

        for k, v in rois.items():
            row_dict[f"roi_{k}"] = v if isinstance(v, float) else v.cpu().numpy().item()

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        row_dict = {'filename': file_path, 'pred_class': None, 'pred_probabilities': None, 'error': str(e)}

    results.append(row_dict)

# Save inference results
results_df = pd.DataFrame(results)
results_df.to_csv('inference_results_ADNI_corrected_v1.csv', index=False)
print("Saved corrected inference results to inference_results_ADNI_corrected_v2.csv")

# Generate heatmap row
roi_columns = [col for col in results_df.columns if col.startswith("roi_")]
roi_sums = results_df[roi_columns].sum()

min_val, max_val = roi_sums.min(), roi_sums.max()
roi_normalized = ((roi_sums - min_val) / (max_val - min_val)) if max_val != min_val else roi_sums * 0

heatmap_row = {'filename': 'HEATMAP_ROW', 'pred_class': None, 'pred_probabilities': None}
heatmap_row.update({col: roi_normalized[col] for col in roi_columns})

results_df = pd.concat([results_df, pd.DataFrame([heatmap_row])], ignore_index=True)

# Save results with heatmap
results_df.to_csv('inference_results_with_heatmap_ADNI_corrected_v2.csv', index=False)
print("Saved results with heatmap to inference_results_with_heatmap_ADNI_corrected_v1.csv")

# Calculate and print evaluation metrics
report = classification_report(targets, predictions, output_dict=True)
accuracy = report["accuracy"]
sensitivity = report["1"]["recall"]
specificity = report["0"]["recall"]
auc = roc_auc_score(targets, scores)

print("\nEvaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Sensitivity: {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"AUC: {auc:.4f}")
