import pandas as pd
import torch
from torchvision.transforms import Compose
from transform import MinMaxInstance
from dataset import ADNIDataset
from aagn import AAGN
from sklearn.metrics import classification_report, roc_auc_score
from torch.utils.data import ConcatDataset, DataLoader

# Load model exactly as before
model = AAGN.load_from_checkpoint("logs/aagn/version_21/checkpoints/aagn.ckpt")
model.eval()

# EXACT SAME TRANSFORMS AS TRAINING/TESTING
test_transforms = Compose([MinMaxInstance()])

# Combine datasets: train, val, and test
train_dataset = ADNIDataset("data/train (3).csv", test_transforms)
val_dataset = ADNIDataset("data/val (1).csv", test_transforms)
test_dataset = ADNIDataset("data/test (7).csv", test_transforms)

combined_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])
combined_loader = DataLoader(combined_dataset, batch_size=1, shuffle=False)

results, predictions, targets, scores = [], [], [], []

for idx, (data, target) in enumerate(combined_loader):
    true_label = target.item()
    targets.append(true_label)

    with torch.no_grad():
        logits, rois = model(data, return_roi=True)
        probabilities = logits.softmax(dim=1)

    pred_class = probabilities.argmax(dim=1).item()
    pred_probs = probabilities.cpu().numpy().tolist()
    predictions.append(pred_class)
    scores.append(probabilities[0, 1].item())

    row_dict = {
        'filename': idx,
        'pred_class': pred_class,
        'pred_probabilities': pred_probs
    }

    for k, v in rois.items():
        row_dict[f"roi_{k}"] = v if isinstance(v, float) else v.cpu().numpy().item()

    results.append(row_dict)

# Save inference results
results_df = pd.DataFrame(results)
results_df.to_csv('inference_results_finetunning_OasisADNI_final_correct_ADNI_v1.csv', index=False)

# Generate heatmap row
roi_columns = [col for col in results_df.columns if col.startswith("roi_")]
roi_sums = results_df[roi_columns].sum()
min_val, max_val = roi_sums.min(), roi_sums.max()
roi_normalized = ((roi_sums - min_val) / (max_val - min_val)) if max_val != min_val else roi_sums * 0

heatmap_row = {'filename': 'HEATMAP_ROW', 'pred_class': None, 'pred_probabilities': None}
heatmap_row.update({col: roi_normalized[col] for col in roi_columns})
results_df = pd.concat([results_df, pd.DataFrame([heatmap_row])], ignore_index=True)

results_df.to_csv('inference_results_with_heatmap_finetunning_OasisADNI_final_correct_ADNI_v1.csv', index=False)

# Final evaluation metrics
report = classification_report(targets, predictions, output_dict=True)
accuracy = report["accuracy"]
sensitivity = report["1"]["recall"]
specificity = report["0"]["recall"]
auc = roc_auc_score(targets, scores)

print("\nFinal Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Sensitivity: {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"AUC: {auc:.4f}")
