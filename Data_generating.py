import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

def generate_data_splits():
    # Create 'data' folder if it doesn't exist
    os.makedirs('data', exist_ok=True)

    files = [
        '/scratch/jjlee/Singularity/ADNI/bids/derivatives/table_preclinical_cross-sectional.csv',
        '/scratch/jjlee/Singularity/ADNI/bids/derivatives/table_cn_cross-sectional.csv',
        '/scratch/jjlee/Singularity/ADNI/bids/derivatives/table_cdr_0p5_apos_cross-sectional.csv',
        '/scratch/jjlee/Singularity/ADNI/bids/derivatives/table_cdr_gt_0p5_apos_cross-sectional.csv'
    ]
    class_labels = ['PCN','CN','MCI','Dementia']  # Must match the order of 'files'

    # We will store (mri_path, dx_label) pairs
    data_entries = []

    for file_path, class_label in zip(files, class_labels):
        df = pd.read_csv(file_path)

        # Keep only "CN" -> DX="CN" and "Dementia" -> DX="AD"
        if class_label == 'CN':
            dx = 'CN'
        elif class_label == 'Dementia':
            dx = 'AD'
        else:
            # Skip other labels (MCI, PCN)
            continue

        # Iterate over each row in the CSV
        for _, row in df.iterrows():
            sub_ses_info = "/".join(row['FdgFilename'].split("/")[8:10])  # sub-XXXX/ses-YYYY
            new_directory = os.path.join(
                '/scratch/l.peiwang/derivatives_less',
                sub_ses_info,
                'anat'
            )

            # Look for MRI files that match '*brain*' (avoid PET)
            mri_files = glob.glob(os.path.join(new_directory, '*brain*'))

            if mri_files:
                # Use the first match if multiple exist
                data_entries.append((mri_files[0], dx))

    # Convert to DataFrame
    all_df = pd.DataFrame(data_entries, columns=['filename', 'DX'])

    # Train/Val/Test split
    train_df, temp_df = train_test_split(
        all_df,
        test_size=0.30,
        random_state=42,
        stratify=all_df['DX']
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df['DX']
    )
    os.makedirs('data', exist_ok=True)
    # Save the splits to CSV files in 'data' folder
    train_df.to_csv('data/train.csv', index=False)
    val_df.to_csv('data/val.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)

    print("Splits generated and saved as data/train.csv, data/val.csv, data/test.csv.")

if __name__ == "__main__":
    generate_data_splits()
