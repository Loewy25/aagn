import os
import pandas as pd
from sklearn.model_selection import train_test_split
import glob

def generate(images, labels, task):
    imagesData = []
    labelsData = []
    cn = 0
    pcn = 0
    dementia = 0
    mci = 0
    for i in range(len(images)):
        if labels[i] == 'CN':
            cn += 1
        if labels[i] == 'MCI':
            mci += 1
        if labels[i] == 'Dementia':
            dementia += 1
        if labels[i] == 'PCN':
            pcn += 1
    print("Number of CN subjects:")
    print(cn)
    print("Number of PCN subjects:")
    print(pcn)
    print("Number of MCI subjects:")
    print(mci)
    print("Number of Dementia subjects:")
    print(dementia)
    if task == "cd":
        for i in range(len(images)):
            if labels[i] == "CN":
                imagesData.append(images[i])
                labelsData.append(labels[i])
            if labels[i] == "Dementia":
                imagesData.append(images[i])
                labelsData.append(labels[i])
    if task == "cm":
        for i in range(len(images)):
            if labels[i] == "CN":
                imagesData.append(images[i])
                labelsData.append(labels[i])
            if labels[i] == "MCI":
                imagesData.append(images[i])
                labelsData.append(labels[i])
    if task == "dm":
        for i in range(len(images)):
            if labels[i] == "Dementia":
                imagesData.append(images[i])
                labelsData.append(labels[i])
            if labels[i] == "MCI":
                imagesData.append(images[i])
                labelsData.append(labels[i])
    if task == "pc":
        for i in range(len(images)):
            if labels[i] == "PCN":
                imagesData.append(images[i])
                labelsData.append(labels[i])
            if labels[i] == "CN":
                imagesData.append(images[i])
                labelsData.append(labels[i])
    if task == 'cdm':
        for i in range(len(images)):
            if labels[i] == "CN":
                imagesData.append(images[i])
                labelsData.append(labels[i])
            if labels[i] == "Dementia" or labels[i] == 'MCI':
                imagesData.append(images[i])
                labelsData.append(labels[i])
    print("lenth of dataset: ")
    print(len(labelsData))

    return imagesData, labelsData


def generate_data_path_less():
    files=['/home/l.peiwang/table_preclinical_cross-sectional.csv','/home/l.peiwang/table_cn_cross-sectional.csv','/home/l.peiwang/table_cdr_0p5_apos_cross-sectional.csv','/home/l.peiwang/table_cdr_gt_0p5_apos_cross-sectional.csv']
    class_labels=['PCN','CN','MCI','Dementia']
    pet_paths = []
    mri_paths = []
    class_labels_out = []

    for file, class_label in zip(files, class_labels):
        df = pd.read_csv(file)

        for _, row in df.iterrows():
            # Extract sub-xxxx and ses-xxxx from original paths
            sub_ses_info = "/".join(row['FdgFilename'].split("/")[8:10])

            # Generate new directory
            new_directory = os.path.join('/ceph/chpc/shared/aristeidis_sotiras_group/l.peiwang_scratch/derivatives_less', sub_ses_info, 'anat')

            # Get all files that match the pattern but then exclude ones that contain 'icv'
            pet_files = [f for f in glob.glob(new_directory + '/*FDG*') if 'icv' not in f]
            mri_files = glob.glob(new_directory + '/*brain*')
            if pet_files and mri_files:  # If both lists are not empty
                pet_paths.append(pet_files[0])  # Append the first PET file found
                mri_paths.append(mri_files[0])  # Append the first MRI file found
                class_labels_out.append(class_label)  # Associate class label with the path

    return pet_paths, mri_paths, class_labels_out

def split_and_save(pet_data, labels):
    """
    Takes two lists: pet_data (file paths) and labels (CN or Dementia),
    splits into train, val, test, then saves them in 'data/' folder as CSVs.
    """
    # 1) Create a DataFrame
    df = pd.DataFrame({
        'filename': pet_data,
        'DX': labels
    })

    # 2) Replace 'Dementia' with 'AD'
    df['DX'] = df['DX'].replace({'Dementia': 'AD'})

    # Safety check: if some are neither 'CN' nor 'AD', you can filter them out
    # (e.g., if there's MCI leftover by mistake)
    df = df[df['DX'].isin(['CN', 'AD'])]

    # If df is empty after filtering, handle gracefully
    if len(df) == 0:
        raise ValueError("No valid CN/AD data found in the provided lists.")

    # 3) Train/Val/Test split
    #    - 70% train, then 15% val, 15% test
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df['DX']
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df['DX']
    )

    # 4) Save the splits to CSV files
    os.makedirs('data', exist_ok=True)

    train_df.to_csv('data/train.csv', index=False)
    val_df.to_csv('data/val.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)

    print("Finished! Splits saved in 'data/train.csv', 'data/val.csv', and 'data/test.csv'.")


# Usage example (assuming you already have the variables pet_data, labels):
if __name__ == "__main__":
    # 1. Suppose you've run:
    images_pet, images_mri, labels = generate_data_path_less()
    images_mri, label = generate(images_mri, labels, 'cd')
    #    and these contain only CN or Dementia in 'label'
    #
    # 2. Then call:
    split_and_save(images_mri, label)
