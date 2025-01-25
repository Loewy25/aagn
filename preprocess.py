import os
import argparse
import pandas as pd

import ants
from ants.utils.bias_correction import n4_bias_field_correction
from tqdm import tqdm
from deepbrain import Extractor
import nibabel as nib

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, default="data/train.csv")
    parser.add_argument("--val_csv", type=str, default="data/val.csv")
    parser.add_argument("--test_csv", type=str, default="data/test.csv")
    parser.add_argument("--o", type=str, default="/scratch/l.peiwang/data",
                        help="Output directory for processed NIfTI files.")
    args = parser.parse_args()

    print(args)

    # Create output directory if needed
    if not os.path.exists(args.o):
        os.makedirs(args.o)

    # Reference brain for registration
    fixed = ants.image_read('MNI152_T1_2mm_Brain.nii.gz', reorient=True)
    ext = Extractor()


    # Helper function to process a single subject
    def process_file(in_path, out_path):
        # 1) Bias field correction
        orig = ants.image_read(in_path, reorient=True)
        orig = n4_bias_field_correction(orig)

        # 2) Brain extraction via deepbrain
        img = orig.numpy()
        prob = ext.run(img)
        mask = prob < 0.5
        img[mask] = 0
        img = ants.copy_image_info(orig, ants.from_numpy(img))

        # 3) Registration
        mytx = ants.registration(fixed=fixed, moving=img, type_of_transform='SyN')
        warped = mytx['warpedmovout']

        # 4) Save with nibabel
        nib.save(ants.from_numpy(warped.numpy()), out_path)


    # Process each CSV
    for csv_file in [args.train_csv, args.val_csv, args.test_csv]:
        if not os.path.exists(csv_file):
            print(f"Warning: {csv_file} does not exist. Skipping.")
            continue

        df = pd.read_csv(csv_file)
        if "filename" not in df.columns:
            print(f"Warning: 'filename' column not found in {csv_file}. Skipping.")
            continue

        new_paths = []

        for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {csv_file}"):
            in_path = row['filename']
            base_name = os.path.basename(in_path)  # e.g. "subject_001.nii"
            out_path = os.path.join(args.o, base_name)

            # Run the preprocessing
            process_file(in_path, out_path)

            # Update the row to the new location
            new_paths.append(out_path)

        # Overwrite 'filename' column with the new processed file paths
        df['filename'] = new_paths

        # Save an updated CSV in the same directory or with a suffix:
        processed_csv = csv_file.replace(".csv", "_processed.csv")
        df.to_csv(processed_csv, index=False)
        print(f"Updated CSV saved to {processed_csv}")

    print("All processing complete :)")
