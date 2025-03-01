import os
import nibabel as nib

def convert_oasis_acquisitions(
    input_base='/scratch/l.peiwang/OASIS_cross_sectional',
    output_base='/scratch/l.peiwang/OASIS_final'):
    """
    Recursively walk through 'input_base', find .img files matching 'mpr-n_anon' 
    (e.g., mpr-1_anon, mpr-2_anon, etc.), and convert them to .nii.gz in 'output_base',
    preserving folder structure.
    """
    # Walk through the directory tree
    for root, dirs, files in os.walk(input_base):
        for file in files:
            # Look for .img files that match the pattern "mpr-<digit>_anon"
            # e.g. "OAS1_0042_MR1_mpr-1_anon.img", "OAS1_0042_MR1_mpr-2_anon.img", etc.
            if file.endswith('.img') and 'mpr-' in file and '_anon' in file:
                # Full path to the Analyze (.img) file
                input_path = os.path.join(root, file)
                
                # Compute relative path so we can replicate the folder structure
                rel_path = os.path.relpath(root, input_base)
                
                # Create corresponding output directory
                out_dir = os.path.join(output_base, rel_path)
                os.makedirs(out_dir, exist_ok=True)
                
                # Build the output filename: replace .img with .nii.gz
                base_name = os.path.splitext(file)[0]
                output_file = base_name + '.nii.gz'
                output_path = os.path.join(out_dir, output_file)
                
                # Convert using nibabel
                try:
                    # Load the Analyze image (nib will automatically pair with .hdr)
                    nii_img = nib.load(input_path)
                    # Save as compressed NIfTI
                    nib.save(nii_img, output_path)
                    print(f"Converted: {input_path} -> {output_path}")
                except Exception as e:
                    print(f"Error converting {input_path}: {e}")

if __name__ == "__main__":
    # Adjust the input and output paths as needed
    convert_oasis_acquisitions(
        input_base='/scratch/l.peiwang/OASIS_cross_sectional',
        output_base='/scratch/l.peiwang/OASIS_final'
    )
