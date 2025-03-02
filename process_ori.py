import glob
import os
import argparse

import ants
from ants.utils.bias_correction import n4_bias_field_correction
from tqdm import tqdm
from deepbrain import Extractor
import nibabel as nib


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", type=str, help="Input directory")
    parser.add_argument("--o", type=str, help="Output directory")
    opt = parser.parse_args()
    print(opt)
    
    if not os.path.exists(opt.o):
        os.makedirs(opt.o)
    
    fixed = ants.image_read('MNI152_T1_2mm_Brain.nii.gz', reorient=True)
    ext = Extractor()
    
    for filename in tqdm(glob.glob(os.path.join(opt.i, "*.nii.gz"))):
        orig = ants.image_read(filename, reorient=True)

        orig = n4_bias_field_correction(orig)

        # brain extraction
        img = orig.numpy()
        prob = ext.run(img) 
        mask = prob < 0.5
        img[mask] = 0
        img = ants.copy_image_info(orig, ants.from_numpy(img))

        # registration
        mytx = ants.registration(fixed=fixed, moving=img, type_of_transform='SyN')
        img = mytx['warpedmovout']

        nib.save(ants.from_numpy(img), opt.o + "/" + filename.split("/")[-1])

    
    print("Processing done :)")
