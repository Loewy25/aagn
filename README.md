## Anatomy-Aware Gating Network

#### Installation
Install conda and run the following command
```
conda env create -f environment.yml
conda env create -f preprocess_environment.yml
unzip atlas.zip
```

#### Data Pre-processing
```
conda activate preprocess
python preprocess.py --i raw_data_dir --o process_data_dir
```
After pre-processing the data, create a .csv file with the following format 
```
filename,DX
process_data_dir/mri1.nii,AD
process_data_dir/mri2.nii,CN
process_data_dir/mri3.nii,AD
```
Place the train.csv, val.csv, and test.csv files into the data folder.

#### Training
```
conda activate miccai
python train.py 
```

#### Inference
```python
import torch
from aagn import AAGN

model = AAGN()
# model.load_from_checkpoint("logs/aagn/version_0/checkpoints/aagn.ckpt")
model.eval()

input = torch.randn(1, 1, 91, 109, 91)
pred, rois = model(input, return_roi=True)
```