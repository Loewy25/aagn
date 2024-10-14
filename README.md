## Anatomy-Aware Gating Network
[![paper](https://img.shields.io/badge/PDF-Available-red?logo=adobeacrobatreader&logoColor=white)](https://papers.miccai.org/miccai-2024/paper/2553_paper.pdf)

This is the official implementation of our MICCAI 2024 paper "Anatomy-Aware Gating Network for Explainable Alzheimer's Disease Diagnosis". In this paper, we propose an Anatomy-Aware Gating Network (AAGN) for identifying neurodegenerative diseases (e.g., Alzheimer's) from 3D MRI scans. The model extracts features from various anatomical regions using an anatomy-aware squeeze-and-excite operation. 
By conditioning on the anatomy-aware features, AAGN dynamically selects task-relevant regions, enabling interpretable classification.

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

#### Citation
```
@inproceedings{jiang2024anatomy,
  title={Anatomy-Aware Gating Network for Explainable Alzheimer’s Disease Diagnosis},
  author={Jiang, Hongchao and Miao, Chunyan},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={90--100},
  year={2024},
  organization={Springer}
}
```
