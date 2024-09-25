import ants
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class ADNIDataset(Dataset):

    def __init__(self, data, transform=None):
        X, y = [], []
        df = pd.read_csv(data)
        dictionary = {"CN": 0, "AD": 1}
        # dictionary = {"sMCI": 0, "pMCI": 1}

        for _, datum in tqdm(df.iterrows(), total=df.shape[0]):    
            X.append(ants.image_read(datum["filename"], reorient=True).numpy())
            y.append(dictionary[datum["DX"]])
                    
        self.X = torch.Tensor(np.array(X))
        self.y = torch.Tensor(y).long()
        
        self.transform = transform

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        sample = self.X[idx].unsqueeze(0)
        if self.transform:        
            sample = self.transform(sample)
        return sample, self.y[idx]
