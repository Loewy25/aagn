import argparse
from argparse import Namespace

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.optim import AdamW

from base import Base


class AAGN(Base):
    def __init__(self, hparams="aagn.yaml"):
        super(AAGN, self).__init__()        
        with open(hparams, "r") as file:
            self.save_hyperparameters(yaml.safe_load(file))
            print(self.hparams)
        
        self.anatomy = AnatomyNet(**self.hparams)
        self.gate = GateNet(**self.hparams)
        self.fc = torch.nn.Linear(self.hparams.roi_emb, 2)

    def forward(self, x, return_roi=False):
        roi_emb = self.anatomy(x)
        pick = self.gate(roi_emb)
        out = (roi_emb * pick.unsqueeze(-1)).sum(dim=1)        
        
        if return_roi:
            return self.fc(out), {k:v for k, v in zip(ROIS, pick.squeeze().tolist())}
        else:
            return self.fc(out)

    def training_step(self, batch, batch_nb):
        data, target = batch
        output = self.forward(data)
        loss = F.cross_entropy(output, target) 
        self.log('loss', loss)
        
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.hparams.lr) 


class MLP(nn.Module):
    def __init__(self, input_dim, hidden, output_dim):
        nn.Module.__init__(self) 
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), 
            nn.ReLU(inplace=True), 
            nn.Linear(hidden, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)
    

class AnatomyNet(nn.Module):
    def __init__(self, channels, hidden, n_layers, roi_hidden, roi_emb, atlas="atlas.pt", **kwargs):
        nn.Module.__init__(self) 
        
        self.register_buffer('atlas_mask', torch.load(atlas))        
        self.conv = nn.Sequential(
            nn.Conv3d(1, channels, 3, padding=1),
            nn.InstanceNorm3d(channels),
            nn.ReLU(inplace=True),
        )
        
        for i in range(1, n_layers):
            self.conv.add_module(f"conv_{i}", 
                                nn.Sequential(            
                                    nn.Conv3d(channels, channels, 3, padding=1),
                                    nn.InstanceNorm3d(channels),
                                    nn.ReLU(inplace=True),
                                ))
        
        self.n_roi = self.atlas_mask.size(0)
        self.scale = nn.ModuleList([nn.Sequential(MLP(channels, hidden, channels), nn.Sigmoid()) for _ in range(self.n_roi)])  
        self.proj = nn.ModuleList([MLP(channels, roi_hidden, roi_emb) for _ in range(self.n_roi)])

    def forward(self, data):
        """
        data shape: (B, 1, D, H, W)
        After self.conv, we get emb of shape (B, C, D, H, W).

        We want to multiply the flattened spatial dimension (D*H*W)
        by atlas_mask (which is (n_roi, voxel_count)), so we do:
          (B*C, voxel_count) x (voxel_count, n_roi) = (B*C, n_roi)
        then reshape back to (B, C, n_roi).
        """
        emb = self.conv(data)
        B, C, D, H, W = emb.shape
        S = D * H * W  # number of voxels per volume

        # 1) Flatten only the spatial dimensions => (B, C, S)
        emb = emb.view(B, C, S)  # do NOT merge C with S

        # 2) Flatten (B, C) into a single dimension => shape (B*C, S)
        emb_2d = emb.view(B * C, S)

        # 3) Your atlas_mask is (n_roi, S) => atlas_mask.t() = (S, n_roi)
        #    So we can do a standard 2D matmul: (B*C, S) x (S, n_roi) => (B*C, n_roi)
        atlas_t = self.atlas_mask.t()  # shape (S, n_roi)
        out_2d = emb_2d.matmul(atlas_t)  # (B*C, n_roi)

        # 4) Reshape back to (B, C, n_roi)
        roi_emb = out_2d.view(B, C, self.n_roi)

        # 5) Normalize each ROI by the number of voxels in that ROI
        #    self.atlas_mask.sum(dim=-1) => shape (n_roi,)
        #    Broadcasting => (B, C, n_roi) / (n_roi,)
        roi_emb = roi_emb / self.atlas_mask.sum(dim=-1)

        # 6) Permute => (B, n_roi, C) so that the ROI dimension is in the middle
        roi_emb = roi_emb.permute(0, 2, 1)  # (B, n_roi, C)

        # 7) For each ROI, apply scale & proj
        out = []
        for i in range(self.n_roi):
            # roi_emb[:, i, :] => shape (B, C)
            feature = roi_emb[:, i, :].unsqueeze(1)  # => (B, 1, C)
            scale_factor = self.scale[i](feature)  # => (B, 1, C)
            scaled_feature = scale_factor * feature  # (B, 1, C)
            out.append(self.proj[i](scaled_feature))  # => (B, 1, roi_emb_dim)

        # 8) Concatenate => shape (B, n_roi, roi_emb_dim)
        return torch.cat(out, dim=1)


class GateNet(nn.Module):
    def __init__(self, tau, hidden, hard, n_roi=57, **kwargs):
        super(GateNet, self).__init__()
        self.gate = MLP(n_roi, hidden, n_roi)         
        self.hard = hard
        self.tau = tau
        
    def forward(self, roi_emb):
        features = roi_emb.sum(dim=-1)
        logit = self.gate(features) 

        if self.training:
            gumbels = (
                -torch.empty_like(logit, memory_format=torch.legacy_contiguous_format).exponential_().log()
            )
            gumbels2 = (
                -torch.empty_like(logit, memory_format=torch.legacy_contiguous_format).exponential_().log()
            )
            gumbels = gumbels - gumbels2

            pick = ((logit + gumbels) / self.tau).sigmoid()

        else:
            pick = logit.sigmoid() # w/o gumbel noise
            
        if self.hard:
            pick = pick - pick.detach() + (pick > 0.5)
        
        return pick


ROIS = [
    'Frontal_Pole',
    'Insular_Cortex',
    'Superior_Frontal_Gyrus',
    'Middle_Frontal_Gyrus',
    'Inferior_Frontal_Gyrus_pars_triangularis',
    'Inferior_Frontal_Gyrus_pars_opercularis',
    'Precentral_Gyrus',
    'Temporal_Pole',
    'Superior_Temporal_Gyrus_anterior_division',
    'Superior_Temporal_Gyrus_posterior_division',
    'Middle_Temporal_Gyrus_anterior_division',
    'Middle_Temporal_Gyrus_posterior_division',
    'Middle_Temporal_Gyrus_temporooccipital_part',
    'Inferior_Temporal_Gyrus_anterior_division',
    'Inferior_Temporal_Gyrus_posterior_division',
    'Inferior_Temporal_Gyrus_temporooccipital_part',
    'Postcentral_Gyrus',
    'Superior_Parietal_Lobule',
    'Supramarginal_Gyrus_anterior_division',
    'Supramarginal_Gyrus_posterior_division',
    'Angular_Gyrus',
    'Lateral_Occipital_Cortex_superior_division',
    'Lateral_Occipital_Cortex_inferior_division',
    'Intracalcarine_Cortex',
    'Frontal_Medial_Cortex',
    'Juxtapositional_Lobule_Cortex_(formerly_Supplementary_Motor_Cortex)',
    'Subcallosal_Cortex',
    'Paracingulate_Gyrus',
    'Cingulate_Gyrus_anterior_division',
    'Cingulate_Gyrus_posterior_division',
    'Precuneous_Cortex',
    'Cuneal_Cortex',
    'Frontal_Orbital_Cortex',
    'Parahippocampal_Gyrus_anterior_division',
    'Parahippocampal_Gyrus_posterior_division',
    'Lingual_Gyrus',
    'Temporal_Fusiform_Cortex_anterior_division',
    'Temporal_Fusiform_Cortex_posterior_division',
    'Temporal_Occipital_Fusiform_Cortex',
    'Occipital_Fusiform_Gyrus',
    'Frontal_Operculum_Cortex',
    'Central_Opercular_Cortex',
    'Parietal_Operculum_Cortex',
    'Planum_Polare',
    "Heschl's_Gyrus_(includes_H1_and_H2)",
    'Planum_Temporale',
    'Supracalcarine_Cortex',
    'Occipital_Pole',
    'Lateral_Ventricle',
    'Thalamus',
    'Caudate',
    'Putamen',
    'Pallidum',
    'Brain-Stem',
    'Hippocampus',
    'Amygdala',
    'Accumbens'
]