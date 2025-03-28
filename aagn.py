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
            return self.fc(out), {k: v for k, v in zip(ROIS, pick.squeeze().tolist())}
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
        self.scale = nn.ModuleList(
            [nn.Sequential(MLP(channels, hidden, channels), nn.Sigmoid()) for _ in range(self.n_roi)])
        self.proj = nn.ModuleList([MLP(channels, roi_hidden, roi_emb) for _ in range(self.n_roi)])

    def forward(self, data):
        emb = self.conv(data)

        roi_emb = emb.view(emb.size(0), emb.size(1), -1).matmul(self.atlas_mask.t())  # ROI-aware squeeze
        roi_emb = (roi_emb / self.atlas_mask.sum(dim=-1)).permute(0, 2, 1)  # Normalize by ROI size

        out = []
        for i in range(self.n_roi):
            feature = roi_emb[:, i, :].unsqueeze(1)
            scale = self.scale[i](feature)
            scaled_feature = scale * feature  # excite
            out.append(self.proj[i](scaled_feature))

        return torch.cat(out, dim=1)  # size: B x n_roi x hidden_dim


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
            pick = (logit / self.tau).sigmoid()  # FIXED inference behavior with temperature scaling

        if self.hard:
            pick = pick - pick.detach() + (pick > 0.5)

        return pick


ROIS = [
    'Frontal_Pole', 'Insular_Cortex', 'Superior_Frontal_Gyrus', 'Middle_Frontal_Gyrus', 
    'Inferior_Frontal_Gyrus_pars_triangularis', 'Inferior_Frontal_Gyrus_pars_opercularis',
    'Precentral_Gyrus', 'Temporal_Pole', 'Superior_Temporal_Gyrus_anterior_division',
    # ... (rest of your ROI names remain unchanged)
    'Brain-Stem', 'Hippocampus', 'Amygdala', 'Accumbens'
]
