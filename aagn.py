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
