import os
import re
import glob
import json
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm

from latlong import BackboneFactory, LatLongDataset


class GeoRegressionModel(nn.Module):
    """Model for regressing latitude and longitude from images."""

    def __init__(self,
                 backbone_name: str,
                 num_regions: int,
                 embedding_dim: int = 16,
                 dropout_rate: float = 0.3,
                 pretrained: bool = True,
                 head_hidden_dim: int = 128):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone, self.feature_dim = BackboneFactory.get_backbone(backbone_name, pretrained)
        self.region_embedding = nn.Embedding(num_regions, embedding_dim)

        combined_dim = self.feature_dim + embedding_dim + 2  # +2 for sin and cos of angle
        self.head_lat = nn.Sequential(
            nn.Linear(combined_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(head_hidden_dim, head_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(head_hidden_dim // 2, 1)
        )
        self.head_lon = nn.Sequential(
            nn.Linear(combined_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(head_hidden_dim, head_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(head_hidden_dim // 2, 1)
        )

    def forward(self, image, region_idx, angle_sin, angle_cos):
        x = self.backbone(image)
        if x.ndim == 4:
            x = torch.flatten(x, 1)
        region_embed = self.region_embedding(region_idx)
        angle_feat = torch.stack([angle_sin, angle_cos], dim=1)
        feats = torch.cat([x, region_embed, angle_feat], dim=1)
        lat = self.head_lat(feats).squeeze(1)
        lon = self.head_lon(feats).squeeze(1)
        return lat, lon


class EnsembleModel:
    """Loads multiple GeoRegressionModels and makes weighted predictions."""

    def __init__(self, models_dir: str, num_regions: int,
                 temperature: float = 1.0,
                 device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.temperature = temperature
        self.models = []
        self.model_infos = []
        self._load_models(models_dir, num_regions)
        if not self.models:
            raise RuntimeError(f"No valid models found in {models_dir}")
        self._compute_weights()

    def _load_models(self, models_dir: str, num_regions: int):
        for backbone_name in os.listdir(models_dir):
            type_dir = os.path.join(models_dir, backbone_name)
            if not os.path.isdir(type_dir):
                continue
            for pth_file in glob.glob(os.path.join(type_dir, "*.pth")):
                fname = os.path.basename(pth_file)
                m = re.match(r"(.+)_emb(\d+)_dr([\d\.]+)\.pth", fname)
                if not m:
                    print(f"Skipping {fname}, bad format")
                    continue
                emb_dim = int(m.group(2))
                dr = float(m.group(3))
                summary_txt = pth_file.replace('.pth', '_summary.txt')
                val_mse = float('inf')
                if os.path.exists(summary_txt):
                    with open(summary_txt) as sf:
                        for line in sf:
                            if 'Best Validation MSE' in line:
                                val_mse = float(line.split(':')[1].strip())
                                break
                chk = torch.load(pth_file, map_location=self.device)
                model = GeoRegressionModel(
                    backbone_name=backbone_name,
                    num_regions=num_regions,
                    embedding_dim=emb_dim,
                    dropout_rate=dr
                )
                model.load_state_dict(chk['model_state_dict'], strict=False)
                model.to(self.device).eval()
                self.models.append(model)
                self.model_infos.append({
                    'backbone': backbone_name,
                    'emb': emb_dim,
                    'dr': dr,
                    'val_mse': val_mse,
                    'weight': 0.0
                })

    def _compute_weights(self):
        mses = np.array([info['val_mse'] for info in self.model_infos], dtype=np.float64)
        scores = -mses / max(self.temperature, 1e-8)
        exp_s = np.exp(scores - scores.max())
        weights = exp_s / exp_s.sum()
        for w, info in zip(weights, self.model_infos):
            info['weight'] = float(w)
            print(f"Model {info['backbone']}_emb{info['emb']}_dr{info['dr']}: weight={w:.4f}")
        self.weights = torch.tensor(weights, device=self.device)

    def predict(self, images, region_idxs, angle_sins, angle_coss):
        lat_preds = []
        lon_preds = []
        for model in self.models:
            with torch.no_grad():
                lat, lon = model(images, region_idxs, angle_sins, angle_coss)
                lat_preds.append(lat.unsqueeze(0))
                lon_preds.append(lon.unsqueeze(0))
        lat_stack = torch.cat(lat_preds, dim=0)
        lon_stack = torch.cat(lon_preds, dim=0)
        w = self.weights.view(-1, 1)
        return (lat_stack * w).sum(dim=0), (lon_stack * w).sum(dim=0)


def predict(args):
    # Device
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # Data
    test_ds = LatLongDataset(
        csv_file=args.test_csv,
        img_dir=args.test_img_dir,
        transform=None
    )
    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    print(f"Loaded {len(test_ds)} samples, {test_ds.num_regions} regions.")
    # Ensemble
    ensemble = EnsembleModel(
        models_dir=args.models_dir,
        num_regions=test_ds.num_regions,
        temperature=args.temperature,
        device=device
    )
    # Scalers
    lat_scaler = joblib.load(args.lat_scalar_path)
    lon_scaler = joblib.load(args.long_scalar_path)

    records = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            imgs = batch['image'].to(device)
            sins = batch['angle_sin'].to(device)
            coss = batch['angle_cos'].to(device)
            regs = batch['region_idx'].to(device)
            fnames = batch['filename']
            lats, lons = ensemble.predict(imgs, regs, sins, coss)
            lats_np = lats.cpu().numpy().reshape(-1,1)
            lons_np = lons.cpu().numpy().reshape(-1,1)
            u_lats = lat_scaler.inverse_transform(lats_np).flatten()
            u_lons = lon_scaler.inverse_transform(lons_np).flatten()
            for fn, plat, plon in zip(fnames, u_lats, u_lons):
                records.append({'filename': fn, 'pred_latitude': float(plat), 'pred_longitude': float(plon)})

    df = pd.DataFrame(records)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to {args.output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Ensemble GeoRegression prediction script.")
    parser.add_argument('--test_csv', required=True)
    parser.add_argument('--test_img_dir', required=True)
    parser.add_argument('--models_dir', default="outputs")
    parser.add_argument('--lat_scalar_path', default="lat_scaler.pkl")
    parser.add_argument('--long_scalar_path', default="long_scaler.pkl")
    parser.add_argument('--output_csv', default="LatLongPred.csv")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()
    predict(args)


if __name__ == '__main__':
    main()
