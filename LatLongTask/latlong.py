import os
import math
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.distributions import Beta
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageFile

# Ensure PIL can load truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# For reproducibility
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)

# --- backbone factory ---
class BackboneFactory:
    """Factory to instantiate backbones, strip their heads, and return feature dims."""

    # Map model names to (constructor, attribute holding final head)
    model_map = {
        # ResNet family
        'resnet18':           (models.resnet18,           'fc'),
        'resnet34':           (models.resnet34,           'fc'),
        'resnet50':           (models.resnet50,           'fc'),
        'resnet101':          (models.resnet101,          'fc'),

        # EfficientNet family
        'efficientnet_b0':    (models.efficientnet_b0,    'classifier'),
        'efficientnet_b1':    (models.efficientnet_b1,    'classifier'),
        'efficientnet_b2':    (models.efficientnet_b2,    'classifier'),
        'efficientnet_b3':    (models.efficientnet_b3,    'classifier'),
        'efficientnet_b4':    (models.efficientnet_b4,    'classifier'),
        'efficientnet_b5':    (models.efficientnet_b5,    'classifier'),
        'efficientnet_b6':    (models.efficientnet_b6,    'classifier'),
        'efficientnet_b7':    (models.efficientnet_b7,    'classifier'),

        # MobileNetV3
        'mobilenet_v3_small': (models.mobilenet_v3_small, 'classifier'),
        'mobilenet_v3_large': (models.mobilenet_v3_large, 'classifier'),

        # Vision Transformers
        'vit_b_16':           (models.vit_b_16,           'heads'),
        'vit_b_32':           (models.vit_b_32,           'heads'),
        'vit_l_16':           (models.vit_l_16,           'heads'),
        'vit_l_32':           (models.vit_l_32,           'heads'),

        # Swin Transformers
        'swin_t':             (models.swin_t,             'head'),
        'swin_s':             (models.swin_s,             'head'),
        'swin_b':             (models.swin_b,             'head'),

        # ConvNeXt
        'convnext_tiny':      (models.convnext_tiny,      'classifier'),
        'convnext_small':     (models.convnext_small,     'classifier'),
        'convnext_base':      (models.convnext_base,      'classifier'),
        'convnext_large':     (models.convnext_large,     'classifier'),

        # RegNet
        'regnet_y_400mf':     (models.regnet_y_400mf,     'fc'),
        'regnet_y_800mf':     (models.regnet_y_800mf,     'fc'),
        'regnet_y_1_6gf':     (models.regnet_y_1_6gf,     'fc'),
        'regnet_x_1_6gf':     (models.regnet_x_1_6gf,     'fc'),

        # DenseNet
        'densenet121':        (models.densenet121,        'classifier'),
        'densenet169':        (models.densenet169,        'classifier'),
        'densenet201':        (models.densenet201,        'classifier'),

        # MaxViT
        'maxvit_t':           (models.maxvit_t,           'classifier'),
    }

    @staticmethod
    def get(backbone_name: str, pretrained: bool = True):
        """
        Instantiate a backbone, strip its classification head, and return
        (backbone_model, feature_dim).
        """
        if backbone_name not in BackboneFactory.model_map:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        constructor, head_attr = BackboneFactory.model_map[backbone_name]

        # Load pretrained weights if requested
        weights_enum = getattr(constructor, f"{constructor.__name__}_Weights", None)
        weights = weights_enum.DEFAULT if (pretrained and weights_enum) else None
        model = constructor(weights=weights)

        # Extract feature dim and remove head
        if head_attr == 'fc':
            feat_dim = model.fc.in_features
            model.fc = nn.Identity()
        elif head_attr == 'classifier':
            # assumes last layer of classifier is Linear
            last = model.classifier[-1]
            feat_dim = last.in_features
            model.classifier = nn.Identity()
        elif head_attr == 'heads':
            feat_dim = model.heads.head.in_features
            model.heads = nn.Identity()
        elif head_attr == 'head':
            # e.g. Swin: model.head is a Linear
            feat_dim = model.head.in_features
            model.head = nn.Identity()
        else:
            raise RuntimeError(f"Unexpected head_attr={head_attr}")

        return model, feat_dim

# --- dataset ---
class LatLongDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        # region mapping
        self.regions = sorted(self.df['Region_ID'].unique())
        self.reg2idx = {r: i for i, r in enumerate(self.regions)}
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.img_dir, row['filename'])
        try:
            img = Image.open(path).convert('RGB')
        except:
            img = Image.new('RGB', (224,224), (128,128,128))
        if self.transform:
            img = self.transform(img)
        lat = torch.tensor(row['latitude'], dtype=torch.float32)
        lon = torch.tensor(row['longitude'], dtype=torch.float32)
        angle_rad = row['angle'] * math.pi / 180.0
        angle = torch.tensor([math.sin(angle_rad), math.cos(angle_rad)], dtype=torch.float32)
        region = torch.tensor(self.reg2idx[row['Region_ID']], dtype=torch.long)
        return {'image': img, 'latitude': lat, 'longitude': lon,
                'angle': angle, 'region': region}

# --- model ---
class GeoRegressor(nn.Module):
    def __init__(self, backbone_name, num_regions,
                 embed_dim=16, dropout=0.3, head_dim=128, pretrained=True):
        super().__init__()
        # Backbone and region embedding
        self.backbone, feat_dim = BackboneFactory.get(backbone_name, pretrained)  # pre-trained CNN :contentReference[oaicite:2]{index=2}
        self.region_emb = nn.Embedding(num_regions, embed_dim)

        combined = feat_dim + embed_dim + 2
        # Heads now predict two params (α, β) per output
        def make_head():
            return nn.Sequential(
                nn.Linear(combined, head_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(head_dim, head_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(head_dim // 2, 1), 
            )
        self.head_lat = make_head()
        self.head_lon = make_head()

    def forward(self, image, region, angle):
        feat = self.backbone(image)
        if feat.ndim == 4:
            feat = feat.flatten(1)
        r = self.region_emb(region)
        x = torch.cat([feat, r, angle], dim=1)

        # activate = lambda x: F.softplus(x) / (1 + F.softplus(x))

        # Latitude head
        raw_lat = self.head_lat(x).squeeze(1)
        # Longitude head
        raw_lon= self.head_lon(x).squeeze(1)
        return raw_lat, raw_lon
    
# --- transforms ---
def get_transforms(train: bool = True):
    norm = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, (0.8,1.0)),
            transforms.ColorJitter(0.2,0.2,0.2,0.1),
            transforms.GaussianBlur(3, (0.1,2.0)),
            transforms.ToTensor(), norm,
            transforms.RandomAffine(15,(0.1,0.1),(0.9,1.1),10),
            transforms.RandomErasing(0.2,(0.02,0.15))
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224),
            transforms.ToTensor(), norm
        ])

# --- training & validation utilities ---

def train_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc='Train', leave=False)
    for batch in pbar:
        imgs = batch['image'].to(device)
        lats = batch['latitude'].to(device)
        lons = batch['longitude'].to(device)
        angles = batch['angle'].to(device)
        regs = batch['region'].to(device)
        opt.zero_grad()
        pred_lat, pred_lon = model(imgs, regs, angles)
        loss = (loss_fn(pred_lat, lats) + loss_fn(pred_lon, lons)) * 0.5

        loss.backward()
        opt.step()
        batch_loss = loss.item()
        total_loss += batch_loss * imgs.size(0)
        pbar.set_postfix(loss=batch_loss)
    pbar.close()
    return total_loss / len(loader.dataset)

def validate(model, loader, loss_fn, device, lat_scaler, lon_scaler):
    model.eval()
    losses = 0.0;
    all_true = []
    all_pred = []
    pbar = tqdm(loader, desc='Validate', leave=False)
    with torch.no_grad():
        for batch in pbar:
            imgs = batch['image'].to(device)
            lats = batch['latitude'].to(device)
            lons = batch['longitude'].to(device)
            angles = batch['angle'].to(device)
            regs = batch['region'].to(device)
            p_lat, p_lon = model(imgs, regs, angles)
            batch_loss = ((loss_fn(p_lat,lats) + loss_fn(p_lon,lons))*0.5).item()
            losses += batch_loss*imgs.size(0)
            all_true.append(torch.cat([lats,lons]).view(-1, 2).cpu())
            all_pred.append(torch.cat([p_lat,p_lon]).view(-1, 2).cpu())
            pbar.set_postfix(val_loss=batch_loss)
    pbar.close()
    # concat
    true = torch.cat(all_true).numpy()
    pred = torch.cat(all_pred).numpy()
    # inverse scale
    # true_scaled = np.concatenate([lat_scaler.inverse_transform(true[:,0:1]),
    #                               lon_scaler.inverse_transform(true[:,1:2])], axis=1)
    # pred_scaled = np.concatenate([lat_scaler.inverse_transform(pred[:,0:1]),
    #                               lon_scaler.inverse_transform(pred[:,1:2])], axis=1)
    # metrics
    mse_orig = ((true - pred)**2).mean(axis=0).tolist()
    # mse_scaled = ((true_scaled - pred_scaled)**2).mean(axis=0).tolist()
    return losses/len(loader.dataset), mse_orig

# --- checkpointing ---
def save_ckpt(path, model, opt, sched, epoch, val_mse):
    tqdm.write(f"Saving checkpoint: {path}")
    ckpt = {'epoch': epoch, 'model': model.state_dict(),
            'opt': opt.state_dict(), 'sched': getattr(sched,'state_dict',lambda:None)(),
            'val_mse': val_mse}
    torch.save(ckpt, path)

# --- plotting ---
import matplotlib.pyplot as plt

def plot_metrics(hist: dict, save_path: str):
    epochs = list(range(1, len(hist['train'])+1))
    plt.figure(figsize=(12,8))
    plt.plot(epochs, hist['train'], label='Train Loss')
    plt.plot(epochs, hist['val'], label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.tight_layout(); plt.savefig(save_path); plt.close()

# --- main ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./LatLongDataset')
    parser.add_argument('--output_dir', default='./outputs')
    parser.add_argument('--lat_scaler', default='lat_scaler.pkl')
    parser.add_argument('--lon_scaler', default='lon_scaler.pkl')
    parser.add_argument('--backbone', default='resnet34')
    parser.add_argument('--embed_dim', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--head_dim', type=int, default=128)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--early_stop', type=int, default=10)
    parser.add_argument('--lr_patience', type=int, default=5)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    lat_scaler = joblib.load(args.lat_scaler)
    lon_scaler = joblib.load(args.lon_scaler)

    train_ds = LatLongDataset(os.path.join(args.data_dir,'train_label.csv'), os.path.join(args.data_dir,'images_train'), get_transforms(True))
    val_ds   = LatLongDataset(os.path.join(args.data_dir,'val_label.csv'),   os.path.join(args.data_dir,'images_val'),   get_transforms(False))
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,num_workers=args.workers,pin_memory=True)

    model = GeoRegressor(args.backbone, len(train_ds.regions), args.embed_dim, args.dropout, args.head_dim).to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    loss_fn = nn.MSELoss()
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.1, patience=args.lr_patience)

    history = {'train':[], 'val':[]}
    best_mse = float('inf'); wait = 0
    # best_mse_scaled = [float('inf'), float('inf')]

    pbar = tqdm(range(1, args.epochs + 1), desc='Epochs')
    for epoch in pbar:
        train_loss = train_epoch(model, train_loader, opt, loss_fn, device)
        val_loss, mse_orig = validate(model, val_loader, loss_fn, device, lat_scaler, lon_scaler)
        sched.step(val_loss)
        history['train'].append(train_loss); history['val'].append(val_loss)
        tqdm.write(f"Epoch {epoch}: Train {train_loss:.4f}, Val {val_loss:.4f}, MSE {mse_orig}")
        # early stopping
        if mse_orig[0]+mse_orig[1] < best_mse:
            best_mse = mse_orig[0]+mse_orig[1]; wait = 0
            save_ckpt(os.path.join(args.output_dir,f"best_{args.backbone}.pth"), model, opt, sched, epoch, best_mse)
        else:
            wait += 1
            if wait >= args.early_stop:
                print("Early stopping")
                break
        pbar.set_postfix({"Best MSE": f'{best_mse/1e3}k (~35k good)'})

        # pbar.set_postfix({"Best MSE Lat": f'{best_mse_scaled[0]/1e3}k',"Best MSE Lon": f'{best_mse_scaled[1]/1e3}k'})
    # plot
    plot_metrics(history, os.path.join(args.output_dir,f"{args.backbone}_metrics.png"))
    print("Training complete.")

if __name__ == '__main__':
    main()
