"""
Коротко про подход к решению:
1. Преобразуем PPG-сигналы в спектрограммы и кешируем как PNG
2. Используем Resnet18 (претрейн на ImageNet) с регрессионной головой
3. Двухфазное обучение: сначала обучаем новую голову + поздние слои (с меньшим lr) 6 эпох, 
   затем все слои (с меньшим lr) 10 эпох, оба этапа с ранней остановкой по val
4. Кросс-валидация на 5 фолдах (со стратифицированием по SBP/DBP и числу чанков) с ансамблированием для финального предсказания
5. Эксперименты с num_chunks=0 в трейне не участвуют, а в тесте заменяются средним по трейну.
"""

import os
import random

import numpy as np
import pandas as pd

import h5py
from PIL import Image

from scipy.signal import butter, filtfilt, spectrogram

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from torchvision.models import ResNet18_Weights

from sklearn.model_selection import StratifiedKFold


SEED = 42
IMG_SIZE = 224
FS = 30.0
BANDPASS = (0.5, 6.0)
FMIN, FMAX = 0.3, 6.0
NPERSEG, NOVERLAP, NFFT = 64, 48, 256
EPS = 1e-6

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# Загрузка сигнала из HDF5 файла
def load_experiment_signal(h5_path):
    with h5py.File(h5_path, 'r') as f:
        sorted_keys = sorted(f.keys(), key=lambda k: int(k.split('_')[1]))
        data_list = [np.asarray(f[key][:], dtype=np.float32).reshape(-1) for key in sorted_keys]
        if len(data_list) == 0:
            return np.zeros(1, dtype=np.float32)
        x = np.concatenate(data_list, axis=0)
    return x

# Полосовой фильтр
def bandpass_filter(x, fs=FS, lo=BANDPASS[0], hi=BANDPASS[1], order=4):
    if x.size <= 1 or not np.isfinite(x).all():
        return np.nan_to_num(x, nan=0.0)

    b, a = butter(order, [lo/(fs/2), hi/(fs/2)], btype='band')
    padlen_default = 3 * (max(len(a), len(b)) - 1)

    if x.size > padlen_default:
        return filtfilt(b, a, x)
    else:
        return filtfilt(b, a, x, method='gust')

# Робастная нормализация по медиане и IQR
def robust_norm(x):
    med = np.median(x)
    iqr = np.subtract(*np.percentile(x, [75, 25])) + EPS
    z = (x - med) / iqr
    lo, hi = np.percentile(z, [1, 99])
    return np.clip(z, lo, hi)

# Преобразование сигнала в спектрограмму
def long_signal_to_spectrogram(x,fs=FS,fmin=FMIN, fmax=FMAX,nperseg=NPERSEG, noverlap=NOVERLAP, nfft=NFFT):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == 0:
        x = np.zeros(8, dtype=np.float32)
    x = bandpass_filter(x, fs=fs)
    x = robust_norm(x)

    nper = int(min(nperseg, x.size))
    if nper < 8:
        nper = x.size
    nov = int(min(noverlap, max(0, nper - 1)))
    nfft_eff = max(nfft, 2**int(np.ceil(np.log2(max(8, nper)))))

    f, t, Sxx = spectrogram(x, fs=fs, window='hann', nperseg=nper, noverlap=nov, nfft=nfft_eff, detrend=False, scaling='spectrum', mode='magnitude')
    band = (f >= fmin) & (f <= fmax)
    if band.sum() < 2:
        band = slice(None)
    S = Sxx[band]
    S = np.log1p(S).astype(np.float32)
    return S

# Конвертация спектрограммы в изображение
def spec_to_uint8_image(S, out_size=IMG_SIZE):
    if S.size == 0:
        S = np.zeros((64, 64), dtype=np.float32)
    p1, p99 = np.percentile(S, [1, 99])
    if not np.isfinite(p1) or not np.isfinite(p99) or p99 <= p1:
        p1, p99 = float(np.nanmin(S)), float(np.nanmax(S))
        if not np.isfinite(p1) or not np.isfinite(p99) or p99 <= p1:
            S = np.zeros((64, 64), dtype=np.float32)
            p1, p99 = 0.0, 1.0
    S = np.clip(S, p1, p99)
    S = (S - p1) / (p99 - p1 + EPS)
    S = (S * 255.0).astype(np.uint8)
    img = Image.fromarray(S)
    img = img.resize((out_size, out_size), resample=Image.BILINEAR)
    img = img.convert('L')
    return img

# Предварительная генерация изображений спектрограмм
def precompute_cache_for_df(df, split_name, cache_root, overwrite=False):
    os.makedirs(cache_root, exist_ok=True)
    out_dir = os.path.join(cache_root, split_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f'Precomputing {split_name} cache into {out_dir} ...')
    for idx in range(len(df)):
        row = df.iloc[idx]
        guid = str(row['experiment_guid'])
        h5_path = row['data_hdf5_path']
        out_path = os.path.join(out_dir, f'{guid}.png')

        if (not overwrite) and os.path.exists(out_path):
            continue

        try:
            x = load_experiment_signal(h5_path)
            S = long_signal_to_spectrogram(x)
            img = spec_to_uint8_image(S, out_size=IMG_SIZE)
            img.save(out_path, format='PNG', compress_level=4)
        except Exception as e:
            print(f'Warning: failed {guid} ({h5_path}): {e}')
            img = Image.fromarray(np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8))
            img.save(out_path, format='PNG', compress_level=4)

    print('Done.')

# Преобразование одноканального изображения в трёхканальное
class RepeatTo3Channels(torch.nn.Module):
    def forward(self, x):
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
        return x

# Маскирование по частоте и времени
class RandomFreqTimeMask(torch.nn.Module):
    def __init__(self, max_freq_pct=0.08, max_time_pct=0.1, num_freq_masks=1, num_time_masks=1, fill=0.0):
        super().__init__()
        self.max_freq_pct = max_freq_pct
        self.max_time_pct = max_time_pct
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.fill = fill
    def forward(self, x):
        C, H, W = x.shape
        for _ in range(self.num_freq_masks):
            f = int(np.random.uniform(0, self.max_freq_pct) * H)
            if f > 0:
                f0 = np.random.randint(0, H - f + 1)
                x[:, f0:f0+f, :] = self.fill
        for _ in range(self.num_time_masks):
            w = int(np.random.uniform(0, self.max_time_pct) * W)
            if w > 0:
                t0 = np.random.randint(0, W - w + 1)
                x[:, :, t0:t0+w] = self.fill
        return x

class PPGCachedDataset(Dataset):
    def __init__(self, df, cache_root, split_name, targets_cols, augment=False):
        self.df = df.reset_index(drop=True)
        self.cache_dir = os.path.join(cache_root, split_name)
        self.split_name = split_name
        self.return_targets = targets_cols is not None and all(c in df.columns for c in targets_cols)
        self.targets_cols = targets_cols if self.return_targets else None

        t_list = [T.ToTensor()]
        if augment:
            t_list += [RandomFreqTimeMask(max_freq_pct=0.08, max_time_pct=0.1, num_freq_masks=1, num_time_masks=1)]
        t_list += [RepeatTo3Channels(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        self.transform = T.Compose(t_list)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        guid = str(row['experiment_guid'])
        img_path = os.path.join(self.cache_dir, f'{guid}.png')
        img = Image.open(img_path).convert('L')
        x = self.transform(img)
        if self.return_targets:
            y = torch.tensor([float(row[self.targets_cols[0]]), float(row[self.targets_cols[1]])], dtype=torch.float32)
            return x, y, guid
        else:
            return x, guid


# Разметка фолдов со стратификацией по SBP, DBP и числу чанков
def add_folds_with_chunks(df, n_folds=5, seed=SEED, sbp_bins=5, dbp_bins=5):
    df = df.copy()
    sbp_q = pd.qcut(df['sbp_mean_experiment'], q=sbp_bins, duplicates='drop').cat.codes
    dbp_q = pd.qcut(df['dbp_mean_experiment'], q=dbp_bins, duplicates='drop').cat.codes
    cuts = [-1, 5, 16, 18, 10**9]
    nch_cat = pd.cut(df['num_chunks'], bins=cuts, labels=False, include_lowest=True).astype(int)
    strat = sbp_q.astype(int) * 100 + dbp_q.astype(int) * 10 + nch_cat
    df['fold'] = -1
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (_, val_idx) in enumerate(skf.split(df, strat)):
        df.loc[df.index[val_idx], 'fold'] = fold
    return df

# Resnet18 с регрессионной головой на 2 выхода (SBP, DBP)
class ResNet18Regressor(nn.Module):
    def __init__(self, pretrained=True, freeze_upto='layer2', dropout=0.25):
        super().__init__()
        m = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.backbone = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3, m.layer4)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        in_feat = m.fc.in_features
        self.head = nn.Sequential(nn.Linear(in_feat, 128), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(128, 2))
        self.freeze_upto(freeze_upto)

    def freeze_upto(self, up_to='layer2'):
        freeze_names = ['conv1','bn1','layer1','layer2']
        to_freeze = set(freeze_names[:freeze_names.index(up_to)+1] if up_to in freeze_names else [])
        for name, module in self.backbone.named_children():
            requires_grad = not (name in to_freeze)
            for p in module.parameters():
                p.requires_grad = requires_grad

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x).flatten(1)
        x = self.head(x)
        return x

# Обучение одной эпохи со штрафом по лоссу (DBP < SBP)
def train_one_epoch(model, loader, optimizer, scaler, lam_constraint=0.05):
    model.train()
    running = 0.0
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(enabled=True, device_type='cuda'):
            pred = model(x)
            mse = F.mse_loss(pred, y, reduction='none').mean(dim=0).sum()
            penalty = F.relu(pred[:,1] - pred[:,0] + 5).mean()
            loss = mse + lam_constraint * penalty
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running += loss.item() * x.size(0)
    return running / len(loader.dataset)

# Валидация
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total = 0
    se_sum = torch.zeros(2, device=device)
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x)
        se_sum += ((pred - y) ** 2).sum(dim=0)
        total += x.size(0)
    mse_per_target = se_sum / total
    return float(mse_per_target.sum().item()), mse_per_target.detach().cpu().numpy()

# Обучение одного фолда
def train_fold_cached(df_fold, fold: int, cache_root='./cache_specs',
                      epochs_head=6, epochs_finetune=10,
                      batch_size=64, lr_head=3e-4, lr_backbone=1e-4,
                      wd=1e-4, num_workers=6):
    train_df = df_fold[df_fold['fold'] != fold].reset_index(drop=True)
    val_df   = df_fold[df_fold['fold'] == fold].reset_index(drop=True)

    train_ds = PPGCachedDataset(train_df, cache_root=cache_root, split_name='train', augment=True, imagenet_norm=True)
    val_ds   = PPGCachedDataset(val_df, cache_root=cache_root, split_name='train', augment=False, imagenet_norm=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=max(2, num_workers//2), pin_memory=True, persistent_workers=True)

    model = ResNet18Regressor(pretrained=True, freeze_upto='layer2', dropout=0.25).to(device)

    head_params = list(model.head.parameters())
    backbone_params = [p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('head.')]
    optimizer = torch.optim.AdamW([
        {'params': head_params, 'lr': lr_head},
        {'params': backbone_params, 'lr': lr_backbone}
    ], weight_decay=wd)
    scaler = torch.amp.GradScaler(enabled=True)

    best_val, best_state = 1e9, None
    patience, no_improve = 6, 0

    # Учим голову + размороженную часть
    for epoch in range(epochs_head):
        tr_loss = train_one_epoch(model, train_loader, optimizer, scaler)
        val_sum_mse, mse_vec = evaluate(model, val_loader)
        print(f'[Fold {fold}] Head E{epoch+1}/{epochs_head}: train_loss={tr_loss:.4f} val_sumMSE={val_sum_mse:.4f} (SBP={mse_vec[0]:.4f}, DBP={mse_vec[1]:.4f})')
        if val_sum_mse < best_val:
            best_val, best_state, no_improve = val_sum_mse, {k:v.cpu() for k,v in model.state_dict().items()}, 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # Размораживаем всё и дообучаем
    model.freeze_upto('')
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_backbone, weight_decay=wd)
    no_improve = 0
    for epoch in range(epochs_finetune):
        tr_loss = train_one_epoch(model, train_loader, optimizer, scaler)
        val_sum_mse, mse_vec = evaluate(model, val_loader)
        print(f'[Fold {fold}] Finetune E{epoch+1}/{epochs_finetune}: train_loss={tr_loss:.4f} val_sumMSE={val_sum_mse:.4f} (SBP={mse_vec[0]:.4f}, DBP={mse_vec[1]:.4f})')
        if val_sum_mse < best_val:
            best_val, best_state, no_improve = val_sum_mse, {k:v.cpu() for k,v in model.state_dict().items()}, 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    
    # Загружаем лучшие веса
    if best_state is not None:
        model.load_state_dict({k:v.to(device) for k,v in best_state.items()})
    return model, best_val


# Инференс
@torch.no_grad()
def predict_dataset(model, loader):
    model.eval()
    preds, guids = [], []
    for batch in loader:
        if len(batch) == 3:
            x, _, ids = batch
        else:
            x, ids = batch
        x = x.to(device, non_blocking=True)
        y = model(x).detach().cpu().numpy()
        preds.append(y)
        guids += list(ids)
    preds = np.concatenate(preds, axis=0)
    return guids, preds

def main(train_csv='train.csv', test_csv='test.csv', cache_root='./cache_specs', epochs_head=6, epochs_finetune=10, batch_size=64, num_workers=6):
    train_df_full = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    # Фильтруем пустые эксперименты из трейна
    def valid_h5(p):
        try:
            return os.path.exists(p) and os.path.getsize(p) > 1024
        except:
            return False
    mask_valid = (train_df_full['num_chunks'] > 0) & train_df_full['data_hdf5_path'].map(valid_h5)
    removed = (~mask_valid).sum()
    print(f'Фильтруем пустые train-эксперименты: удаляем {removed} из {len(train_df_full)}')
    train_df = train_df_full[mask_valid].reset_index(drop=True)

    # Размечаем фолды со стратификацией по SBP/DBP и числу чанков
    train_df = add_folds_with_chunks(train_df, n_folds=5, seed=SEED, sbp_bins=5, dbp_bins=5)

    # Обучаем 5 фолдов
    models_fold = []
    cv_scores = []
    for fold in range(5):
        print(f'=== Training fold {fold} ===')
        model, val_score = train_fold_cached(train_df, fold, cache_root=cache_root,
                                             epochs_head=epochs_head, epochs_finetune=epochs_finetune,
                                             batch_size=batch_size, num_workers=num_workers)
        models_fold.append(model)
        cv_scores.append(val_score)
        os.makedirs('./weights', exist_ok=True)
        torch.save(model.state_dict(), f'./weights/resnet18_fold{fold}.pth')
    print('CV sumMSE per fold:', cv_scores, 'mean:', np.mean(cv_scores))

    # Инференс на тесте
    test_ds = PPGCachedDataset(test_df, cache_root=cache_root, split_name='test', targets_cols=None, augment=False, imagenet_norm=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)

    all_preds = []
    for m in models_fold:
        _, preds = predict_dataset(m, test_loader)
        all_preds.append(preds)
    preds_mean = np.mean(np.stack(all_preds, axis=0), axis=0)

    submit = test_df.copy()
    submit['sbp_mean_experiment'] = preds_mean[:,0]
    submit['dbp_mean_experiment'] = preds_mean[:,1]

    # Отдельно пустые эксперименты
    sbp_mu = train_df['sbp_mean_experiment'].mean()
    dbp_mu = train_df['dbp_mean_experiment'].mean()
    mask_empty_test = (test_df['num_chunks'] == 0) | (~test_df['data_hdf5_path'].map(valid_h5))
    n_empty = mask_empty_test.sum()
    if n_empty > 0:
        submit.loc[mask_empty_test, 'sbp_mean_experiment'] = sbp_mu
        submit.loc[mask_empty_test, 'dbp_mean_experiment']  = dbp_mu

    submit.to_csv('submission.csv', index=False)
    print('Saved submission.csv')

if __name__ == '__main__':
    main(
        train_csv='train.csv',
        test_csv='test.csv',
        cache_root='./cache_specs',
        epochs_head=6,
        epochs_finetune=10,
        batch_size=128,
        num_workers=6
    )