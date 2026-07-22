"""
CLI entry point for training the color-cluster channel-selection model.

Usage:
  python -m channels_model.train --config channels_model/config/default_config.json
  python -m channels_model.train --images-dir path/to/images --labels-dir path/to/labels

Runs num_folds-fold cross-validation (reporting per-class accuracy,
confusion matrices, and super-accuracy each epoch), derives an epoch budget
from the CV run's best-super-accuracy epochs, then refits a fresh model on
ALL of the data for that budget and exports it to ONNX -- see config.py's
final_epochs docstring for why CV and the deployed model are different runs.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

try:
    from .config import ChannelsModelConfig
    from .dataset import build_pairs, build_train_eval_datasets
    from .export import export_onnx
    from .metrics import confusion_and_accuracy, format_confusion_matrices, probs_to_preds, super_accuracy
    from .model import ChannelsModel
except ImportError:
    # Running as `python channels_model/train.py` rather than
    # `python -m channels_model.train` -- add the project root to sys.path
    # so absolute imports resolve correctly.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from channels_model.config import ChannelsModelConfig
    from channels_model.dataset import build_pairs, build_train_eval_datasets
    from channels_model.export import export_onnx
    from channels_model.metrics import confusion_and_accuracy, format_confusion_matrices, probs_to_preds, super_accuracy
    from channels_model.model import ChannelsModel


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_device(config: ChannelsModelConfig) -> torch.device:
    if config.device in ('cuda', 'cpu'):
        return torch.device(config.device)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _resolve_amp(config: ChannelsModelConfig, device: torch.device) -> bool:
    if config.use_amp is not None:
        return config.use_amp
    return device.type == 'cuda'


def _build_optimizer(model: ChannelsModel, config: ChannelsModelConfig) -> torch.optim.Optimizer:
    params = model.parameters()
    if config.optimizer == 'adamw':
        return torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
    if config.optimizer == 'adam':
        return torch.optim.Adam(params, lr=config.lr, weight_decay=config.weight_decay)
    if config.optimizer == 'sgd':
        return torch.optim.SGD(params, lr=config.lr, weight_decay=config.weight_decay, momentum=0.9)
    raise ValueError(f"unknown optimizer: {config.optimizer}")


def _random_channel_perm(batch_size: int, num_channels: int, device: torch.device) -> torch.Tensor:
    """Per-sample random permutation of the 8 cluster channels -- see
    model.ChannelsModel.forward's channel_perm and config.py's
    aug_channel_shuffle docstring for what this augmentation does and why."""
    return torch.stack([torch.randperm(num_channels, device=device) for _ in range(batch_size)])


def _run_epoch_train(model, loader, criterion, optimizer, device, use_amp, scaler, grad_clip_norm, desc,
                      channel_shuffle: bool) -> tuple[float, float]:
    """Returns (train_loss, train_super_acc). Super-accuracy is computed
    from the SAME logits/labels already produced for the loss/gradient step
    -- no extra forward pass -- so it reflects performance on the actual
    (possibly channel-shuffled) batches the model trained on this epoch,
    same convention as train_loss."""
    model.train()
    total_loss, n = 0.0, 0
    all_probs, all_labels = [], []
    pbar = tqdm(loader, desc=desc, unit='batch', leave=False)
    for rgb, channels, labels in pbar:
        rgb, channels, labels = rgb.to(device), channels.to(device), labels.to(device)
        channel_perm = None
        if channel_shuffle:
            channel_perm = _random_channel_perm(rgb.size(0), labels.size(1), device)
            # Same per-sample permutation applied to the labels -- label[k]
            # must keep meaning "is the cluster now at channel k selected"
            # once the input channels themselves are shuffled.
            labels = torch.gather(labels, dim=1, index=channel_perm)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(rgb, channels, channel_perm=channel_perm)
            loss = criterion(logits, labels)
        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
        total_loss += loss.item() * rgb.size(0)
        n += rgb.size(0)
        pbar.set_postfix(loss=f"{total_loss / n:.4f}")
        all_probs.append(torch.sigmoid(logits.detach()).float().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
    probs = np.concatenate(all_probs)
    labels_arr = np.concatenate(all_labels)
    preds = probs_to_preds(probs)
    train_s_acc = super_accuracy(labels_arr, preds)
    return total_loss / n, train_s_acc


@torch.no_grad()
def _run_epoch_eval(model, loader, criterion, device, desc):
    model.eval()
    total_loss, n = 0.0, 0
    all_probs, all_labels = [], []
    pbar = tqdm(loader, desc=desc, unit='batch', leave=False)
    for rgb, channels, labels in pbar:
        rgb, channels, labels = rgb.to(device), channels.to(device), labels.to(device)
        logits = model(rgb, channels)
        loss = criterion(logits, labels)
        total_loss += loss.item() * rgb.size(0)
        n += rgb.size(0)
        pbar.set_postfix(loss=f"{total_loss / n:.4f}")
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    probs = np.concatenate(all_probs)
    labels_arr = np.concatenate(all_labels)
    preds = probs_to_preds(probs)
    per_class_acc, conf = confusion_and_accuracy(labels_arr, preds)
    s_acc = super_accuracy(labels_arr, preds)
    return total_loss / n, per_class_acc, conf, s_acc


def _make_loaders(train_ds, eval_ds, train_idx, val_idx, config):
    train_loader = DataLoader(
        Subset(train_ds, train_idx), batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        Subset(eval_ds, val_idx), batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers,
    )
    return train_loader, val_loader


def _run_fold(fold_idx, num_folds, train_ds, eval_ds, train_idx, val_idx, config, device, use_amp) -> dict:
    train_loader, val_loader = _make_loaders(train_ds, eval_ds, train_idx, val_idx, config)

    criterion = nn.BCEWithLogitsLoss()

    model = ChannelsModel(config).to(device)
    optimizer = _build_optimizer(model, config)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
        if config.use_lr_scheduler else None
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    best_super_acc, best_epoch, best_test_loss = -1.0, 0, float('inf')
    for epoch in range(1, config.epochs + 1):
        train_loss, train_s_acc = _run_epoch_train(
            model, train_loader, criterion, optimizer, device, use_amp, scaler, config.grad_clip_norm,
            desc=f"Fold {fold_idx}/{num_folds} epoch {epoch}/{config.epochs} [train]",
            channel_shuffle=config.aug_channel_shuffle,
        )
        test_loss, per_class_acc, conf, test_s_acc = _run_epoch_eval(
            model, val_loader, criterion, device,
            desc=f"Fold {fold_idx}/{num_folds} epoch {epoch}/{config.epochs} [eval]",
        )
        if scheduler is not None:
            scheduler.step()

        print(f"[Fold {fold_idx}/{num_folds}][Epoch {epoch}/{config.epochs}] "
              f"train_loss={train_loss:.4f} test_loss={test_loss:.4f} "
              f"train_super_acc={train_s_acc:.3f} test_super_acc={test_s_acc:.3f}")
        print(f"  per_class_acc: {np.round(per_class_acc, 3).tolist()}")
        print("  Confusion matrices (rows=actual/predicted counts, cols=channel):")
        print(format_confusion_matrices(conf))

        # Fold/epoch selection is based on VALIDATION (test) super-accuracy,
        # never train -- picking by train performance would just reward
        # overfitting.
        if test_s_acc > best_super_acc or (test_s_acc == best_super_acc and test_loss < best_test_loss):
            best_super_acc, best_epoch, best_test_loss = test_s_acc, epoch, test_loss

    return {
        'fold': fold_idx, 'best_epoch': best_epoch,
        'best_super_acc': best_super_acc, 'best_test_loss': best_test_loss,
    }


def _final_refit(pairs, config, device, use_amp, final_epochs) -> ChannelsModel:
    full_ds, _ = build_train_eval_datasets(pairs, config)
    loader = DataLoader(full_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    criterion = nn.BCEWithLogitsLoss()

    model = ChannelsModel(config).to(device)
    optimizer = _build_optimizer(model, config)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=final_epochs)
        if config.use_lr_scheduler else None
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    print(f"\n=== Final refit on all {len(pairs)} sample(s) for {final_epochs} epoch(s) ===")
    for epoch in range(1, final_epochs + 1):
        train_loss, train_s_acc = _run_epoch_train(
            model, loader, criterion, optimizer, device, use_amp, scaler, config.grad_clip_norm,
            desc=f"Final refit epoch {epoch}/{final_epochs}",
            channel_shuffle=config.aug_channel_shuffle,
        )
        if scheduler is not None:
            scheduler.step()
        print(f"[Final][Epoch {epoch}/{final_epochs}] train_loss={train_loss:.4f} train_super_acc={train_s_acc:.3f}")

    return model


def run(config: ChannelsModelConfig) -> None:
    _set_seed(config.seed)
    device = _resolve_device(config)
    use_amp = _resolve_amp(config, device)
    print(f"Using device: {device} | AMP enabled: {use_amp}")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = build_pairs(config.images_dir, config.labels_dir)
    print(f"Loaded {len(pairs)} paired sample(s) from {config.images_dir} / {config.labels_dir}")

    train_ds, eval_ds = build_train_eval_datasets(pairs, config)

    kfold = KFold(n_splits=config.num_folds, shuffle=True, random_state=config.seed)
    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(range(len(pairs))), start=1):
        result = _run_fold(
            fold_idx, config.num_folds, train_ds, eval_ds, train_idx, val_idx, config, device, use_amp
        )
        fold_results.append(result)

    best_epochs = [r['best_epoch'] for r in fold_results]
    best_super_accs = [r['best_super_acc'] for r in fold_results]
    print("\n=== Cross-validation summary ===")
    for r in fold_results:
        print(f"  fold {r['fold']}: best_epoch={r['best_epoch']} best_super_acc={r['best_super_acc']:.3f} "
              f"best_test_loss={r['best_test_loss']:.4f}")
    print(f"  mean best_epoch={np.mean(best_epochs):.1f} (std {np.std(best_epochs):.1f})")
    print(f"  mean best_super_acc={np.mean(best_super_accs):.3f} (std {np.std(best_super_accs):.3f})")

    final_epochs = config.final_epochs
    if final_epochs is None:
        final_epochs = max(1, round(float(np.mean(best_epochs))))
        print(f"  derived final_epochs (mean of per-fold best epochs) = {final_epochs}")
    else:
        print(f"  using explicitly configured final_epochs = {final_epochs}")

    final_model = _final_refit(pairs, config, device, use_amp, final_epochs)

    checkpoint_path = output_dir / 'final_model.pt'
    torch.save(final_model.state_dict(), checkpoint_path)
    print(f"\nSaved final model checkpoint -> {checkpoint_path}")

    cv_results_path = output_dir / 'cv_results.json'
    with open(cv_results_path, 'w') as f:
        json.dump({'folds': fold_results, 'final_epochs': final_epochs}, f, indent=2)
    print(f"Saved CV results -> {cv_results_path}")

    effective_config_path = output_dir / 'effective_config.json'
    config.save(effective_config_path)
    print(f"Saved effective config -> {effective_config_path}")

    onnx_path = output_dir / 'channels_model.onnx'
    export_onnx(final_model.to('cpu'), config, onnx_path)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='channels_model.train',
        description='Train the color-cluster channel-selection model via K-fold CV, '
                     'then refit on all data and export to ONNX.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--config', '-c', type=Path, default=None, help='Path to a ChannelsModelConfig JSON.')
    p.add_argument('--images-dir', type=str, default=None)
    p.add_argument('--labels-dir', type=str, default=None)
    p.add_argument('--output', '-o', dest='output_dir', type=str, default=None)
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--final-epochs', type=int, default=None)
    p.add_argument('--folds', type=int, default=None)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--save-config', type=Path, default=None, metavar='PATH',
                    help='Save the effective config to a JSON file after parsing all arguments.')
    return p


def main() -> None:
    args = _build_parser().parse_args()
    config = ChannelsModelConfig.from_args(args)
    if args.save_config is not None:
        config.save(args.save_config)
        print(f"Config saved to {args.save_config}")
    run(config)


if __name__ == '__main__':
    main()
