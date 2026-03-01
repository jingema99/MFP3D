import argparse
import os
import sys
from datetime import datetime
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import GTFoodDataset, SUPPORTED_TARGETS
from model import PointImageRegressor

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    mae_sum = 0.0
    mape_sum = 0.0
    n = 0
    eps = 1e-6

    with torch.no_grad():
        for points, images, targets in loader:
            points = points.to(device)
            images = images.to(device)
            targets = targets.to(device)
            preds = model(points, images)

            abs_err = torch.abs(preds - targets)
            mae_sum += abs_err.sum().item()
            mape_sum += (abs_err / torch.clamp(torch.abs(targets), min=eps)).sum().item()
            n += targets.numel()

    mae = mae_sum / max(n, 1)
    mape = 100.0 * (mape_sum / max(n, 1))
    return mae, mape


def main() -> None:
    parser = argparse.ArgumentParser(description="Train point+image regressor on gt h5.")
    parser.add_argument("--data_dir", type=str, default="./data/gt")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="",
        help="输出目录中的数据集名称。默认取 data_dir 的最后一级目录名。",
    )
    parser.add_argument("--target", type=str, default="weight", choices=SUPPORTED_TARGETS)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./outputs",
        help="输出根目录。最终结构为 save_dir/dataset_name/target/...",
    )
    parser.add_argument(
        "--console_mode",
        type=str,
        default="compact",
        choices=("compact", "verbose"),
        help="compact: 单行进度条；verbose: 每个epoch打印一行。",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    dataset_name = args.dataset_name.strip() or os.path.basename(os.path.normpath(args.data_dir))
    run_dir = os.path.join(args.save_dir, dataset_name, args.target)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "train.log")

    train_h5 = os.path.join(args.data_dir, "train.h5")
    test_h5 = os.path.join(args.data_dir, "test.h5")

    train_set = GTFoodDataset(train_h5, target_key=args.target, augment=True, return_path=False)
    test_set = GTFoodDataset(test_h5, target_key=args.target, augment=False, return_path=False)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointImageRegressor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.L1Loss()

    best_mape = float("inf")
    best_path = os.path.join(run_dir, "best.pt")

    log_f = open(log_path, "a", encoding="utf-8")

    def log(msg: str, also_print: bool = False) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_f.write(f"[{ts}] {msg}\n")
        log_f.flush()
        if also_print:
            print(msg)

    print(f"Device: {device}")
    print(f"Train/Test size: {len(train_set)}/{len(test_set)}")
    print(f"Dataset: {dataset_name}")
    print(f"Target: {args.target}")
    print(f"Output dir: {run_dir}")
    print(f"Full log file: {log_path}")
    log(f"Device: {device}")
    log(f"Train/Test size: {len(train_set)}/{len(test_set)}")
    log(f"Dataset: {dataset_name}")
    log(f"Target: {args.target}")
    log(f"Output dir: {run_dir}")

    use_compact = args.console_mode == "compact"
    use_tqdm = use_compact and (tqdm is not None) and sys.stdout.isatty()
    epoch_iter = range(1, args.epochs + 1)
    epoch_bar = (
        tqdm(epoch_iter, total=args.epochs, dynamic_ncols=True, leave=True, desc="Epoch")
        if use_tqdm
        else epoch_iter
    )

    try:
        for epoch in epoch_bar:
            model.train()
            train_loss_sum = 0.0
            train_count = 0

            for step, (points, images, targets) in enumerate(train_loader, start=1):
                points = points.to(device, non_blocking=True)
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                preds = model(points, images)
                loss = criterion(preds, targets)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                bs = targets.size(0)
                train_loss_sum += loss.item() * bs
                train_count += bs

                # Full details go to file log; terminal remains compact.
                log(
                    f"epoch={epoch:03d} step={step:04d}/{len(train_loader):04d} "
                    f"batch_mae={loss.item():.6f}"
                )

            train_mae = train_loss_sum / max(train_count, 1)
            val_mae, val_mape = evaluate(model, test_loader, device)
            epoch_msg = (
                f"Epoch {epoch:03d} | train_mae={train_mae:.4f} "
                f"| val_mae={val_mae:.4f} | val_mape={val_mape:.2f}%"
            )
            if use_compact and not use_tqdm:
                # Non-TTY fallback: one updating line per epoch progress.
                status = (
                    f"\rEpoch {epoch:03d}/{args.epochs:03d} "
                    f"train_mae={train_mae:.4f} val_mae={val_mae:.4f} val_mape={val_mape:.2f}%"
                )
                sys.stdout.write(status)
                sys.stdout.flush()
                log(epoch_msg, also_print=False)
            elif use_compact:
                # In compact+tqdm mode, keep terminal output to progress bars only.
                log(epoch_msg, also_print=False)
            else:
                log(epoch_msg, also_print=(not use_compact or epoch == args.epochs))

            if use_tqdm:
                epoch_bar.set_postfix(
                    train_mae=f"{train_mae:.4f}",
                    val_mae=f"{val_mae:.4f}",
                    val_mape=f"{val_mape:.2f}%",
                )

            if val_mape < best_mape:
                best_mape = val_mape
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "target": args.target,
                        "best_mape": best_mape,
                    },
                    best_path,
                )
                log(
                    f"Saved best checkpoint to {best_path}",
                    also_print=(not use_compact),
                )

        if use_compact and not use_tqdm:
            sys.stdout.write("\n")
            sys.stdout.flush()
        if not use_compact:
            print(f"Done. Best val_mape={best_mape:.2f}%")
        log(f"Done. Best val_mape={best_mape:.2f}%")
    finally:
        log_f.close()


if __name__ == "__main__":
    main()
