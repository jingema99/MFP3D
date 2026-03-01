import argparse
import csv
import os

import torch
from torch.utils.data import DataLoader

from dataset import GTFoodDataset, SUPPORTED_TARGETS
from model import PointImageRegressor


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on gt test.h5.")
    parser.add_argument("--data_dir", type=str, default="./data/gt")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="",
        help="输出目录中的数据集名称。默认取 data_dir 的最后一级目录名。",
    )
    parser.add_argument("--target", type=str, default="weight", choices=SUPPORTED_TARGETS)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="模型路径。留空时将读取 output_root/dataset_name/target/best.pt",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="./outputs",
        help="输出根目录。默认结构 output_root/dataset_name/target/",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_csv", type=str, default="")
    args = parser.parse_args()

    dataset_name = args.dataset_name.strip() or os.path.basename(os.path.normpath(args.data_dir))
    run_dir = os.path.join(args.output_root, dataset_name, args.target)
    os.makedirs(run_dir, exist_ok=True)

    test_h5 = os.path.join(args.data_dir, "test.h5")
    test_set = GTFoodDataset(
        test_h5,
        target_key=args.target,
        augment=False,
        return_path=bool(args.save_csv),
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
    checkpoint = args.checkpoint or os.path.join(run_dir, "best.pt")
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    mae_sum = 0.0
    mape_sum = 0.0
    n = 0
    eps = 1e-6
    rows = []

    with torch.no_grad():
        for batch in test_loader:
            if args.save_csv:
                points, images, targets, paths = batch
            else:
                points, images, targets = batch
                paths = None

            points = points.to(device, non_blocking=True)
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            preds = model(points, images)
            abs_err = torch.abs(preds - targets)
            mae_sum += abs_err.sum().item()
            mape_sum += (abs_err / torch.clamp(torch.abs(targets), min=eps)).sum().item()
            n += targets.numel()

            if args.save_csv:
                for i in range(targets.size(0)):
                    rows.append(
                        [
                            paths[i],
                            float(targets[i].item()),
                            float(preds[i].item()),
                            float(abs_err[i].item()),
                        ]
                    )

    mae = mae_sum / max(n, 1)
    mape = 100.0 * (mape_sum / max(n, 1))
    print(f"Test target={args.target} | MAE={mae:.4f} | MAPE={mape:.2f}%")

    if args.save_csv:
        save_csv = args.save_csv or os.path.join(run_dir, f"pred_{args.target}.csv")
        os.makedirs(os.path.dirname(save_csv) or ".", exist_ok=True)
        with open(save_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", f"true_{args.target}", f"pred_{args.target}", "abs_error"])
            writer.writerows(rows)
        print(f"Saved prediction csv: {save_csv}")


if __name__ == "__main__":
    main()
