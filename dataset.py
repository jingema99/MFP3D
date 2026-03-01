import math
from typing import Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


SUPPORTED_TARGETS = ("weight", "volume", "energy", "protein", "fat", "carb")


def _translate_pointcloud(pointcloud: np.ndarray) -> np.ndarray:
    xyz1 = np.random.uniform(low=2.0 / 3.0, high=3.0 / 2.0, size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
    return (pointcloud * xyz1 + xyz2).astype(np.float32)


def _rotate_pointcloud(pointcloud: np.ndarray) -> np.ndarray:
    theta = 2.0 * math.pi * np.random.uniform()
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.float32,
    )
    out = pointcloud.copy()
    out[:, [0, 2]] = out[:, [0, 2]].dot(rotation)
    return out


class GTFoodDataset(Dataset):
    """
    H5 schema expected:
    - data: (N, 1024, 3)
    - image: (N, 3, 224, 224)
    - image_path: (N,)
    - nutrition/target keys: weight, volume, energy, protein, fat, carb
    """

    def __init__(
        self,
        h5_path: str,
        target_key: str = "weight",
        augment: bool = False,
        return_path: bool = False,
    ) -> None:
        if target_key not in SUPPORTED_TARGETS:
            raise ValueError(
                f"Unsupported target_key={target_key}. "
                f"Use one of {SUPPORTED_TARGETS}."
            )

        with h5py.File(h5_path, "r") as f:
            required = {"data", "image", "image_path", target_key}
            missing = [k for k in required if k not in f]
            if missing:
                raise KeyError(f"Missing keys in {h5_path}: {missing}")

            self.points = f["data"][:].astype(np.float32)
            self.images = f["image"][:].astype(np.float32)
            self.targets = f[target_key][:].astype(np.float32)
            self.image_paths = f["image_path"][:]

        self.target_key = target_key
        self.augment = augment
        self.return_path = return_path

        # Drop invalid targets to avoid NaN/Inf losses during training.
        valid = np.isfinite(self.targets)
        if not np.all(valid):
            self.points = self.points[valid]
            self.images = self.images[valid]
            self.targets = self.targets[valid]
            self.image_paths = self.image_paths[valid]

        if self.targets.size == 0:
            raise ValueError(f"No valid samples left in {h5_path} for target={target_key}.")

    def __len__(self) -> int:
        return self.points.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pointcloud = self.points[idx]
        image = self.images[idx]
        target = self.targets[idx]

        if self.augment:
            pointcloud = _translate_pointcloud(pointcloud)
            pointcloud = _rotate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)

        points_t = torch.from_numpy(pointcloud)  # (1024, 3)
        image_t = torch.from_numpy(image)        # (3, 224, 224)
        target_t = torch.tensor(target, dtype=torch.float32)

        if not self.return_path:
            return points_t, image_t, target_t

        raw_path = self.image_paths[idx]
        img_path = raw_path.decode("utf-8") if isinstance(raw_path, (bytes, bytearray)) else str(raw_path)
        return points_t, image_t, target_t, img_path
