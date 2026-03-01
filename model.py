import torch
import torch.nn as nn
import torch.nn.functional as F


class PointEncoder(nn.Module):
    def __init__(self, out_dim: int = 512) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(256 * 2, out_dim)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        # Accept (B, N, 3) or (B, 3, N)
        if points.dim() != 3:
            raise ValueError(f"points must be 3D tensor, got shape={tuple(points.shape)}")
        if points.shape[1] != 3 and points.shape[2] == 3:
            points = points.permute(0, 2, 1)
        elif points.shape[1] != 3:
            raise ValueError(f"points expected shape (B,N,3) or (B,3,N), got {tuple(points.shape)}")

        x = self.conv(points)
        x_max = F.adaptive_max_pool1d(x, 1).squeeze(-1)
        x_avg = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        x = torch.cat([x_max, x_avg], dim=1)
        return self.fc(x)


class ImageEncoder(nn.Module):
    def __init__(self, out_dim: int = 256) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(256, out_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() != 4:
            raise ValueError(f"images must be 4D tensor, got shape={tuple(images.shape)}")
        x = self.backbone(images)
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        return self.fc(x)


class PointImageRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.point_encoder = PointEncoder(out_dim=512)
        self.image_encoder = ImageEncoder(out_dim=256)
        self.regressor = nn.Sequential(
            nn.Linear(512 + 256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 1),
        )

    def forward(self, points: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        p = self.point_encoder(points)
        i = self.image_encoder(images)
        x = torch.cat([p, i], dim=1)
        return self.regressor(x).squeeze(1)
