import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from curvenet_util import CIC, LPFA


curve_config = {
    "default": [[100, 5], [100, 5], None, None],
    "long": [[10, 30], None, None, None],
}


class PointEncoder(nn.Module):
    def __init__(self, out_dim: int = 512, k: int = 20, setting: str = "default") -> None:
        super().__init__()
        if out_dim != 512:
            raise ValueError("PointEncoder uses the original CurveNet feature size of 512.")
        if setting not in curve_config:
            raise ValueError(f"Unsupported CurveNet setting: {setting}")

        additional_channel = 32
        self.lpfa = LPFA(9, additional_channel, k=k, mlp_num=1, initial=True)

        self.cic11 = CIC(
            npoint=1024,
            radius=0.05,
            k=k,
            in_channels=additional_channel,
            output_channels=64,
            bottleneck_ratio=2,
            mlp_num=1,
            curve_config=curve_config[setting][0],
        )
        self.cic12 = CIC(
            npoint=1024,
            radius=0.05,
            k=k,
            in_channels=64,
            output_channels=64,
            bottleneck_ratio=4,
            mlp_num=1,
            curve_config=curve_config[setting][0],
        )

        self.cic21 = CIC(
            npoint=1024,
            radius=0.05,
            k=k,
            in_channels=64,
            output_channels=128,
            bottleneck_ratio=2,
            mlp_num=1,
            curve_config=curve_config[setting][1],
        )
        self.cic22 = CIC(
            npoint=1024,
            radius=0.1,
            k=k,
            in_channels=128,
            output_channels=128,
            bottleneck_ratio=4,
            mlp_num=1,
            curve_config=curve_config[setting][1],
        )

        self.cic31 = CIC(
            npoint=256,
            radius=0.1,
            k=k,
            in_channels=128,
            output_channels=256,
            bottleneck_ratio=2,
            mlp_num=1,
            curve_config=curve_config[setting][2],
        )
        self.cic32 = CIC(
            npoint=256,
            radius=0.2,
            k=k,
            in_channels=256,
            output_channels=256,
            bottleneck_ratio=4,
            mlp_num=1,
            curve_config=curve_config[setting][2],
        )

        self.cic41 = CIC(
            npoint=64,
            radius=0.2,
            k=k,
            in_channels=256,
            output_channels=512,
            bottleneck_ratio=2,
            mlp_num=1,
            curve_config=curve_config[setting][3],
        )
        self.cic42 = CIC(
            npoint=64,
            radius=0.4,
            k=k,
            in_channels=512,
            output_channels=512,
            bottleneck_ratio=4,
            mlp_num=1,
            curve_config=curve_config[setting][3],
        )

        self.conv0 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Linear(1024 * 2, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if points.dim() != 3:
            raise ValueError(f"points must be 3D tensor, got shape={tuple(points.shape)}")
        if points.shape[1] != 3 and points.shape[2] == 3:
            points = points.permute(0, 2, 1)
        elif points.shape[1] != 3:
            raise ValueError(f"points expected shape (B,N,3) or (B,3,N), got {tuple(points.shape)}")

        xyz = points
        l0_points = self.lpfa(xyz, xyz)

        l1_xyz, l1_points = self.cic11(xyz, l0_points)
        l1_xyz, l1_points = self.cic12(l1_xyz, l1_points)

        l2_xyz, l2_points = self.cic21(l1_xyz, l1_points)
        l2_xyz, l2_points = self.cic22(l2_xyz, l2_points)

        l3_xyz, l3_points = self.cic31(l2_xyz, l2_points)
        l3_xyz, l3_points = self.cic32(l3_xyz, l3_points)

        l4_xyz, l4_points = self.cic41(l3_xyz, l3_points)
        _, l4_points = self.cic42(l4_xyz, l4_points)

        x = self.conv0(l4_points)
        x_max = F.adaptive_max_pool1d(x, 1)
        x_avg = F.adaptive_avg_pool1d(x, 1)
        x = torch.cat((x_max, x_avg), dim=1).squeeze(-1)
        x = F.relu(self.bn1(self.conv1(x).unsqueeze(-1)), inplace=True).squeeze(-1)
        x = self.dp1(x)
        return x


class ImageEncoder(nn.Module):
    def __init__(self, out_dim: int = 512) -> None:
        super().__init__()
        resnet = models.resnet50(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.fc = nn.Linear(2048, out_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() != 4:
            raise ValueError(f"images must be 4D tensor, got shape={tuple(images.shape)}")
        x = self.backbone(images)
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        return F.relu(self.fc(x), inplace=True)


class PointImageRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.point_encoder = PointEncoder(out_dim=512)
        self.image_encoder = ImageEncoder(out_dim=512)
        self.regressor = nn.Sequential(
            nn.Linear(512 + 512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 1),
        )

    def forward(self, points: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        p = self.point_encoder(points)
        i = self.image_encoder(images)
        x = torch.cat([p, i], dim=1)
        return self.regressor(x).squeeze(1)
