import torch
import torch.nn as nn
import torch.nn.functional as F

from walk import Walk


def knn(x, k):
    k = k + 1
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def square_distance(src, dst):
    b, n, _ = src.shape
    _, m, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(b, n, 1)
    dist += torch.sum(dst**2, -1).view(b, 1, m)
    return dist


def index_points(points, idx):
    device = points.device
    b = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(b, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    )
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    device = xyz.device
    b, n, _ = xyz.shape
    centroids = torch.zeros(b, npoint, dtype=torch.long, device=device)
    distance = torch.ones(b, n, device=device) * 1e10
    farthest = torch.zeros(b, dtype=torch.long, device=device)
    batch_indices = torch.arange(b, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(b, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    device = xyz.device
    b, n, _ = xyz.shape
    _, s, _ = new_xyz.shape
    group_idx = torch.arange(n, dtype=torch.long, device=device).view(1, 1, n).repeat([b, s, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius**2] = n
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(b, s, 1).repeat([1, 1, nsample])
    mask = group_idx == n
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points):
    new_xyz = index_points(xyz, farthest_point_sample(xyz, npoint))
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    new_points = index_points(points, idx)
    return new_xyz, new_points


class LPFA(nn.Module):
    def __init__(self, in_channel, out_channel, k, mlp_num=2, initial=False):
        super().__init__()
        self.k = k
        self.initial = initial

        if not initial:
            self.xyz2feature = nn.Sequential(
                nn.Conv2d(9, in_channel, kernel_size=1, bias=False),
                nn.BatchNorm2d(in_channel),
            )

        mlp = []
        for _ in range(mlp_num):
            mlp.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, out_channel, 1, bias=False),
                    nn.BatchNorm2d(out_channel),
                    nn.LeakyReLU(0.2),
                )
            )
            in_channel = out_channel
        self.mlp = nn.Sequential(*mlp)

    def forward(self, x, xyz, idx=None):
        x = self.group_feature(x, xyz, idx)
        x = self.mlp(x)
        if self.initial:
            x = x.max(dim=-1, keepdim=False)[0]
        else:
            x = x.mean(dim=-1, keepdim=False)
        return x

    def group_feature(self, x, xyz, idx):
        batch_size, num_dims, num_points = x.size()
        if idx is None:
            idx = knn(xyz, k=self.k)[:, :, : self.k]

        idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)

        xyz = xyz.transpose(2, 1).contiguous()
        point_feature = xyz.view(batch_size * num_points, -1)[idx, :]
        point_feature = point_feature.view(batch_size, num_points, self.k, -1)
        points = xyz.view(batch_size, num_points, 1, 3).expand(-1, -1, self.k, -1)
        point_feature = torch.cat((points, point_feature, point_feature - points), dim=3)
        point_feature = point_feature.permute(0, 3, 1, 2).contiguous()

        if self.initial:
            return point_feature

        x = x.transpose(2, 1).contiguous()
        feature = x.view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, self.k, num_dims)
        x = x.view(batch_size, num_points, 1, num_dims)
        feature = feature - x
        feature = feature.permute(0, 3, 1, 2).contiguous()
        point_feature = self.xyz2feature(point_feature)
        feature = F.leaky_relu(feature + point_feature, 0.2)
        return feature


class CIC(nn.Module):
    def __init__(
        self,
        npoint,
        radius,
        k,
        in_channels,
        output_channels,
        bottleneck_ratio=2,
        mlp_num=2,
        curve_config=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.output_channels = output_channels
        self.radius = radius
        self.k = k
        self.npoint = npoint

        planes = in_channels // bottleneck_ratio

        self.use_curve = curve_config is not None
        if self.use_curve:
            self.curveaggregation = CurveAggregation(planes)
            self.curvegrouping = CurveGrouping(planes, k, curve_config[0], curve_config[1])

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, planes, kernel_size=1, bias=False),
            nn.BatchNorm1d(planes),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(planes, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(output_channels),
        )

        if in_channels != output_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, output_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(output_channels),
            )
        else:
            self.shortcut = None

        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.maxpool = MaskedMaxPool(npoint, radius, k)
        self.lpfa = LPFA(planes, planes, k, mlp_num=mlp_num, initial=False)

    def forward(self, xyz, x):
        if xyz.size(-1) != self.npoint:
            xyz, x = self.maxpool(xyz.transpose(1, 2).contiguous(), x)
            xyz = xyz.transpose(1, 2)

        shortcut = x
        x = self.conv1(x)
        idx = knn(xyz, self.k)

        if self.use_curve:
            curves = self.curvegrouping(x, xyz, idx[:, :, 1:])
            x = self.curveaggregation(x, curves)

        x = self.lpfa(x, xyz, idx=idx[:, :, : self.k])
        x = self.conv2(x)

        if self.shortcut is not None:
            shortcut = self.shortcut(shortcut)

        x = self.relu(x + shortcut)
        return xyz, x


class CurveAggregation(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        mid_feature = in_channel // 2
        self.conva = nn.Conv1d(in_channel, mid_feature, kernel_size=1, bias=False)
        self.convb = nn.Conv1d(in_channel, mid_feature, kernel_size=1, bias=False)
        self.convc = nn.Conv1d(in_channel, mid_feature, kernel_size=1, bias=False)
        self.convn = nn.Conv1d(mid_feature, mid_feature, kernel_size=1, bias=False)
        self.convl = nn.Conv1d(mid_feature, mid_feature, kernel_size=1, bias=False)
        self.convd = nn.Sequential(
            nn.Conv1d(mid_feature * 2, in_channel, kernel_size=1, bias=False),
            nn.BatchNorm1d(in_channel),
        )
        self.line_conv_att = nn.Conv2d(in_channel, 1, kernel_size=1, bias=False)

    def forward(self, x, curves):
        curves_att = self.line_conv_att(curves)
        curver_inter = torch.sum(curves * F.softmax(curves_att, dim=-1), dim=-1)
        curves_intra = torch.sum(curves * F.softmax(curves_att, dim=-2), dim=-2)

        curver_inter = self.conva(curver_inter)
        curves_intra = self.convb(curves_intra)

        x_logits = self.convc(x).transpose(1, 2).contiguous()
        x_inter = F.softmax(torch.bmm(x_logits, curver_inter), dim=-1)
        x_intra = F.softmax(torch.bmm(x_logits, curves_intra), dim=-1)

        curver_inter = self.convn(curver_inter).transpose(1, 2).contiguous()
        curves_intra = self.convl(curves_intra).transpose(1, 2).contiguous()

        x_inter = torch.bmm(x_inter, curver_inter)
        x_intra = torch.bmm(x_intra, curves_intra)
        curve_features = torch.cat((x_inter, x_intra), dim=-1).transpose(1, 2).contiguous()
        x = x + self.convd(curve_features)
        return F.leaky_relu(x, negative_slope=0.2)


class CurveGrouping(nn.Module):
    def __init__(self, in_channel, k, curve_num, curve_length):
        super().__init__()
        self.curve_num = curve_num
        self.att = nn.Conv1d(in_channel, 1, kernel_size=1, bias=False)
        self.walk = Walk(in_channel, k, curve_num, curve_length)

    def forward(self, x, xyz, idx):
        x_att = torch.sigmoid(self.att(x))
        x = x * x_att
        _, start_index = torch.topk(x_att, self.curve_num, dim=2, sorted=False)
        start_index = start_index.squeeze().unsqueeze(2)
        curves = self.walk(xyz, x, idx, start_index)
        return curves


class MaskedMaxPool(nn.Module):
    def __init__(self, npoint, radius, k):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.k = k

    def forward(self, xyz, features):
        sub_xyz, neighborhood_features = sample_and_group(
            self.npoint, self.radius, self.k, xyz, features.transpose(1, 2)
        )
        neighborhood_features = neighborhood_features.permute(0, 3, 1, 2).contiguous()
        sub_features = F.max_pool2d(
            neighborhood_features, kernel_size=[1, neighborhood_features.shape[3]]
        )
        sub_features = torch.squeeze(sub_features, -1)
        return sub_xyz, sub_features
