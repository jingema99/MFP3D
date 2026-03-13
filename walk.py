import torch
import torch.nn as nn
import torch.nn.functional as F


def batched_index_select(input, dim, index):
    views = [input.shape[0]] + [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim, index)


def gumbel_softmax(logits, dim, temperature=1):
    """
    Straight-through gumbel-softmax without random gumbel samples.
    """
    y = F.softmax(logits / temperature, dim=dim)

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    y_hard = (y_hard - y).detach() + y
    return y_hard


class Walk(nn.Module):
    """
    Walk in the cloud.
    """

    def __init__(self, in_channel, k, curve_num, curve_length):
        super().__init__()
        self.curve_num = curve_num
        self.curve_length = curve_length
        self.k = k

        self.agent_mlp = nn.Sequential(
            nn.Conv2d(in_channel * 2, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
        )
        self.momentum_mlp = nn.Sequential(
            nn.Conv1d(in_channel * 2, 2, kernel_size=1, bias=False),
            nn.BatchNorm1d(2),
        )

    def crossover_suppression(self, cur, neighbor, bn, n, k):
        neighbor = neighbor.detach()
        cur = cur.unsqueeze(-1).detach()
        dot = torch.bmm(cur.transpose(1, 2), neighbor)
        norm1 = torch.norm(cur, dim=1, keepdim=True)
        norm2 = torch.norm(neighbor, dim=1, keepdim=True)
        divider = torch.clamp(norm1 * norm2, min=1e-8)
        ans = torch.div(dot, divider).squeeze()
        ans = 1.0 + ans
        ans = torch.clamp(ans, 0.0, 1.0)
        return ans.detach()

    def forward(self, xyz, x, adj, cur):
        bn, c, tot_points = x.size()

        xyz = xyz.transpose(1, 2).contiguous()
        x = x.transpose(1, 2).contiguous()

        flatten_x = x.view(bn * tot_points, -1)
        batch_offset = torch.arange(0, bn, device=x.device).detach() * tot_points

        tmp_adj = (adj + batch_offset.view(-1, 1, 1)).view(adj.size(0) * adj.size(1), -1)
        flatten_cur = (cur + batch_offset.view(-1, 1, 1)).view(-1)

        curves = []

        for step in range(self.curve_length):
            if step == 0:
                starting_points = flatten_x[flatten_cur, :].contiguous()
                pre_feature = starting_points.view(bn, self.curve_num, -1, 1).transpose(1, 2)
            else:
                cat_feature = torch.cat((cur_feature.squeeze(), pre_feature.squeeze()), dim=1)
                att_feature = F.softmax(self.momentum_mlp(cat_feature), dim=1).view(
                    bn, 1, self.curve_num, 2
                )
                cat_feature = torch.cat((cur_feature, pre_feature), dim=-1)
                pre_feature = torch.sum(cat_feature * att_feature, dim=-1, keepdim=True)
                pre_feature_cos = pre_feature.transpose(1, 2).contiguous().view(
                    bn * self.curve_num, -1
                )

            pick_idx = tmp_adj[flatten_cur]
            pick_values = flatten_x[pick_idx.view(-1), :]

            pick_values_cos = pick_values.view(bn * self.curve_num, self.k, c)
            pick_values = pick_values_cos.view(bn, self.curve_num, self.k, c)
            pick_values_cos = pick_values_cos.transpose(1, 2).contiguous()
            pick_values = pick_values.permute(0, 3, 1, 2)

            pre_feature_expand = pre_feature.expand_as(pick_values)
            pre_feature_expand = torch.cat((pick_values, pre_feature_expand), dim=1)
            pre_feature_expand = self.agent_mlp(pre_feature_expand)

            if step != 0:
                d = self.crossover_suppression(
                    cur_feature_cos - pre_feature_cos,
                    pick_values_cos - cur_feature_cos.unsqueeze(-1),
                    bn,
                    self.curve_num,
                    self.k,
                )
                d = d.view(bn, self.curve_num, self.k).unsqueeze(1)
                pre_feature_expand = torch.mul(pre_feature_expand, d)

            pre_feature_expand = gumbel_softmax(pre_feature_expand, -1)
            cur_feature = torch.sum(pick_values * pre_feature_expand, dim=-1, keepdim=True)
            cur_feature_cos = cur_feature.transpose(1, 2).contiguous().view(
                bn * self.curve_num, c
            )
            cur = torch.argmax(pre_feature_expand, dim=-1).view(-1, 1)
            flatten_cur = batched_index_select(pick_idx, 1, cur).squeeze()
            curves.append(cur_feature)

        return torch.cat(curves, dim=-1)
