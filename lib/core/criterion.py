# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.nn import functional as F


class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_label)

    def forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(input=score, size=(h, w), mode='bilinear')

        loss = self.criterion(score, target)

        return loss


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, thres=0.7, min_kept=100000, weight=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_label,
                                             reduction='none')

    def forward(self, score, target, **kwargs):
        # score: model output logits
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(input=score, size=(h, w), mode='bilinear')

        pred = F.softmax(score, dim=1)  # probs
        pixel_losses = self.criterion(score, target).contiguous().view(-1)  # to vector
        mask = target.contiguous().view(-1) != self.ignore_label

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0  # todo: 忽略置0?

        # gather 将 target 作为 idx；完成 对应 prob 提取
        pred = pred.gather(dim=1, index=tmp_target.unsqueeze(1))
        # sort probs
        pred, ind = pred.contiguous().view(-1)[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)]  # prob 从小到大排序 保留的个数
        threshold = max(min_value, self.thresh)  # < thre 都认为是困难样本

        pixel_losses = pixel_losses[mask][ind]  # loss 有序
        pixel_losses = pixel_losses[pred < threshold]  # mask 得到 < thre 的 loss
        return pixel_losses.mean()
