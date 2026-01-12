import torch
import torch.nn.functional as F
import numpy as np
import math
from torch import nn
class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor).cuda()

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]#如果一个数组的形状是 (5, 4, 3, 2)，则 shape[2:] 将选择索引为2及其之后的维度，即 (3, 2)，而 shape[:2] 将选择索引为0和1的维度，即 (5, 4)。

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class Grad3d(torch.nn.Module):
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        super(Grad3d, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_pred, y_true):

        #loss for RDN__start__________________________
        # reg_loss = 0.0
        # for i in range(len(y_pred)):
        #     reg_loss = reg_loss + regularize_loss(y_pred[i])
        #
        # grad = reg_loss * 1
        #end_________________________________

        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad




class NCC_vxm(torch.nn.Module):
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        super(NCC_vxm, self).__init__()
        self.win = win

    def forward(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)




def dice_loss_VOI(y_pred, y_true, smooth=1e-5):
    """
    针对特定VOI标签（1-13）计算Dice损失，用于模型优化（可反向传播）
    损失计算逻辑：1 - 平均Dice系数（系数越大，损失越小，符合优化目标）

    Args:
        y_pred: 模型预测输出，shape为 [B, 1, H, W, D]（B=批量，1=通道，H/W/D=空间维度）
                注：若预测是概率图（如sigmoid输出），需确保值在[0,1]；若为logits，需先过sigmoid
        y_true: 真实标签，shape与y_pred一致，值为0（背景）或VOI标签（1-13）
        smooth: 平滑项，避免分母为0（默认1e-5，可根据数据调整）

    Returns:
        dice_loss: 平均Dice损失，shape为 [1]（标量损失，可直接用于backward()）
    """
    # 1. 定义需要计算的VOI标签（与原评估函数一致：1-13）

    # min_val = y_pred.min()
    # max_val = y_pred.max()
    # u = torch.unique(y_pred)
    # print(f"outpu张量的最小值: {min_val.item()}")
    # print(f"张量的最大值: {max_val.item()}")
    # print(f"张量的值: ", u)

    VOI_lbls = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                48, 49, 50, 51, 52, 53, 54]
    #VOI_lbls = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    device = y_pred.device  # 确保张量在同一设备（CPU/GPU）

    # 2. 若预测是logits（未经过sigmoid），先转为概率图（确保值在[0,1]，符合标签分布）
    # 注：若模型输出已过sigmoid，可注释这行
    y_pred = torch.sigmoid(y_pred)

    # 3. 初始化存储每个VOI标签的Dice系数列表
    dice_coeffs = []

    # 4. 遍历每个VOI标签，计算单独的Dice系数
    for lbl in VOI_lbls:
        # 4.1 提取当前标签的预测掩码和真实掩码（批量内所有样本）
        # y_pred_lbl: 预测中属于当前标签的概率（shape [B,1,H,W,D]）
        # y_true_lbl: 真实标签中属于当前标签的二值掩码（0/1，shape [B,1,H,W,D]）
        y_pred_lbl = y_pred * (y_true == lbl).float()  # 仅保留当前标签的预测值
        y_true_lbl = (y_true == lbl).float()  # 真实标签转为二值张量

        # 4.2 计算Dice系数（批量内平均，避免因样本数影响损失）
        # 分子：2 * 预测与真实的交集和（批量内所有元素求和）
        intersection = 2 * torch.sum(y_pred_lbl * y_true_lbl, dim=[1, 2, 3, 4])  # 按样本维度求和，shape [B]
        # 分母：预测和 + 真实和（避免分母为0，加smooth）
        union = torch.sum(y_pred_lbl, dim=[1, 2, 3, 4]) + torch.sum(y_true_lbl, dim=[1, 2, 3, 4])  # shape [B]
        # 单个标签的Dice系数（批量内平均，smooth确保数值稳定）
        dice = (intersection + smooth) / (union + smooth)  # shape [B]
        dice_coeffs.append(dice)  # 收集当前标签的批量Dice系数

    # 5. 计算平均Dice系数（所有VOI标签的批量平均再取均值）
    avg_dice = torch.mean(torch.stack(dice_coeffs, dim=0), dim=[0, 1])  # 先按标签堆叠，再求标签+批量平均

    # 6. 转为Dice损失（1 - 平均Dice系数：系数越大，损失越小，符合优化目标）
    dice_loss = 1 - avg_dice

    return dice_loss










