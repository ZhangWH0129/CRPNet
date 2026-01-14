import math
import numpy as np
import torch.nn.functional as F
import torch, sys
from torch import nn
import pystrum.pynd.ndutils as nd
import re
from medpy import metric
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = []
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals.append(val)
        self.std = np.std(self.vals)


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
        shape = flow.shape[2:]

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

class register_model(nn.Module):
    def __init__(self, img_size=(64, 256, 256), mode='bilinear'):
        super(register_model, self).__init__()
        self.spatial_trans = SpatialTransformer(img_size, mode)

    def forward(self, x):
        img = x[0].cuda()
        flow = x[1].cuda()
        out = self.spatial_trans(img, flow)
        return out






def metric_val_VOI(y_pred, y_true):
    VOI_lbls = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                48, 49, 50, 51, 52, 53, 54]
    # OASIS
    # VOI_lbls = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    #             12, 13, 14]
    # IXI
    # VOI_lbls = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    #             12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    #             24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
    #             36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
    pred = y_pred.detach().cpu().numpy()[0, 0, ...]
    true = y_true.detach().cpu().numpy()[0, 0, ...]
    Lists_dsc = np.zeros((len(VOI_lbls), 1))
    Lists_hd = np.zeros((len(VOI_lbls), 1))
    Lists_asd = np.zeros((len(VOI_lbls), 1))
    idx = 0
    for i in VOI_lbls:
        pred_i = pred == i
        true_i = true == i

        dsc = metric.binary.dc(pred_i, true_i)
        hd = metric.binary.hd95(pred_i, true_i)
        asd = metric.binary.asd(pred_i, true_i)
        Lists_dsc[idx] = dsc
        Lists_hd[idx] = hd
        Lists_asd[idx] = asd
        idx += 1
    return np.mean(Lists_dsc), np.mean(Lists_hd), np.mean(Lists_asd)


def dice_raw_VOI(y_pred, y_true):

    VOI_lbls = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                48, 49, 50, 51, 52, 53, 54]
    # OASIS
    # VOI_lbls = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    #             12, 13, 14]
    # IXI
    # VOI_lbls = [1]
    # VOI_lbls = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    #             12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    #             24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
    #             36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]
    pred = y_pred.detach().cpu().numpy()[0, 0, ...]
    true = y_true.detach().cpu().numpy()[0, 0, ...]
    num_classes = len(VOI_lbls)
    Lists_dsc = np.zeros(num_classes) 
    # Lists_hd = np.zeros(num_classes) 
    # Lists_asd = np.zeros(num_classes) 
    idx = 0
    for i in VOI_lbls:
        # if i+1 in [1,2,3,4,6,8,10,13,14,15,20,23,26,29,30,31,32,33,34,35,36,37,39,40,41,42,43,44,45,46,47]:
        #     continue
        pred_i = pred == i
        true_i = true == i
        # if not pred_i.any() or not true_i.any():
        #     print(f"Skipping class {i} due to empty mask")
        #     continue
        # if i in [10,11,18,26,27,37]:
        #     continue

        dsc = metric.binary.dc(pred_i, true_i)
        # hd = metric.binary.hd95(pred_i, true_i)
        # asd = metric.binary.asd(pred_i, true_i)
        Lists_dsc[idx] = dsc
        # Lists_hd[idx] = hd
        # Lists_asd[idx] = asd
        idx += 1
    return np.mean(Lists_dsc)



def dice_val_VOI(y_pred, y_true):
    # LPBA40
    VOI_lbls = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
              12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
              24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
               36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
               48, 49, 50, 51, 52, 53, 54]
    #OASIS
    # VOI_lbls = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    #            12, 13, 14]
    #Abdomen
    # VOI_lbls = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    #            12, 13]
    # IXI
    # VOI_lbls = [ 5, 7, 9, 11,
    #             12,  16, 17, 18, 19, 21, 22,
    #             24, 25,  27, 28,  38, ]

    pred = y_pred.detach().cpu().numpy()[0, 0, ...]
    true = y_true.detach().cpu().numpy()[0, 0, ...]
    #print("Output range:", y_pred.min(), y_pred.max())

    DSCs = np.zeros((len(VOI_lbls), 1))
    #print(f"Pred unique: {np.unique(pred)}, True unique: {np.unique(true)}")
    idx = 0
    for i in VOI_lbls:
        pred_i = pred == i
        true_i = true == i
        dsc = metric.binary.dc(pred_i, true_i)
        DSCs[idx] = dsc
        # intersection = pred_i * true_i
        # intersection = np.sum(intersection)
        # union = np.sum(pred_i) + np.sum(true_i)
        # dsc = (2.*intersection) / (union + 1e-5)
        # DSCs[idx] =dsc
        idx += 1
    return np.mean(DSCs)



def jacobian_determinant_vxm(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.
    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims
    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    disp = disp.transpose(1, 2, 3, 0)
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:  # must be 2

        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]
