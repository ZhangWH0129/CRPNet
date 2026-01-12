import glob
import os
from  Module import  utils
from torch.utils.data import DataLoader
from Data import Dataset, trans
import numpy as np
import SimpleITK as sitk
import torch
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion, distance_transform_edt
from torchvision import transforms
from scipy.ndimage import _ni_support
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from natsort import natsorted

from Model import CRPNet

import random
import torch.nn as nn


def count_elements_in_3d_array(array):
    # 创建一个字典来存储每个数值的出现次数
    element_count = {}

    # 遍历三维数组的每个元素
    for layer in array:
        for row in layer:
            for element in row:
                # 更新字典中的计数
                element_count[element] = element_count.get(element, 0) + 1

                # 对字典的键进行排序
    sorted_keys = sorted(element_count.keys())

    # 按照排序后的键顺序输出结果
    for num in sorted_keys:
        print(f"数值 {num} 出现了 {element_count[num]} 次")

def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True

same_seeds(24)

def hausdorff_distance(result, reference, voxelspacing=None, connectivity=1, percentage=None):
    hd1 = __surface_distances(result, reference, voxelspacing, connectivity)
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity)
    if percentage is None:
        distance = max(hd1.max(), hd2.max())
    elif isinstance(percentage, (int, float)):
        distance = np.percentile(np.hstack((hd1, hd2)), percentage)
    else:
        raise ValueError
    return distance

def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    result = np.atleast_1d(result.astype('bool'))
    reference = np.atleast_1d(reference.astype('bool'))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()

    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)

    # test for emptiness
    if 0 == np.count_nonzero(result):
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if 0 == np.count_nonzero(reference):
        raise RuntimeError('The second supplied array does not contain any binary object.')

        # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)

    # compute average surface distance
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]

    return sds


def csv_writter(line, name):
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')


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

def main():

    val_dir = './Train_dataset/LPBA/test/'
    model_idx = -1
    model_folder = 'CRPNet_LPBA_ncc_1_reg_1_lr_0.0005_54r/'
    model_dir = 'Para/' + model_folder


    img_size = (160, 192, 160 )

    model = CRPNet()

    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx],map_location='cuda:0')['state_dict']

    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)
    model.cuda()
    reg_model = utils.register_model(img_size, 'nearest')
    reg_model.cuda()
    test_composed = transforms.Compose([trans.Seg_norm_LPBA(),
                                        trans.NumpyType((np.float32, np.int16)),
                                        ])


    test_set = Dataset.LPBABrainInferDatasetS2S(glob.glob(val_dir + '*.pkl'), transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

    eval_dsc_def = AverageMeter()
    eval_dsc = utils.AverageMeter()
    eval_hd = utils.AverageMeter()
    eval_asd = utils.AverageMeter()
    eval_dsc_raw = AverageMeter()
    eval_det = AverageMeter()
    eval_Hos_det = AverageMeter()
    times = []
    with torch.no_grad():
        stdy_idx = 0
        for data in test_loader:
            model.eval()
            data = [t.cuda() for t in data]
            move = data[0]
            fix = data[1]
            move_seg = data[2]
            fix_seg = data[3]


            move_def, flow = model(move, fix)




#******************************************************************************
            def_out = reg_model([move_seg.cuda().float(), flow.cuda()])
            tar = move.detach().cpu().numpy()[0, 0, :, :, :]

            dsc, hd, asd = utils.metric_val_VOI(def_out.long(), fix_seg.long())
            dsc_raw = utils.dice_val_VOI(fix_seg.long(), move_seg.long())
            eval_dsc.update(dsc.item(), fix.size(0))
            eval_hd.update(hd.item(), fix.size(0))
            eval_asd.update(asd.item(), fix.size(0))

            #大于0的雅可比行列式
            jac_det = utils.jacobian_determinant_vxm(flow.detach().cpu().numpy()[0, :, :, :, :])
            eval_det.update(np.sum(jac_det <= 0) / np.prod(tar.shape), fix.size(0))
            stdy_idx += 1
            print('pair:{} -- dsc {}'.format(stdy_idx, eval_dsc.avg))
            print('hd {} '.format(eval_hd.avg))
            print('asd {}'.format(eval_asd.avg))
            print(' jacobian: {}'.format(eval_det.avg))

            #Dice系数
            #dsc_trans = utils.dice_val_VOI(def_out.long(), fix_seg.long())
            #dsc_raw = utils.dice_val_VOI(fix_seg.long(), move_seg.long())  #dice_val_VOI用于计算Dice
            # #
            # def_out_3d = torch.squeeze(def_out)
            # def_out_3d = def_out_3d.cpu().numpy()
            # fix_seg_3d = torch.squeeze(fix_seg)
            # fix_seg_3d = fix_seg_3d.cpu().numpy()
            # Hos = hausdorff_distance(def_out_3d, fix_seg_3d)



            # print('pair:{} Trans dsc: {:.4f}, Raw dsc: {:.4f}'.format(stdy_idx,dsc_trans.item(),dsc_raw.item()))
            # print('pair:{} hos: {}'.format(stdy_idx, Hos))
            # print('pair:{} jacobian: {}'.format(stdy_idx, eval_det.avg))
            #
            # eval_Hos_det.update(Hos.item(), fix.size(0))
            # eval_dsc_def.update(dsc_trans.item(), fix.size(0))
            eval_dsc_raw.update(dsc_raw.item(), fix.size(0))
            # stdy_idx += 1
        # print(' Deformed DSC: {:.3f} +- {:.3f}, Affine DSC: {:.3f} +- {:.3f}'.format(
        #                                                                             eval_dsc_def.avg,
        #                                                                             eval_dsc_def.std,
        #                                                                             eval_dsc_raw.avg,
        #                                                                             eval_dsc_raw.std))
        # print('deformed jacobian det: {}, std: {}'.format(eval_det.avg, eval_det.std))
        # print('Hausduff 95 det: {}, std: {}'.format(eval_Hos_det.avg, eval_Hos_det.std))

        print(times)
        print('Sum---dsc{:.4f}dsc_std{:.4f}'.format(eval_dsc.avg, eval_dsc.std))
        print('Sum--dsc{:.4f}dsc_std{:.4f}'.format( eval_dsc_raw.avg, eval_dsc_raw.std))
        print('hd{:.4f}hd_std{:.4f}'.format(eval_hd.avg, eval_hd.std))
        print('asd{:.4f}asd_std{:.4f}'.format(eval_asd.avg, eval_asd.std))
        print('deformed jacobian det: {}, std: {}'.format(eval_det.avg, eval_det.std))



if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()