import glob
# from torch.utils.tensorboard import SummaryWriter
import os, losses
from Module import utils
import sys
import torch.distributed as dist
from torch.utils.data import DataLoader
from Data import Dataset, trans
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from natsort import natsorted
#from ZReg.ZModel import ZReg
from Model import CRPNet

import random
import torch.multiprocessing as mp

def get(rank, world_size):
    
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    

def do():
    world_size = 4  
    mp.spawn(get, args=(world_size,), nprocs=world_size, join=True)

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
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True


same_seeds(24)


class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir + "logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def main():
    batch_size = 1

    train_dir = './Train_dataset/LPBA/train/'
    val_dir = './Train_dataset/LPBA/val/'
    weights = [1, 1]  # loss weights
    lr = 0.0005
    save_dir = 'CRPNet_LPBA_ncc_{}_reg_{}_lr_{}_54r/'.format(*weights, lr)
    if not os.path.exists('Para/' + save_dir):
        os.makedirs('Para/' + save_dir)
    if not os.path.exists('logs/' + save_dir):
        os.makedirs('logs/' + save_dir)
    sys.stdout = Logger('logs/' + save_dir)
    f = open(os.path.join('logs/' + save_dir, 'losses and dice' + ".txt"), "a")

    epoch_start = 0
    max_epoch = 40
    img_size = (160, 192, 160)
    cont_training = False
    # device_ids = [0,1]
    '''
    Initialize model
    '''
    
    model = CRPNet()
    reg_model = utils.register_model(img_size, 'nearest')
    GPU_iden= 0
    
    torch.cuda.set_device(GPU_iden)
    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs!")
    #     model = nn.DataParallel(model)
    #     reg_model = nn.DataParallel(reg_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    reg_model.to(device)
    if cont_training:
        model_dir = 'experiments/' + save_dir
        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch, 0.9), 8)
        best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-1])['state_dict']
        model.load_state_dict(best_model)
        print(model_dir + natsorted(os.listdir(model_dir))[-1])
    else:
        updated_lr = lr
    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)

    # checkpoint = torch.load('experiments/' + save_dir + 'dsc0.823.pth.tar')
    # curr_epoch = checkpoint["epoch"]
    # model.load_state_dict(checkpoint['state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # model.cuda()

    # reg_model.cuda()

    '''
    If continue from previous training
    '''
    # if cont_training:
    #     model_dir = 'experiments/' + save_dir
    #     updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch, 0.9), 8)
    #     best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-1])['state_dict']
    #     model.load_state_dict(best_model)
    #     print(model_dir + natsorted(os.listdir(model_dir))[-1])
    # else:
    #     updated_lr = lr

    '''
    Initialize training
    '''
    train_composed = transforms.Compose([trans.NumpyType((np.float32, np.float32))])

    val_composed = transforms.Compose([trans.Seg_norm_LPBA(),
                                       trans.NumpyType((np.float32, np.int16))])

    # **********************PATH******************************
    a = train_dir + '*.pkl'

    train_set = Dataset.LPBABrainDatasetS2S(glob.glob(a), transforms=train_composed)
    val_set = Dataset.LPBABrainInferDatasetS2S(glob.glob(val_dir + '*.pkl'), transforms=val_composed)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    # *******************DATASET*******************************
    # IXI DATASET
    # a = train_IXI_dir + '*.pkl'
    # train_set = datasets.IXIBrainDatasetS2S(glob.glob(a), atlas_dir, transforms=train_composed)
    # val_set = datasets.IXIBrainInferDatasetS2S(glob.glob(val_IXI_dir + '*.pkl'), atlas_dir, transforms=val_composed)
    # #
    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    # val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    # Abdomen Dataset
    # Abdomen_mr = AbdomenMR_train_dir + '*.pkl'
    # Abdomen_ct = AbdomenCT_train_dir + '*.pkl'
    #
    # Abdomen_mr_v = AbdomenMR_val_dir + '*.pkl'
    # Abdomen_ct_v = AbdomenCT_val_dir + '*.pkl'
    # train_set = datasets.AbdomenMRCTDatasetS2S(glob.glob(Abdomen_mr), glob.glob(Abdomen_ct), transforms=train_composed)
    # val_set = datasets.AbdomenMRICTInferDatasetS2S(glob.glob(Abdomen_mr_v), glob.glob(Abdomen_ct_v), transforms=val_composed)
    #
    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    # val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    # *******************************************


    criterion = losses.NCC_vxm()
    criterions = [criterion]
    criterions += [losses.Grad3d(penalty='l2')]
    best_dsc = 0
    # writer = SummaryWriter(log_dir='logs/'+save_dir)
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts epoch {}'.format(epoch))

        loss_all = utils.AverageMeter()
        idx = 0
        for data in train_loader:
            idx += 1
            # model.train()
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            data = [t.cuda() for t in data]
            move = data[0]
            fix = data[1]
           
            
            output = model(move,fix)

            loss = 0
            loss_vals = []
            #output[0] = output[0][np.newaxis,:]
            for n, loss_function in enumerate(criterions):

                curr_loss = loss_function(output[n], fix) * weights[n]
                loss_vals.append(curr_loss)
                loss += curr_loss
            loss_all.update(loss.item(), fix.numel())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}'.format(idx, len(train_loader), loss.item(),
                                                                                   loss_vals[0].item(),
                                                                                   loss_vals[1].item()))
            if (idx == 10000):
                print('iter 10000 times')

        print('{} Epoch {} loss {:.4f}'.format(save_dir, epoch, loss_all.avg))
        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg), file=f, end=' ')
        '''
        Validation
        '''
        eval_dsc = utils.AverageMeter()
        with torch.no_grad():
            for data in val_loader:
                model.eval()
                data = [t.cuda() for t in data]
                move = data[0]
                fix = data[1]
                move_seg = data[2]
                fix_seg = data[3]
                
                output = model(move, fix )
                def_out = reg_model([move_seg.cuda().float(), output[1].cuda()])
                dsc = utils.dice_val_VOI(def_out.long(), fix_seg.long())
                eval_dsc.update(dsc.item(), move.size(0))
                print(epoch, ':', eval_dsc.avg)
        best_dsc = max(eval_dsc.avg, best_dsc)
        print(eval_dsc.avg, file=f)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_dsc': best_dsc,
            'optimizer': optimizer.state_dict(),
        }, save_dir='experiments/' + save_dir, filename='dsc{:.3f}.pth.tar'.format(eval_dsc.avg))
        #
        # NO ROI VERSION
        # best_dsc = best_dsc + 1
        # print(eval_dsc.avg, file=f)
        # if best_dsc % 4 == 0:
        #     save_checkpoint({
        #         'epoch': epoch + 1,
        #         'state_dict': model.state_dict(),
        #         'best_dsc': best_dsc,
        #         'optimizer': optimizer.state_dict(),
        #     }, save_dir='experiments/' + save_dir, filename='dsc{:.3f}.pth.tar'.format(eval_dsc.avg))

        loss_all.reset()


def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)


def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=3):
    torch.save(state, save_dir + filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))


if __name__ == '__main__':

  
    main()
