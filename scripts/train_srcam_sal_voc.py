import sys
import os

sys.path.append(os.getcwd())

import torch
import argparse
import time
import shutil
import my_optim
import torch.optim as optim
import models
from torchvision import ops
import torch.nn.functional as F
from utils import AverageMeter
from utils.LoadData import train_srcam_sal_crop_data_loader
from models import vgg
import importlib
import numpy as np
import random
import torchsnooper
import torch.nn as nn
import copy
device = torch.device("cuda:0")

def get_arguments():
    parser = argparse.ArgumentParser(description='The Pytorch code of L2G')
    parser.add_argument("--img_dir", type=str, default='./data/VOCdevkit/VOC2012/')
    parser.add_argument("--train_list", type=str, default='./data/voc12/train_cls.txt')
    parser.add_argument("--test_list", type=str, default='./data/voc12/val_cls.txt')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--iter_size", type=int, default=1)
    parser.add_argument("--input_size", type=int, default=448)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--dataset", type=str, default='imagenet')
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--decay_points", type=str, default='61')
    parser.add_argument("--epoch", type=int, default=15)
    parser.add_argument("--num_workers", type=int, default=20)
    parser.add_argument("--disp_interval", type=int, default=100)
    parser.add_argument("--snapshot_dir", type=str, default='./runs/exp8/model/')
    parser.add_argument("--resume", type=str, default='False')
    parser.add_argument("--global_counter", type=int, default=0)
    parser.add_argument("--current_epoch", type=int, default=0)
    parser.add_argument("--att_dir", type=str, default='./runs/exp8/')
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--sal_dir", type=str, default="Sal")
    parser.add_argument("--poly_optimizer", action="store_true", default=False)
    parser.add_argument("--load_checkpoint", type=str, default="")
    parser.add_argument("--kd_weights", type=int, default=15)
    parser.add_argument("--bg_thr", type=float, default=0.001)

    return parser.parse_args()

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed & (2**32 - 1))
     random.seed(seed & (2**32 - 1))
     torch.backends.cudnn.deterministic = True

def save_checkpoint(args, state, is_best, filename='checkpoint.pth.tar'):
    savepath = os.path.join(args.snapshot_dir, filename)
    torch.save(state, savepath)
    if is_best:
        shutil.copyfile(savepath, os.path.join(args.snapshot_dir, 'model_best.pth.tar'))

def get_model(args):
    model = vgg.vgg16(pretrained=True, num_classes=args.num_classes)
    model = torch.nn.DataParallel(model).cuda()

    model_local = vgg.vgg16(pretrained=True, num_classes=args.num_classes)
    model_local = torch.nn.DataParallel(model_local).cuda()

    param_groups = model.module.get_parameter_groups()
    param_groups_local = model_local.module.get_parameter_groups()

    optimizer = optim.SGD([
        {'params': param_groups[0], 'lr': args.lr},
        {'params': param_groups_local[0], 'lr': args.lr},
        {'params': param_groups[1], 'lr': 2 * args.lr},
        {'params': param_groups_local[1], 'lr': 2 * args.lr},
        {'params': param_groups[2], 'lr': 10 * args.lr},
        {'params': param_groups_local[2], 'lr': 10 * args.lr},
        {'params': param_groups[3], 'lr': 20 * args.lr},
        {'params': param_groups_local[3], 'lr': 20 * args.lr}], momentum=0.9, weight_decay=args.weight_decay,
        nesterov=True)
    criterion = torch.nn.MSELoss()

    return model, model_local, optimizer, criterion

def get_resnet38_model(args):
    model_name = "models.resnet38"
    # print(model_name)
    model = getattr(importlib.import_module(model_name), 'Net')(args)
    model_local = getattr(importlib.import_module(model_name), 'Net')(args)

    if len(args.load_checkpoint) == 0:
        weights_dict = models.resnet38_base.convert_mxnet_to_torch('./models/ilsvrc-cls_rna-a1_cls1000_ep-0001.params')
        model.load_state_dict(weights_dict, strict=False)
        model = torch.nn.DataParallel(model).cuda()

        model_local.load_state_dict(weights_dict, strict=False)
        model_local = torch.nn.DataParallel(model_local).cuda()
    else:
        weights_dict = torch.load(args.load_checkpoint)
        model = torch.nn.DataParallel(model).cuda()
        model.load_state_dict(weights_dict["state_dict"])

        model_local = torch.nn.DataParallel(model_local).cuda()
        model_local.load_state_dict(weights_dict["local_dict"])

    param_groups = model.module.get_parameter_groups()
    param_groups_local = model_local.module.get_parameter_groups()
    
    #weight_out = WeightOut()
    disentangler = Disentangler(21,3,6)
    #channelDisentangler = ChannelDisentangler(21,3)
    #model_x3=Module_X3()
    #channel_weight = Channel_Weight(40)

    optimizer = optim.SGD([
        {'params': param_groups[0], 'lr': args.lr},
        {'params': param_groups_local[0], 'lr': args.lr},
        {'params': param_groups[1], 'lr': 2 * args.lr},
        {'params': param_groups_local[1], 'lr': 2 * args.lr},
        {'params': param_groups[2], 'lr': 10 * args.lr},
        {'params': param_groups_local[2], 'lr': 10 * args.lr},
        {'params': param_groups[3], 'lr': 20 * args.lr},
        {'params': param_groups_local[3], 'lr': 20 * args.lr},
        #{'params': weight_out.parameters(), 'lr': args.lr}, 
        #{'params': channelDisentangler.parameters(), 'lr': args.lr}, 
        {'params': disentangler.parameters(), 'lr': args.lr}
        #{'params': model_x3.parameters(), 'lr': args.lr}
        #{'params': channel_weight.parameters(), 'lr': args.lr}
        ], 
        momentum=0.9, weight_decay=args.weight_decay,
        nesterov=True)
    if len(args.load_checkpoint) > 0:
        opt_weights_dict = torch.load(args.load_checkpoint)["optimizer"]
        optimizer.load_state_dict(opt_weights_dict)
    #criterion = torch.nn.MSELoss()
    criterion = torch.nn.MSELoss(reduction = 'none')
    
    return model, model_local, optimizer, criterion, disentangler #, channel_weight #, disentangler #, channelDisentangler

class Channel_Weight(nn.Module):
    def __init__(self, size):
        super(Channel_Weight, self).__init__()
        self.size = size
        self.avg = torch.nn.AvgPool2d((self.size,self.size))
        self.max = torch.nn.MaxPool2d((self.size,self.size))
        self.mlp = nn.Sequential(
            nn.Linear(21, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 21),
            nn.LeakyReLU(inplace=True),
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        max_value = torch.max_pool2d(x,[self.size,self.size])
        avg_value = torch.max_pool2d(x,[self.size,self.size])
        max_value = self.mlp(max_value.squeeze())
        avg_value = self.mlp(avg_value.squeeze())
        out = self.sigmoid((max_value + avg_value))
        out = out.view(out.shape[0], out.shape[1], 1, 1)
        
        return out



class Bottleneck(nn.Module):
    def __init__(self,
                 in_channels:int=256,
                 out_channels:int=128,
                 dilation:int=2
                 ):

        super(Bottleneck, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(out_channels,out_channels,kernel_size=1,padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
            
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(out_channels,out_channels,kernel_size=3,
                      padding=dilation,dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
            
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(out_channels,out_channels,kernel_size=1,padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        
        )
        self.pre1=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=1,padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
            
        )
        self.pre2=nn.Sequential(
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
            
        )
        
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        identity=self.pre2(self.pre1(x))
        out=self.conv1(identity)
        out=self.conv2(out)
        out=self.conv3(out)
        out=out+identity
        
        return out



class WeightOut(nn.Module):
    def __init__(self):
        super(WeightOut, self).__init__()
        #权重
        self.weight_out = torch.ones(4 * 6,1,1,1)
        #torch.nn.init.kaiming_normal_(self.weight_out)
        self.weight_out = torch.nn.functional.softmax(self.weight_out,dim = 0)
        self.weight_out = torch.nn.Parameter(self.weight_out)

    def forward(self, loss):
        #print(self.weight_out)
        return torch.mean(torch.mul(loss,F.softmax(self.weight_out, dim=0)))
        

class Disentangler(nn.Module):
    def __init__(self, in_channels, batch, number):
        super(Disentangler, self).__init__()
        self.dis = nn.Sequential(
            nn.Conv2d(in_channels,1,kernel_size=3,padding = 1, bias = False),
            nn.BatchNorm2d(1)
        )
        self.batch = batch
        self.num = number
        
        
    def cos_simi(self,embedded_fg, embedded_bg):
        embedded_fg = F.normalize(embedded_fg, dim=1)
        embedded_bg = F.normalize(embedded_bg, dim=1)
        sim = torch.matmul(embedded_fg, embedded_bg.T) #/embedded_fg.shape[1]
        #print(sim)
        return torch.clamp(sim, min=0.0005, max=0.9995)
    
    def forward(self, x):
        loss = 0
        #print(x.shape[0])
        if x.shape[0] != (self.batch * self.num) :
          return loss
          
        for i in range(self.batch):
            temp = x[i*self.num:(i+1)*self.num,:,:,:] #6,21,H,W
            N,C,H,W = temp.size()
            out = torch.sigmoid(self.dis(temp))  # 6,1,H,W
            out_ = out.reshape(N,1,-1) # N,1,H*W
            
            temp = temp.reshape(N,C,-1).permute(0,2,1) # N,H*W,C
            fg_feats = torch.matmul(out_,temp)/(H * W) # N,1,C
            bg_feats = torch.matmul(1.0 - out_,temp)/(H * W) # N,1,C
            
            #fg_feats = out_         # N,1,H*W
            #bg_feats = 1.0 - out_   # N,1,H*W
            
            
            #前景与前景
            #print("1:",torch.mean(self.cos_simi(fg_feats.squeeze(),fg_feats.squeeze())))
            loss = loss - torch.mean(self.cos_simi(fg_feats.squeeze(),fg_feats.squeeze()))
            
            #背景与背景
            #print("2:",torch.mean(self.cos_simi(bg_feats.squeeze(),bg_feats.squeeze())))
            loss = loss - torch.mean(self.cos_simi(bg_feats.squeeze(),bg_feats.squeeze()))
            
            #背景与前景
            #print("3:",torch.mean(self.cos_simi(bg_feats.squeeze(),fg_feats.squeeze())))
            loss = loss + torch.mean(self.cos_simi(bg_feats.squeeze(),fg_feats.squeeze()))
         
        #print(loss)
        return loss / (5 * self.batch * self.num) 
     
class ChannelDisentangler(nn.Module):
    def __init__(self, in_channels, batch):
        super(ChannelDisentangler, self).__init__()
        self.dis = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,kernel_size=3,padding = 1, bias = False),
            nn.BatchNorm2d(in_channels)
        )
        self.batch = batch
        self.in_channels = in_channels
        
        
    def cos_simi(self,embedded):
        embedded = F.normalize(embedded, dim=2)
        N,H,W = embedded.size()
        sim = torch.matmul(embedded,embedded.permute(0,2,1))/(H * W)
        e = 1.0 - torch.eye(H,W)
        e = e.cuda(0)
        sim = sim * e
        
        return torch.clamp(sim, min=0.0005, max=0.9995)
    
    def forward(self, x):
        loss = 0
        
        if x.shape[0] != self.batch :
          return loss
          
        N,C,H,W = x.size()
        out = torch.sigmoid(self.dis(x))
        out_ = out.reshape(N,C,-1)
        x = x.reshape(N,C,-1).permute(0,2,1)
        feature = torch.matmul(out_,x)/(H * W)
        
        #各个通道之间的相似度
        loss = loss + torch.mean(self.cos_simi(feature)) 
          
        
        return loss * 5
        
import random
def getMask(times, batch, nums, img): 
    mask = torch.ones(img.shape[2],img.shape[3])
    for i in range(times):
        x = random.randint(0, nums - 1)
        y = random.randint(0, nums - 1)
        mask[x*batch : (x+1)*batch , y*batch : (y+1)*batch] = 0
    return img * mask
    
    

def train(args):
    batch_time = AverageMeter()
    losses = AverageMeter()

    total_epoch = args.epoch
    global_counter = args.global_counter
    current_epoch = args.current_epoch

    train_loader, val_loader = train_srcam_sal_crop_data_loader(args)
    max_step = total_epoch * len(train_loader)
    args.max_step = max_step
    print('Max step:', max_step)
    
    #model, model_local, optimizer, criterion = get_resnet38_model(args)
    #model, model_local, optimizer, criterion, channel_weight= get_resnet38_model(args)
    model, model_local, optimizer, criterion, disentangler = get_resnet38_model(args)
    #model, model_local, optimizer, criterion, disentangler,channelDisentangler = get_resnet38_model(args)
    # print(model)
    model.train()
    model_local.train()
    end = time.time()
    #channel_weight = channel_weight.cuda(0)
    #weight_out = weight_out.cuda(0)
    disentangler = disentangler.cuda(0)
    #channelDisentangler = channelDisentangler.cuda(0)
    #model_x3 = model_x3.cuda(0)
    #model_x3.train()
    salloss = nn.MSELoss(reduction = 'mean')
    
    
    while current_epoch < total_epoch:
        model.train()
        losses.reset()
        batch_time.reset()
        if not args.poly_optimizer:
            res = my_optim.reduce_lr(args, optimizer, current_epoch)
        steps_per_epoch = len(train_loader)

        flag = 0
        for idx, dat in enumerate(train_loader):
            img, crop_imgs, crop_sals, boxes, label, local_label, img_name ,sal, seg, crop_segs = dat
            #img, crop_imgs, crop_sals, boxes, label, local_label, img_name ,sal,seg, cannyEdge = dat
            crop_sals = crop_sals.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            local_label = local_label.cuda(non_blocking=True)
            sal = sal.cuda(non_blocking=True)
            
            #print("label:",label)
            #print("boxes:",boxes)
            #print("crop_segs:",crop_segs.shape)
            #print("seg:",seg)
            #print("sal:",sal)
            
            
            # dealing with batch size
            bs, bxs, c, h, w = crop_imgs.shape
            
            
            _, _, c_s, h_s, w_s = crop_sals.shape
            crop_imgs = crop_imgs.reshape(bs * bxs, c, h, w)
            crop_sals = crop_sals.reshape(bs * bxs, c_s, h_s, w_s)
            local_label = local_label.reshape(bs * bxs, args.num_classes)
            box_ind = torch.cat([torch.zeros(args.patch_size).fill_(i) for i in range(bs)])
            boxes = boxes.reshape(bs * bxs, 5)
            boxes[:, 0] = box_ind
            
            
            #depth = copy.deepcopy(seg)
            #depth = depth.reshape(depth.shape[0],1,depth.shape[1],depth.shape[2])/ 130.0
            #img = 0.5*img + 0.5*depth
            
#            from thop import profile
#            flops, params = profile(model.to(device), inputs = (img.to(device), ))
#            flopsl, paramsl = profile(model_local.cpu(), crop_imgs.cpu())
#            print("flops:", flops, "params", params)
#            print("flopsl", flopsl, "paramsl", paramsl)
            
            
            
            
            
            
            
            #得到模型的输出
            feat, logits = model(img) 
            feat_local, logits_local = model_local(crop_imgs)
            boxes = boxes.cuda(non_blocking=True).type_as(feat)
            
            
            #seg (3,448,448)为超像素分割结果(3指的是batch)，将seg格式进行转换
            out = np.zeros([seg.shape[0],256,seg.shape[1],seg.shape[2]])  #3,128,448,448
            for ba in range(seg.shape[0]):
                for c in range(256):
                    out[ba,c][seg[ba] == c] = 1
            
            out = torch.from_numpy(out) #3,128,448,448
            pool_seg = nn.MaxPool2d(8,8)
            out = pool_seg(out).cuda(0) # 3,128,56,56

            
            #再搞一下显著性约束
            seg_sal = pool_seg(sal).cuda(0) # 3,1,56,56
##################################################################################            
##            #利用前后背景平均特征筛选超像素特征
##            bg_sal = 1 - sal
##            bg_sal_feat = F.normalize(feat * F.interpolate(bg_sal,size=(feat.shape[2],feat.shape[3])))
##            fg_sal_feat = F.normalize(feat * F.interpolate(sal,size=(feat.shape[2],feat.shape[3])))
##            feat_re = torch.zeros_like(feat)
##            for k in range(feat.shape[1]):
##                t = feat[:,k:k+1,:,:] * out
##                fg_t = fg_sal_feat[:,k:k+1,:,:]
##                bg_t = bg_sal_feat[:,k:k+1,:,:]
##                fg_pro = torch.sum(t*fg_t,dim=[2,3])
##                
##                bg_pro = torch.sum(t*bg_t,dim=[2,3])#[3,256]
##                for kk in range(feat.shape[0]):
##                    kkk = 0
##                    while(kkk < 256):
##                        if fg_pro[kk,kkk]>=bg_pro[kk,kkk]:
##                            feat_re[kk, k, :, :] += t[kk, kkk, :, :]
##                            kkk += 1
##                        else:
##                            break
##           
##            feat = 0.5 * feat + 0.5 * feat_re
####################################################################################
            
            #利用超像素分割的结果，约束类激活图的预测结果 feat  3,21,56,56
            feat_re = torch.zeros_like(feat)
            
            feat_s = torch.softmax(feat, dim = 1)
            error = torch.zeros(1,1,56,56).cuda(0)
            
            for k in range(feat.shape[1]):
                t = feat[:,k:k+1,:,:] * out 
                #t2 = feat[:,k:k+1,:,:] * out  #思路是拿t做排序，拿t2修正结果
                #不能直接求和，对获得结果做排序
                out_sum = torch.sum(out,dim = [2,3])
                t_sum = torch.sum(t,dim = [2,3])
                c = t_sum/out_sum
                
                _,index = torch.sort(c,dim=1,descending=True) #index 3,64
                for kk in range(feat.shape[0]):
                      
                    #按照值来决定加多少
                    kkk = 0
                    
                    while(kkk < 256):
                        
                        #if(c[kk,index[kk,kkk]] > 1):
                        #    continue
                        
                        if c[kk,index[kk,kkk]] > 0.5:
                            feat_re[kk,k,:,:] += t[kk,index[kk,kkk],:,:]
                            #value = torch.sum(t[kk,index[kk,kkk],:,:] * seg_sal[kk,0], dim = [0,1]) / torch.sum(t[kk,index[kk,kkk],:,:], dim = [0,1])
                            #if value > 0.9 or value < 0.1 :      
                            #    feat_re[kk,k,:,:] += t[kk,index[kk,kkk],:,:]
                            #else:
                                #feat_re[kk,k,:,:] += t[kk,index[kk,kkk],:,:]
                            kkk += 1
                        else:
                            break
                            
                            
                    #while(kkk < index.shape[1]) :
                    #    if  c[kk,index[kk,kkk]] < 0.2:      
                    #        error += t[kk,index[kk,kkk],:,:]
                    #        
                    #    kkk += 1

                                     
                
            feat = 0.5 * feat + 0.5 * feat_re
            
            
            
            
            #对局部分支进行超像素约束
            #crop_segs (3,6,320,320)为超像素分割结果(3指的是batch)，将seg格式进行转换
            local_out = np.zeros([crop_segs.shape[0],crop_segs.shape[1],256,crop_segs.shape[2],crop_segs.shape[3]])  #3,6,128,320,320
            for lba in range(crop_segs.shape[0]):
                for lbb in range(crop_segs.shape[1]):
                    for lc in range(256):
                        local_out[lba,lbb,lc][crop_segs[lba,lbb] == lc] = 1
            
            local_out = torch.from_numpy(local_out).view(local_out.shape[0]*local_out.shape[1],local_out.shape[2],local_out.shape[3],local_out.shape[4]) #3,6,128,320,320
            local_pool_seg = nn.MaxPool2d(8,8)
            local_out = local_pool_seg(local_out).cuda(0) #  18,128,40,40

            
##################################################################################################            
#            #对局部利用前后景特征做超像素块筛选
#            bg_crop_sal = 1 - crop_sals
#            bg_crop_sal_feat = F.normalize(feat_local * F.interpolate(bg_crop_sal,size=(feat_local.shape[2],feat_local.shape[3])))
#            fg_crop_sal_feat = F.normalize(feat_local * F.interpolate(crop_sals,size=(feat_local.shape[2],feat_local.shape[3])))
#            local_feat_re = torch.zeros_like(feat_local)
#            for lk in range(feat_local.shape[1]):
#                lt = feat_local[:, lk:lk + 1, :, :] * local_out
#                fg_lt = fg_crop_sal_feat[:, lk:lk + 1, :, :]
#                bg_lt = bg_crop_sal_feat[:, lk:lk + 1, :, :]
#                fg_lpro = torch.sum(lt*fg_lt,dim=[2,3])
#                bg_lpro = torch.sum(lt*bg_lt,dim=[2,3])
#                for lkk in range(feat_local.shape[0]):
#                    lkkk = 0
#                    while (lkkk < 256):
#                        if fg_lpro[lkk, lkkk] >= bg_lpro[lkk, lkkk]:
#                            local_feat_re[lkk, lk, :, :] += lt[lkk, lkkk, :, :]
#                            lkkk += 1
#                        else:
#                            break
#            
#            feat_local = 0.5 * feat_local + 0.5 * local_feat_re
######################################################################################################
            
            
            
            #利用超像素分割的结果，约束类激活图的预测结果 feat  18,21,40,40
            local_feat_re = torch.zeros_like(feat_local)
            
            feat_s = torch.softmax(feat, dim = 1)
            error = torch.zeros(1,1,56,56).cuda(0)
            
            for lk in range(feat_local.shape[1]):
                lt = feat_local[:,lk:lk+1,:,:] * local_out 
                #t2 = feat[:,k:k+1,:,:] * out  #思路是拿t做排序，拿t2修正结果
                #不能直接求和，对获得结果做排序
                local_out_sum = torch.sum(local_out,dim = [2,3])
                local_t_sum = torch.sum(lt,dim = [2,3])
                lc = local_t_sum/local_out_sum
                
                _,local_index = torch.sort(lc,dim=1,descending=True)
                for lkk in range(feat_local.shape[0]):
                      
                    #按照值来决定加多少
                    lkkk = 0
                    while(lkkk < 256):
                        
                        #if(c[kk,index[kk,kkk]] > 1):
                        #    continue
                        
                        if lc[lkk,local_index[lkk,lkkk]] > 0.5:
                            
                            local_feat_re[lkk,lk,:,:] += lt[lkk,local_index[lkk,lkkk],:,:]
                            lkkk += 1
                        else:
                            break
                            
                    #while(kkk < index.shape[1]) :
                    #    if  c[kk,index[kk,kkk]] < 0.2:      
                    #        error += t[kk,index[kk,kkk],:,:]
                    #        
                    #    kkk += 1
                      
                
            feat_local = 0.5 * feat_local + 0.5 * local_feat_re
            
            
            
            
            #根据超像素的结果来确定权重
            feat_local_weight = torch.zeros(feat_local.shape[0],feat_local.shape[1],feat_local.shape[2],feat_local.shape[3])
            for wi in range(local_out.shape[0]):
                nums = 0
                for wj in range(local_out.shape[1]):
                    if torch.sum(local_out[wi,wj,:,:],dim = [0,1]) > 10:
                        nums = nums + 1
                    
                feat_local_weight[wi] = nums
               
            feat_local_weight = (feat_local_weight/torch.max(feat_local_weight)).cuda(0)
            
                
            
            # visualize
            feat_local_label = feat_local.clone().detach()  # 4, 20, 224, 224

            # normalize
            ba = logits_local.shape[0]
            feat_local_label[feat_local_label < 0] = 0
            ll_max = torch.max(torch.max(feat_local_label, dim=3)[0], dim=2)[0]
            feat_local_label = feat_local_label / (ll_max.unsqueeze(2).unsqueeze(3) + 1e-8)
            for i in range(bs):
                ind = torch.nonzero(label[i] == 0)
                feat_local_label[i * bxs:(i + 1) * bxs, ind] = 0

            # keep max value among all classes
            n, c, h, w = feat_local_label.shape
            feat_local_label_c = feat_local_label.permute(1, 0, 2, 3).reshape(c, -1)
            ind_f = torch.argsort(-feat_local_label_c, axis=0)
            pos = torch.eye(c, device=device)[ind_f[0]].transpose(0, 1).type_as(feat_local_label_c)
            feat_local_label_c = pos * feat_local_label_c
            feat_local_label = feat_local_label_c.reshape(c, n, h, w).permute(1, 0, 2, 3)

            # match the sal label    hyper-parameter
            feat_local_label_bool = (feat_local_label > args.bg_thr).type_as(feat_local_label)
            crop_sals = F.interpolate(crop_sals, (h, w)).type_as(feat_local_label)
            feat_local_label[:, :-1, :, :] = feat_local_label_bool[:, :-1, :, :] * crop_sals.repeat(1, 20, 1, 1)
            feat_local_label[:, -1, :, :] = feat_local_label_bool[:, -1, :, :] * ((1 - crop_sals).squeeze(1))
            
            #根据feat_local_label得到了一组权重
            #feat_local_weight = channel_weight(feat_local_label)
            
            # roi align
            feat_aligned = ops.roi_align(feat, boxes, (h, w), 1 / 8.0)
            feat_aligned = F.softmax(feat_aligned, dim=1)
            #loss_kd = criterion(feat_aligned, feat_local_label) * args.kd_weights
            loss_kd = torch.mean(criterion(feat_aligned, feat_local_label) * feat_local_weight) * args.kd_weights
            
            
            
            
            #利用来修正后的类激活图计算分类结果
            #logits_local = F.avg_pool2d(feat_local[:, :-1], kernel_size=(feat_local.shape[2], feat_local.shape[3]), padding=0)
            #logits_local = logits_local.view(logits_local.size(0), -1)
            
#            #超像素池化
            local_out = local_out.to(torch.float32)
            feat_seg_pool = torch.matmul(feat_local.reshape(feat_local.shape[0],feat_local.shape[1],feat_local.shape[2] * feat_local.shape[3]),local_out.reshape(local_out.shape[0],local_out.shape[1],local_out.shape[2]*local_out.shape[3]).permute(0,2,1))
            logits_spx_pool = feat_seg_pool.mean(dim=2)
            
#            logits_spx_pool = F.avg_pool2d(feat_seg_pool[:, :-1], kernel_size=(feat.shape[2], feat.shape[3]), padding=0)
            logits_spx_pool = logits_spx_pool.view(logits_spx_pool.size(0), -1)
            
            # cls loss
            if len(logits.shape) == 1:
                logits = logits.reshape(label.shape)

            loss_cls_global = F.multilabel_soft_margin_loss(logits, label) / args.iter_size
            loss_cls_local = F.multilabel_soft_margin_loss(logits_spx_pool[:, :-1], local_label) / args.iter_size
            
            #自己增加的一些loss
            #轮廓和边缘loss
            #max_pool = nn.MaxPool2d(kernel_size=5,stride=1,padding=2)
            #counter = max_pool(sal + max_pool(-sal)).cuda(0)
            #pool1 = nn.MaxPool2d(2,2)
            #pool2 = nn.MaxPool2d(4,4)
            #loss_counter = counterloss(counters[0],pool1(counter)) + counterloss(counters[1],pool2(counter)) 
            #up = torch.nn.Upsample(scale_factor=2)
            #loss_edge = edgeloss(up(edges[0]) + up(up(edges[1])),cannyEdge) 
            
            
            loss_sim = disentangler(feat_local)  #局部图之间的相似性
            #loss_channel_sim = channelDisentangler(feat)  #通道间的相似性
            
            #feat_x3, pred_x3 , edge_x3= model_x3(features[0])
            #loss_cls_x3 = F.multilabel_soft_margin_loss(pred_x3, label) / args.iter_size
            #loss_multual_x3 = torch.mean(torch.cosine_similarity(feat,feat_x3,dim=1))
            #max_pool = nn.MaxPool2d(kernel_size=5,stride=1,padding=2)
            #pool = nn.MaxPool2d(8,8)
            #loss_edge_x3 = mse_x3(edge_x3, pool(sal + max_pool(-sal)).cuda(0)) + mse_x3(torch.sum(feat_x3[:,:-1],dim = 1, keepdim= True),pool(sal).cuda(0))

            #自己增加的loss  feat  3,21,56,56  这个loss是衡量全局分支存在的类别相加后与显著性检测结果的差异
            feat_fore = copy.copy(feat)
            feat_fore = F.softmax(feat_fore, dim=1)
            feat_fore = copy.copy(feat_fore[:,:-1,:,:])  # 3,20,56,56
            cls_label_fore = label.reshape([label.shape[0],label.shape[1],1,1]) #3,20,1,1
            out = torch.sum(feat_fore * cls_label_fore, dim=1, keepdim=True) #3,1,56,56
            pool = nn.MaxPool2d(8,8)
            small_sal = pool(sal) # 3,1,56,56
            loss_sal = salloss(out, small_sal.cuda())
            
            #feat_fore = F.softmax(feat, dim=1)
            #feat_fore = feat_fore[:,:-1,:,:]  # 3,20,56,56
            #cls_label_fore = label.reshape([label.shape[0],label.shape[1],1,1]) #3,20,1,1
            #out = torch.sum(feat_fore * cls_label_fore, dim=1, keepdim=True) #3,1,56,56
            #pool = nn.MaxPool2d(8,8)
            #small_sal = pool(sal) # 3,1,56,56
            #loss_sal = salloss(out, small_sal.cuda())
            
            
            
            loss_val = loss_kd + loss_cls_local  + 0.5* loss_sim  + loss_sal #+loss_cls_global 
            loss_val.backward()
            
            
            

            
            
            flag += 1
            if flag == args.iter_size:
                optimizer.step()
                optimizer.zero_grad()
                flag = 0

            losses.update(loss_val.data.item(), img.size()[0])
            batch_time.update(time.time() - end)
            end = time.time()

            global_counter += 1
            if global_counter % 1000 == 0:
                losses.reset()

            if global_counter % args.disp_interval == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'LR: {:.5f}\t'
                      'Loss_kd {:.4f}\t'
                      'Loss_cls_global {:.4f}\t'
                      'Loss_cls_local {:.4f}\t'
#                      'Loss_sim {:.4f}\t'
                      #'Loss_channel_sim {:.4f}\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    current_epoch, global_counter % len(train_loader), len(train_loader),
                    optimizer.param_groups[0]['lr'], loss_kd, loss_cls_global,
                    loss_cls_local,  loss=losses))
                    
                #print("loss_cls_x3:",loss_cls_x3)
                #print("loss_edge_x3:",loss_edge_x3)
                #print("loss_multual_x3:",loss_multual_x3)
                #print("loss_counter:",loss_counter)
                

        if current_epoch == args.epoch - 1:
            save_checkpoint(args,
                            {
                                'epoch': current_epoch,
                                'global_counter': global_counter,
                                'state_dict': model.state_dict(),
                                'local_dict': model_local.state_dict(),
                                'optimizer': optimizer.state_dict()
                            }, is_best=False,
                            filename='256spx_spd-%s_epoch_%d.pth' % (args.dataset, current_epoch))
                            
        if current_epoch > args.epoch / 2:
            save_checkpoint(args,
                            {
                                'epoch': current_epoch,
                                'global_counter': global_counter,
                                'state_dict': model.state_dict(),
                                'local_dict': model_local.state_dict(),
                                'optimizer': optimizer.state_dict()
                            }, is_best=False,
                            filename='256spx_spd-%s_epoch_%d_temp.pth' % (args.dataset, current_epoch))
        current_epoch += 1


if __name__ == '__main__':
    setup_seed(15742315057023588855)
    args = get_arguments()
    print('Running parameters:\n', args)
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    train(args)
