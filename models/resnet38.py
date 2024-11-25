import torch
import torch.nn as nn
import torch.nn.functional as F

import models.resnet38_base


class Net(models.resnet38_base.Net):
    def __init__(self, args):
        super().__init__()

        self.dropout7 = torch.nn.Dropout2d(0.5)

        self.fc8 = nn.Conv2d(4096, args.num_classes + 1, 1, bias=False)
        self.f8_3 = torch.nn.Conv2d(512, 64, 1, bias=False)
        self.f8_4 = torch.nn.Conv2d(1024, 128, 1, bias=False)
        self.f9 = torch.nn.Conv2d(192 + 3, 192, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.fc8.weight)
        torch.nn.init.kaiming_normal_(self.f8_3.weight)
        torch.nn.init.kaiming_normal_(self.f8_4.weight)
        torch.nn.init.xavier_uniform_(self.f9.weight, gain=4)

        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]
        self.from_scratch_layers = [self.fc8, self.f8_3, self.f8_4, self.f9]

    def forward(self, x):
        d = super().forward_as_dict(x)
        #d, edges,counters = super().forward_as_dict(x)
        #print("2", features[0].shape)
        #print("3", features[1].shape)
        #print("4", features[2].shape)
        
        cam = self.fc8(d['conv6'])
        n, c, h, w = d['conv6'].size()

        f8_3 = self.f8_3(d['conv4'].detach())
        f8_4 = self.f8_4(d['conv5'].detach())
        x_s = F.interpolate(x, (h, w), mode='bilinear', align_corners=True)
        f = torch.cat([x_s, f8_3, f8_4], dim=1)

        x_att = self.PCM(d['conv6'], f)
        cam_att = self.fc8(x_att)

        self.featmap = cam + cam_att

        _, _, h, w = cam.size()
        
        #自己添加的，对类激活图进行一定的处理 self.featmap的维度是3,21,56,56 或者 18,21,40,40
        #backgroundFeature = self.featmap.clone()
        #for i in range(backgroundFeature.shape[0]):
        #    for j in range(backgroundFeature.shape[1]):
        #        backgroundFeature[i,j] = self.our_pooling(backgroundFeature[i,j],5,backgroundFeature.shape[2],backgroundFeature.shape[3])
                                
        #pred1 = F.max_pool2d(self.featmap[:, :-1], kernel_size=(h, w), padding=0) - F.max_pool2d(backgroundFeature[:, : -1], kernel_size=(h, w), padding=0)
        #pred2 = F.avg_pool2d(self.featmap[:, :-1], kernel_size=(h, w), padding=0) - F.avg_pool2d(backgroundFeature[:, : -1], kernel_size=(h, w), padding=0)
        #pred = 0.5*pred1 + 0.5*pred2
        
        #在这里将类激活图转换为分类的预测值
        pred = F.avg_pool2d(self.featmap[:, :-1], kernel_size=(h, w), padding=0)
        
        pred = pred.view(pred.size(0), -1)
        return self.featmap, pred  #, edges, counters # featmap是类激活图，pred是分类的预测结果
        
    #自己添加的
    def our_pooling(self,x, iterm, height, weight):
        for i in range(iterm): #迭代次数
            
            index = torch.argmax(x)
            
            h = index // height
            w = index % weight
            min_h = max(h - 10, 0)
            min_w = max(w - 10, 0)
            max_h = min(h + 10, height)
            max_w = min(w + 10, weight)
            x[min_h:max_h, min_w:max_w] = 0
        
        return x
      
    def PCM(self, cam, f):
        n, c, h, w = f.size()
        cam = F.interpolate(cam, (h, w), mode='bilinear', align_corners=True).view(n, -1, h * w)
        f = self.f9(f)
        f = f.view(n, -1, h * w)
        f = f / (torch.norm(f, dim=1, keepdim=True) + 1e-5)

        aff = F.relu(torch.matmul(f.transpose(1, 2), f), inplace=True)
        aff = aff / (torch.sum(aff, dim=1, keepdim=True) + 1e-5)
        cam_rv = torch.matmul(cam, aff).view(n, -1, h, w)\

        return cam_rv

    def forward_cam(self, x):
        x = super().forward(x)
        cam = self.fc8(x)

        return cam

    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:

                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups
