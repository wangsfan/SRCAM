import torch
from torch import nn

import torch.nn.functional as F
from utils.imutils import Normalize


class ResBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1, first_dilation=None, dilation=1):
        super(ResBlock, self).__init__()

        self.same_shape = (in_channels == out_channels and stride == 1)

        if first_dilation == None: first_dilation = dilation

        self.bn_branch2a = nn.BatchNorm2d(in_channels)

        self.conv_branch2a = nn.Conv2d(in_channels, mid_channels, 3, stride,
                                       padding=first_dilation, dilation=first_dilation, bias=False)

        self.bn_branch2b1 = nn.BatchNorm2d(mid_channels)

        self.conv_branch2b1 = nn.Conv2d(mid_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False)

        if not self.same_shape:
            self.conv_branch1 = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)

    def forward(self, x, get_x_bn_relu=False):

        branch2 = self.bn_branch2a(x)
        branch2 = F.relu(branch2)

        x_bn_relu = branch2

        if not self.same_shape:
            branch1 = self.conv_branch1(branch2)
        else:
            branch1 = x

        branch2 = self.conv_branch2a(branch2)
        branch2 = self.bn_branch2b1(branch2)
        branch2 = F.relu(branch2)
        branch2 = self.conv_branch2b1(branch2)

        x = branch1 + branch2

        if get_x_bn_relu:
            return x, x_bn_relu

        return x

    def __call__(self, x, get_x_bn_relu=False):
        return self.forward(x, get_x_bn_relu=get_x_bn_relu)

class ResBlock_bot(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, dropout=0.):
        super(ResBlock_bot, self).__init__()

        self.same_shape = (in_channels == out_channels and stride == 1)

        self.bn_branch2a = nn.BatchNorm2d(in_channels)
        self.conv_branch2a = nn.Conv2d(in_channels, out_channels//4, 1, stride, bias=False)

        self.bn_branch2b1 = nn.BatchNorm2d(out_channels//4)
        self.dropout_2b1 = torch.nn.Dropout2d(dropout)
        self.conv_branch2b1 = nn.Conv2d(out_channels//4, out_channels//2, 3, padding=dilation, dilation=dilation, bias=False)

        self.bn_branch2b2 = nn.BatchNorm2d(out_channels//2)
        self.dropout_2b2 = torch.nn.Dropout2d(dropout)
        self.conv_branch2b2 = nn.Conv2d(out_channels//2, out_channels, 1, bias=False)

        if not self.same_shape:
            self.conv_branch1 = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)

    def forward(self, x, get_x_bn_relu=False):

        branch2 = self.bn_branch2a(x)
        branch2 = F.relu(branch2)
        x_bn_relu = branch2

        branch1 = self.conv_branch1(branch2)

        branch2 = self.conv_branch2a(branch2)

        branch2 = self.bn_branch2b1(branch2)
        branch2 = F.relu(branch2)
        branch2 = self.dropout_2b1(branch2)
        branch2 = self.conv_branch2b1(branch2)

        branch2 = self.bn_branch2b2(branch2)
        branch2 = F.relu(branch2)
        branch2 = self.dropout_2b2(branch2)
        branch2 = self.conv_branch2b2(branch2)

        x = branch1 + branch2

        if get_x_bn_relu:
            return x, x_bn_relu

        return x

    def __call__(self, x, get_x_bn_relu=False):
        return self.forward(x, get_x_bn_relu=get_x_bn_relu)

class EPM(nn.Module):
    def __init__(self, in_channel):
        super(EPM, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.edge = nn.Sequential(
            nn.Conv2d(64, 21, 3, padding=1, bias=False),
            nn.BatchNorm2d(21),
            nn.ReLU(),
            nn.Conv2d(21, 1, 1, padding=0, bias=False),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.edge2 = nn.Sequential(
            nn.Conv2d(65, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 120, 3, padding=1, bias=False),
            nn.BatchNorm2d(120),
            nn.ReLU()
        )
        self.edge3 = nn.Sequential(
            nn.Conv2d(128, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        self.counter = nn.Conv2d(128, 1, 1, padding=0, bias=False)
    
    def forward(self, x):
        x = self.conv1(x)
        edge = self.edge(x)
        x2 = self.conv2(x)
        edge2 = self.edge2(torch.cat([x2,edge],dim = 1))
        x3 = self.conv3(x2)
        edge3 = self.edge3(torch.cat([x3,edge2],dim = 1))
        out = torch.cat([x3,edge3],dim = 1)
        counter = self.counter(out)
        return edge, counter, out     # edge是边缘检测的预测结果，counter是轮廓检测的结果， out是传给下一层的特征
        

class Merge(nn.Module):
    def __init__(self, in_channel):
        super(Merge, self).__init__()        
        self.merge = nn.Sequential(
            nn.Conv2d(128+in_channel, in_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )
    def forward(self,x):
        
        return self.merge(x)
        
        
        
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1a = nn.Conv2d(3, 64, 3, padding=1, bias=False)

        self.b2 = ResBlock(64, 128, 128, stride=2)
        self.b2_1 = ResBlock(128, 128, 128)
        self.b2_2 = ResBlock(128, 128, 128)

        self.b3 = ResBlock(128, 256, 256, stride=2)
        self.b3_1 = ResBlock(256, 256, 256)
        self.b3_2 = ResBlock(256, 256, 256)

        self.b4 = ResBlock(256, 512, 512, stride=2)
        self.b4_1 = ResBlock(512, 512, 512)
        self.b4_2 = ResBlock(512, 512, 512)
        self.b4_3 = ResBlock(512, 512, 512)
        self.b4_4 = ResBlock(512, 512, 512)
        self.b4_5 = ResBlock(512, 512, 512)

        self.b5 = ResBlock(512, 512, 1024, stride=1, first_dilation=1, dilation=2)
        self.b5_1 = ResBlock(1024, 512, 1024, dilation=2)
        self.b5_2 = ResBlock(1024, 512, 1024, dilation=2)

        self.b6 = ResBlock_bot(1024, 2048, stride=1, dilation=4, dropout=0.3)

        self.b7 = ResBlock_bot(2048, 4096, dilation=4, dropout=0.5)

        self.bn7 = nn.BatchNorm2d(4096)

        self.not_training = [self.conv1a]

        self.normalize = Normalize()
        
        #自己增加的模块
        #self.epm1 = EPM(64)
        #self.epm2 = EPM(128)
        #self.merge2 = Merge(128)
        #self.epm3 = EPM(256)
        #self.merge3 = Merge(256)
        #self.epm4 = EPM(512)
        #self.merge4 = Merge(512)
        
        return

    def forward(self, x):
        return self.forward_as_dict(x)['conv6']

    def forward_as_dict(self, x):
        #edges = []
        #counters = []
        #print("x0",x.shape)
        
        x = self.conv1a(x)
        #print("x1",x.shape)
        #if x.shape[0] < 10:
        #    edge1, counter1, out1 = self.epm1(x)
        #    edges.append(edge1)
        #    #counters.append(counter2)
        #    #x = self.merge2(torch.cat([x,out2],dim = 1))
        
        
        x = self.b2(x)
        x = self.b2_1(x)
        x = self.b2_2(x)
        #print("x2",x.shape)
        #features.append(x)
        #if x.shape[0] < 10:
        #    edge2, counter2, out2 = self.epm2(x)
        #    edges.append(edge2)
        #    counters.append(counter2)
        #    x = self.merge2(torch.cat([x,out2],dim = 1))
            
            
        x = self.b3(x)
        x = self.b3_1(x)
        x = self.b3_2(x)
        #print("x3",x.shape)
        #features.append(x)
        #if x.shape[0] < 10:
        #    edge3, counter3, out3 = self.epm3(x)
        #    edges.append(edge3)
        #    counters.append(counter3)
        #    x = self.merge3(torch.cat([x,out3],dim = 1))
        
        x = self.b4(x)
        x = self.b4_1(x)
        x = self.b4_2(x)
        x = self.b4_3(x)
        x = self.b4_4(x)
        x = self.b4_5(x)
        #print("x4",x.shape)
        #features.append(x)
        #if x.shape[0] < 10:
        #    _, counter4, out4 = self.epm4(x)
        #    #edges.append(edge4)
        #    counters.append(counter4)
        #    #x = self.merge4(torch.cat([x,out4],dim = 1))
        
        x, conv4 = self.b5(x, get_x_bn_relu=True)
        x = self.b5_1(x)
        x = self.b5_2(x)

        x, conv5 = self.b6(x, get_x_bn_relu=True)

        x = self.b7(x)
        conv6 = F.relu(self.bn7(x))

        return dict({'conv4': conv4, 'conv5': conv5, 'conv6': conv6}) #, edges, counters


    def train(self, mode=True):

        super().train(mode)

        for layer in self.not_training:

            if isinstance(layer, torch.nn.Conv2d):
                layer.weight.requires_grad = False

            elif isinstance(layer, torch.nn.Module):
                for c in layer.children():
                    c.weight.requires_grad = False
                    if c.bias is not None:
                        c.bias.requires_grad = False

        for layer in self.modules():

            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.eval()
                layer.bias.requires_grad = False
                layer.weight.requires_grad = False

        return

def convert_mxnet_to_torch(filename):
    import mxnet

    save_dict = mxnet.nd.load(filename)

    renamed_dict = dict()

    bn_param_mx_pt = {'beta': 'bias', 'gamma': 'weight', 'mean': 'running_mean', 'var': 'running_var'}

    for k, v in save_dict.items():

        v = torch.from_numpy(v.asnumpy())
        toks = k.split('_')

        if 'conv1a' in toks[0]:
            renamed_dict['conv1a.weight'] = v

        elif 'linear1000' in toks[0]:
            pass

        elif 'branch' in toks[1]:

            pt_name = []

            if toks[0][-1] != 'a':
                pt_name.append('b' + toks[0][-3] + '_' + toks[0][-1])
            else:
                pt_name.append('b' + toks[0][-2])

            if 'res' in toks[0]:
                layer_type = 'conv'
                last_name = 'weight'

            else:  # 'bn' in toks[0]:
                layer_type = 'bn'
                last_name = bn_param_mx_pt[toks[-1]]

            pt_name.append(layer_type + '_' + toks[1])

            pt_name.append(last_name)

            torch_name = '.'.join(pt_name)
            renamed_dict[torch_name] = v

        else:
            last_name = bn_param_mx_pt[toks[-1]]
            renamed_dict['bn7.' + last_name] = v

    return renamed_dict
