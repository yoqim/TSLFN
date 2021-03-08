import os
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from models.resnet import resnet50
from .models_utils.rga_branches import RGA_Branch_Simple


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)

            
def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power
    
    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1./self.power)
        out = x.div(norm)
        return out


class FeatureModule(nn.Module):
    def __init__(self, input_dim, low_dim, class_num, dropout=0.5, relu=True):
        super(FeatureModule, self).__init__()
  
        feat_block = []
        feat_block += [nn.Linear(input_dim, low_dim)] 
        bn = nn.BatchNorm1d(low_dim)
        bn.bias.requires_grad_(False)
        feat_block += [bn]
        
        feat_block = nn.Sequential(*feat_block)
        feat_block.apply(weights_init_kaiming)
        self.feat_block = feat_block

        classifier = []       
        if relu:
            classifier += [nn.LeakyReLU(0.1)]
        if dropout:
            classifier += [nn.Dropout(p=dropout)]
        
        classifier += [nn.Linear(low_dim, class_num, bias=False)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)
        self.cls = classifier

    def forward(self, x):
        feat = self.feat_block(x)
        cls_feat = self.cls(feat)
        return feat, cls_feat



class visible_module(nn.Module):
    def __init__(self, share_net=1):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.share_net = share_net

        if self.share_net == 0:
            pass
        else:
            self.visible = nn.ModuleList()
            self.visible.conv1 = model_v.conv1
            self.visible.bn1 = model_v.bn1
            self.visible.relu = model_v.relu
            self.visible.maxpool = model_v.maxpool
            if self.share_net > 1:
                for i in range(1, self.share_net):               
                    setattr(self.visible,'layer'+str(i), getattr(model_v,'layer'+str(i)))

    def forward(self, x):
        if self.share_net == 0:
            return x
        else:
            x = self.visible.conv1(x)
            x = self.visible.bn1(x)
            x = self.visible.relu(x)
            x = self.visible.maxpool(x)

            if self.share_net > 1:
                for i in range(1, self.share_net):
                    x = getattr(self.visible, 'layer'+str(i))(x)
            return x


class thermal_module(nn.Module):
    def __init__(self, share_net=1):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.share_net = share_net
        
        if self.share_net == 0:
            pass
        else:
            self.thermal = nn.ModuleList()
            self.thermal.conv1 = model_t.conv1
            self.thermal.bn1 = model_t.bn1
            self.thermal.relu = model_t.relu
            self.thermal.maxpool = model_t.maxpool
            if self.share_net > 1:
                for i in range(1, self.share_net):               
                    setattr(self.thermal,'layer'+str(i), getattr(model_t,'layer'+str(i)))

    def forward(self, x):
        if self.share_net == 0:
            return x
        else:
            x = self.thermal.conv1(x)
            x = self.thermal.bn1(x)
            x = self.thermal.relu(x)
            x = self.thermal.maxpool(x)

            if self.share_net > 1:
                for i in range(1, self.share_net):           
                    x = getattr(self.thermal, 'layer'+str(i))(x)             
            return x


class base_resnet(nn.Module):
    def __init__(self, share_net=1):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)

        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.share_net = share_net       
        if self.share_net == 0:
            self.base = model_base
        else:
            self.base = nn.ModuleList()

            if self.share_net > 4:
                pass
            else:
                for i in range(self.share_net, 5):
                    setattr(self.base,'layer'+str(i), getattr(model_base,'layer'+str(i)))

    def forward(self, x):
        if self.share_net == 0:
            x = self.base.conv1(x)
            x = self.base.bn1(x)
            x = self.base.relu(x)
            x = self.base.maxpool(x)

            x = self.base.layer1(x)
            x = self.base.layer2(x)
            x = self.base.layer3(x)
            x = self.base.layer4(x)
            return x
        elif self.share_net > 4:
            return x
        else:
            for i in range(self.share_net, 5):
                x = getattr(self.base, 'layer'+str(i))(x)
            return x


class embed_net_shared(nn.Module):
    def __init__(self, low_dim, class_num, height, width, share_net=1, dropout=0.0):
        super(embed_net_shared, self).__init__()

        pool_dim = 2048
        self.thermal_module = thermal_module(share_net=share_net)
        self.visible_module = visible_module(share_net=share_net)
        self.base_resnet = base_resnet(share_net=share_net)

        self.drop = dropout
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier1 = FeatureModule(pool_dim, low_dim, class_num, dropout=dropout)
        self.classifier2 = FeatureModule(pool_dim, low_dim, class_num, dropout=dropout)
        self.classifier3 = FeatureModule(pool_dim, low_dim, class_num, dropout=dropout)
        self.classifier4 = FeatureModule(pool_dim, low_dim, class_num, dropout=dropout)
        self.classifier5 = FeatureModule(pool_dim, low_dim, class_num, dropout=dropout)
        self.classifier6 = FeatureModule(pool_dim, low_dim, class_num, dropout=dropout)

        self.l2norm = Normalize(2)

    def forward(self, x1, x2, modal=0):
        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            x = torch.cat((x1, x2), 0)
            
        elif modal == 1:
            x = self.visible_module(x1)

        elif modal == 2:
            x = self.thermal_module(x2)

        x = self.base_resnet(x)
        x_pool = self.avgpool(x)
        x_pool = x_pool.view(x_pool.size(0), x_pool.size(1))

        num_part = 6 # number of part
        # pool size
        sx = x.size(2) / num_part
        sx = int(sx)
        kx = x.size(2) - sx * (num_part-1)
        kx = int(kx)
        
        x = nn.functional.avg_pool2d(x, kernel_size=(kx, x.size(3)), stride=(sx, x.size(3)))
        x = x.view(x.size(0), x.size(1), x.size(2))  

        x = x.chunk(6,2)
        x_0 = x[0].contiguous().view(x[0].size(0), -1)
        x_1 = x[1].contiguous().view(x[1].size(0), -1)
        x_2 = x[2].contiguous().view(x[2].size(0), -1)
        x_3 = x[3].contiguous().view(x[3].size(0), -1)
        x_4 = x[4].contiguous().view(x[4].size(0), -1)
        x_5 = x[5].contiguous().view(x[5].size(0), -1)

        y_0, out_0 = self.classifier1(x_0)
        y_1, out_1 = self.classifier2(x_1)
        y_2, out_2 = self.classifier3(x_2)
        y_3, out_3 = self.classifier4(x_3)
        y_4, out_4 = self.classifier5(x_4)
        y_5, out_5 = self.classifier6(x_5)

        if self.training:
            return x_pool, (out_0, out_1, out_2, out_3, out_4, out_5), (self.l2norm(y_0), self.l2norm(y_1), self.l2norm(y_2), self.l2norm(y_3), self.l2norm(y_4), self.l2norm(y_5))
        else:
            x_0 = self.l2norm(x_0)
            x_1 = self.l2norm(x_1)
            x_2 = self.l2norm(x_2)
            x_3 = self.l2norm(x_3)
            x_4 = self.l2norm(x_4)
            x_5 = self.l2norm(x_5)
            x = torch.cat((x_0, x_1, x_2, x_3, x_4, x_5), 1)

            y_0 = self.l2norm(y_0)
            y_1 = self.l2norm(y_1)
            y_2 = self.l2norm(y_2)
            y_3 = self.l2norm(y_3)
            y_4 = self.l2norm(y_4)
            y_5 = self.l2norm(y_5)
            y = torch.cat((y_0, y_1, y_2, y_3, y_4, y_5), 1)            #(batch_size, low_dim*6)
            return x, y