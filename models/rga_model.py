# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torch.autograd import Variable

import torchvision
import numpy as np

from .models_utils.rga_branches import RGA_Branch

WEIGHT_PATH = './models/resnet50-19c8e357.pth'

# ===================
#   Initialization 
# ===================

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

            
def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        init.zeros_(m.bias.data)

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power
    
    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1./self.power)
        out = x.div(norm)
        return out

# ===============
#    RGA Model 
# ===============

class ResNet50_RGA_Model(nn.Module):
    '''
    Backbone: ResNet-50 + RGA modules.
    '''
    def __init__(self, pretrained=True, num_feat=2048, height=256, width=128, 
        dropout=0, num_classes=0, last_stride=1, branch_name='rgasc', scale=8, d_scale=8,
        model_path=WEIGHT_PATH):
        super(ResNet50_RGA_Model, self).__init__()
        self.pretrained = pretrained
        self.num_feat = num_feat
        self.dropout = dropout
        self.num_classes = num_classes
        self.branch_name = branch_name
        print ('Num of features: {}.'.format(self.num_feat))
        
        if 'rgasc' in branch_name:
            spa_on=True 
            cha_on=True
        elif 'rgas' in branch_name:
            spa_on=True
            cha_on=False
        elif 'rgac' in branch_name:
            spa_on=False
            cha_on=True
        else:
            raise NameError
        
        self.backbone = RGA_Branch(pretrained=pretrained, last_stride=last_stride, 
                        spa_on=spa_on, cha_on=cha_on, height=height, width=width,
                        s_ratio=scale, c_ratio=scale, d_ratio=d_scale, model_path=model_path)

        # self.feat_bn = nn.BatchNorm1d(self.num_feat)
        # self.feat_bn.bias.requires_grad_(False)
        # if self.dropout > 0:
        # 	self.drop = nn.Dropout(self.dropout)
        # self.cls = nn.Linear(self.num_feat, self.num_classes, bias=False)

        # self.feat_bn.apply(weights_init_kaiming)
        # self.cls.apply(weights_init_classifier)

    def forward(self, inputs, training=True):
        im_input = inputs[0]

        feat_ = self.backbone(im_input)
        feat_ = F.avg_pool2d(feat_, feat_.size()[2:]).view(feat_.size(0), -1)
        feat = self.feat_bn(feat_)
        if self.dropout > 0:
            feat = self.drop(feat)
        if training and self.num_classes is not None:
            cls_feat = self.cls(feat)
        
        if training:
            return (feat_, feat, cls_feat)
        else:
            return (feat_, feat)


class FeatureModule(nn.Module):
    def __init__(self, input_dim, low_dim, class_num, dropout=0.5):
        super(FeatureModule, self).__init__()
        self.dropout=dropout
        self.fc = nn.Linear(input_dim, low_dim)
        self.feat_bn = nn.BatchNorm1d(low_dim)
        self.feat_bn.bias.requires_grad_(False)
        if self.dropout > 0:
        	self.drop = nn.Dropout(dropout)
        self.cls = nn.Linear(low_dim, class_num, bias=False)

        self.feat_bn.apply(weights_init_kaiming)

    def forward(self, x):
        # feat_ = F.avg_pool2d(x, x.size()[2:]).view(x.size(0), -1)
        feat = self.feat_bn(self.fc(x))
        if self.dropout > 0:
            feat = self.drop(feat)
        cls_feat = self.cls(feat)
        return feat, cls_feat

class FeatureBlock(nn.Module):
    def __init__(self, input_dim, low_dim):
        super(FeatureBlock, self).__init__()
        feat_block = []
        feat_block += [nn.Linear(input_dim, low_dim)] 
        feat_block += [nn.BatchNorm1d(low_dim)]
        
        feat_block = nn.Sequential(*feat_block)
        feat_block.apply(weights_init_kaiming)
        self.feat_block = feat_block
    def forward(self, x):
        x = self.feat_block(x)
        return x
        
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=0.5, relu=True):
        super(ClassBlock, self).__init__()
        classifier = []       
        if relu:
            classifier += [nn.LeakyReLU(0.1)]
        if dropout:
            classifier += [nn.Dropout(p=dropout)]
        
        classifier += [nn.Linear(input_dim, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.classifier = classifier
    def forward(self, x):
        x = self.classifier(x)
        return x  


class thermal_net_resnet(nn.Module):
    def __init__(self, pretrained=True, num_feat=2048, height=256, width=128, 
        dropout=0, num_classes=0, last_stride=1, branch_name='rgasc', scale=8, d_scale=8,
        model_path=WEIGHT_PATH):
        super(thermal_net_resnet, self).__init__()

        if 'rgasc' in branch_name:
            spa_on=True 
            cha_on=True
        elif 'rgas' in branch_name:
            spa_on=True
            cha_on=False
        elif 'rgac' in branch_name:
            spa_on=False
            cha_on=True
        else:
            raise NameError

        self.thermal = RGA_Branch(pretrained=pretrained, last_stride=last_stride, spa_on=spa_on, cha_on=cha_on, height=height, width=width, s_ratio=scale, c_ratio=scale, d_ratio=d_scale, model_path=model_path)


    def forward(self, x):
        x = self.thermal(x)

        num_part = 6 # number of part
        # pool size
        sx = x.size(2) / num_part
        sx = int(sx)
        kx = x.size(2) - sx * (num_part-1)
        kx = int(kx)
        x = nn.functional.avg_pool2d(x, kernel_size=(kx, x.size(3)), stride=(sx, x.size(3)))
        x = x.view(x.size(0), x.size(1), x.size(2))        
        return x


class visible_net_resnet(nn.Module):
    def __init__(self, pretrained=True, num_feat=2048, height=256, width=128, dropout=0, num_classes=0, last_stride=1, branch_name='rgasc', scale=8, d_scale=8, model_path=WEIGHT_PATH):
        super(visible_net_resnet, self).__init__()

        if 'rgasc' in branch_name:
            spa_on=True 
            cha_on=True
        elif 'rgas' in branch_name:
            spa_on=True
            cha_on=False
        elif 'rgac' in branch_name:
            spa_on=False
            cha_on=True
        else:
            raise NameError

        self.visible = RGA_Branch(pretrained=pretrained, last_stride=last_stride, spa_on=spa_on, cha_on=cha_on, height=height, width=width, s_ratio=scale, c_ratio=scale, d_ratio=d_scale, model_path=model_path)


    def forward(self, x):
        x = self.visible(x)

        num_part = 6 # number of part
        # pool size
        sx = x.size(2) / num_part
        sx = int(sx)
        kx = x.size(2) - sx * (num_part-1)
        kx = int(kx)
        x = nn.functional.avg_pool2d(x, kernel_size=(kx, x.size(3)), stride=(sx, x.size(3)))
        x = x.view(x.size(0), x.size(1), x.size(2))                 
        return x


class embed_net_rga(nn.Module):
    def __init__(self, low_dim, class_num, height, width, pretrained=True, dropout=0.0, branch_name='rgasc'):
        super(embed_net_rga, self).__init__()

        pool_dim = 2048
        self.visible_net = visible_net_resnet(pretrained, pool_dim, height, width, dropout=dropout, num_classes=class_num, branch_name=branch_name)
        self.thermal_net = thermal_net_resnet(pretrained, pool_dim, height, width, dropout=dropout, num_classes=class_num,branch_name=branch_name)

        self.classifier1 = FeatureModule(pool_dim, low_dim, class_num, dropout=dropout)
        self.classifier2 = FeatureModule(pool_dim, low_dim, class_num, dropout=dropout)
        self.classifier3 = FeatureModule(pool_dim, low_dim, class_num, dropout=dropout)
        self.classifier4 = FeatureModule(pool_dim, low_dim, class_num, dropout=dropout)
        self.classifier5 = FeatureModule(pool_dim, low_dim, class_num, dropout=dropout)
        self.classifier6 = FeatureModule(pool_dim, low_dim, class_num, dropout=dropout)

        # self.feature1 = FeatureBlock(pool_dim, low_dim)
        # self.feature2 = FeatureBlock(pool_dim, low_dim)
        # self.feature3 = FeatureBlock(pool_dim, low_dim)
        # self.feature4 = FeatureBlock(pool_dim, low_dim)
        # self.feature5 = FeatureBlock(pool_dim, low_dim)
        # self.feature6 = FeatureBlock(pool_dim, low_dim)
        # self.classifier1 = ClassBlock(low_dim, class_num, dropout=dropout)
        # self.classifier2 = ClassBlock(low_dim, class_num, dropout=dropout)
        # self.classifier3 = ClassBlock(low_dim, class_num, dropout=dropout)
        # self.classifier4 = ClassBlock(low_dim, class_num, dropout=dropout)
        # self.classifier5 = ClassBlock(low_dim, class_num, dropout=dropout)
        # self.classifier6 = ClassBlock(low_dim, class_num, dropout=dropout)

        self.l2norm = Normalize(2)

    def forward(self, x1, x2, modal=0):
        if modal==0:
            x1 = self.visible_net(x1)
            x1 = x1.chunk(6, 2)          # 沿2轴分成6块

            x1_0 = x1[0].contiguous().view(x1[0].size(0), -1)
            x1_1 = x1[1].contiguous().view(x1[1].size(0), -1)
            x1_2 = x1[2].contiguous().view(x1[2].size(0), -1)
            x1_3 = x1[3].contiguous().view(x1[3].size(0), -1)
            x1_4 = x1[4].contiguous().view(x1[4].size(0), -1)
            x1_5 = x1[5].contiguous().view(x1[5].size(0), -1)
            
            x2 = self.thermal_net(x2)
            x2 = x2.chunk(6, 2)
            x2_0 = x2[0].contiguous().view(x2[0].size(0), -1)
            x2_1 = x2[1].contiguous().view(x2[1].size(0), -1)
            x2_2 = x2[2].contiguous().view(x2[2].size(0), -1)
            x2_3 = x2[3].contiguous().view(x2[3].size(0), -1)
            x2_4 = x2[4].contiguous().view(x2[4].size(0), -1)
            x2_5 = x2[5].contiguous().view(x2[5].size(0), -1)
            
            x_0 = torch.cat((x1_0, x2_0), 0)
            x_1 = torch.cat((x1_1, x2_1), 0)
            x_2 = torch.cat((x1_2, x2_2), 0)
            x_3 = torch.cat((x1_3, x2_3), 0)
            x_4 = torch.cat((x1_4, x2_4), 0)
            x_5 = torch.cat((x1_5, x2_5), 0)
        
        elif modal ==1:
            x = self.visible_net(x1)
            x = x.chunk(6,2)
            x_0 = x[0].contiguous().view(x[0].size(0),-1)
            x_1 = x[1].contiguous().view(x[1].size(0), -1)
            x_2 = x[2].contiguous().view(x[2].size(0), -1)
            x_3 = x[3].contiguous().view(x[3].size(0), -1)
            x_4 = x[4].contiguous().view(x[4].size(0), -1)
            x_5 = x[5].contiguous().view(x[5].size(0), -1)
        
        elif modal==2:
            x = self.thermal_net(x2)
            x = x.chunk(6, 2)
            x_0 = x[0].contiguous().view(x[0].size(0), -1)
            x_1 = x[1].contiguous().view(x[1].size(0), -1)
            x_2 = x[2].contiguous().view(x[2].size(0), -1)
            x_3 = x[3].contiguous().view(x[3].size(0), -1)
            x_4 = x[4].contiguous().view(x[4].size(0), -1)
            x_5 = x[5].contiguous().view(x[5].size(0), -1)

        # y_0 = self.feature1(x_0)
        # y_1 = self.feature2(x_1)
        # y_2 = self.feature3(x_2)
        # y_3 = self.feature4(x_3)
        # y_4 = self.feature5(x_4)
        # y_5 = self.feature6(x_5)

        # out_0 = self.classifier1(y_0)
        # out_1 = self.classifier2(y_1)
        # out_2 = self.classifier3(y_2)
        # out_3 = self.classifier4(y_3)
        # out_4 = self.classifier5(y_4)
        # out_5 = self.classifier6(y_5)

        y_0, out_0 = self.classifier1(x_0)
        y_1, out_1 = self.classifier2(x_1)
        y_2, out_2 = self.classifier3(x_2)
        y_3, out_3 = self.classifier4(x_3)
        y_4, out_4 = self.classifier5(x_4)
        y_5, out_5 = self.classifier6(x_5)

        if self.training:
            return (out_0, out_1, out_2, out_3, out_4, out_5), (self.l2norm(y_0), self.l2norm(y_1), self.l2norm(y_2), self.l2norm(y_3), self.l2norm(y_4), self.l2norm(y_5))
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



