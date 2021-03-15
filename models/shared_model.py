import os
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from models.resnet import resnet50
from .models_utils.rga_modules import RGA_Module
# from cam.grad_cam import BackPropagation, GradCAM


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


class FeatureBlock(nn.Module):
    def __init__(self, input_dim, low_dim):
        super(FeatureBlock, self).__init__()
        feat_block = []
        feat_block += [nn.Linear(input_dim, low_dim)] 
        feat_bn = nn.BatchNorm1d(low_dim)
        feat_bn.bias.requires_grad_(False)

        feat_block += [feat_bn]
        
        feat_block = nn.Sequential(*feat_block)
        feat_block.apply(weights_init_kaiming)
        self.feat_block = feat_block

    def forward(self, x):
        x = self.feat_block(x)
        return x
        

class visible_module(nn.Module):
    def __init__(self, share_net=1):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)

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
    def __init__(self, share_net=1, height=256, width=128, scale=8, d_scale=8, branch_name='rgas'):
        super(base_resnet, self).__init__()

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

        self.rga = RGA_Module(2048, (height//16)*(width//16), use_spatial=spa_on, use_channel=cha_on,
                                cha_ratio=scale, spa_ratio=scale, down_ratio=d_scale)


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

        elif self.share_net > 4:
            pass
        else:
            for i in range(self.share_net, 5):
                x = getattr(self.base, 'layer'+str(i))(x)
        
        att_x = self.rga(x)
        x = x+att_x

        return x


class embed_net_shared(nn.Module):
    def __init__(self, low_dim, class_num, height, width, npart, share_net=1, branch_name='rgas'):
        super(embed_net_shared, self).__init__()
        
        self.npart = npart
        pool_dim = 2048
        self.thermal_module = thermal_module(share_net=share_net)
        self.visible_module = visible_module(share_net=share_net)
        self.base_resnet = base_resnet(share_net=share_net, height=height, width=width, scale=8, d_scale=8,branch_name=branch_name)

        for i in range(self.npart):
            exec('self.feature{} = FeatureBlock(pool_dim, low_dim)'.format(i+1))

        self.classifier = nn.Linear(low_dim, class_num, bias=False)
        # self.classifier = nn.Linear(low_dim*self.npart, class_num, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.l2norm = Normalize(2)

    def _gen_feat_part(self, feat):
        '''
        gen partial feature maps
        ori : (b,c,h,w) -> p_fea: (b,c,npart)

        '''
        sx = feat.size(2) / self.npart
        sx = int(sx)
        kx = feat.size(2) - sx * (self.npart-1)
        kx = int(kx)
        
        x = F.avg_pool2d(feat, kernel_size=(kx, feat.size(3)), stride=(sx, feat.size(3)))
        x = x.view(x.size(0), x.size(1), x.size(2))  
        
        return x
    
    def _chunk_feat(self, feat):
        feat = feat.chunk(self.npart, 2)

        chunk_feat = []
        for i in range(self.npart):
            exec('chunk_feat.append(feat[{}].contiguous().view(feat[{}].size(0), -1))'.format(i,i))
        return chunk_feat

    def _chunk_fmout(self, fmout):
        fmout = fmout.chunk(2,0)

        chunk_feat = []
        for i in range(2):
            exec('chunk_feat.append(fmout[{}].contiguous().view(fmout[{}].size(0), -1))'.format(i,i))
        return chunk_feat

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

        if modal==0:
            [x1,x2] = x.chunk(2, 0)
            x1 = x1.contiguous()
            x2 = x2.contiguous()

            x1 = self._gen_feat_part(x1)
            x1 = self._chunk_feat(x1)

            x2 = self._gen_feat_part(x2)
            x2 = self._chunk_feat(x2)

            x_0 = torch.cat((x1[0], x2[0]), 0)
            x_1 = torch.cat((x1[1], x2[1]), 0)
            x_2 = torch.cat((x1[2], x2[2]), 0)
            x_3 = torch.cat((x1[3], x2[3]), 0)
            # x_4 = torch.cat((x1[4], x2[4]), 0)
            # x_5 = torch.cat((x1[5], x2[5]), 0)

        elif modal == 1:
            x1 = self._gen_feat_part(x)
            x1 = self._chunk_feat(x1)
            
            x_0 = x1[0]
            x_1 = x1[1]
            x_2 = x1[2]
            x_3 = x1[3]
            # x_4 = x1[4]
            # x_5 = x1[5]
        
        elif modal == 2:
            x2 = self._gen_feat_part(x)
            x2 = self._chunk_feat(x2)
            
            x_0 = x2[0]
            x_1 = x2[1]
            x_2 = x2[2]
            x_3 = x2[3]
            # x_4 = x2[4]
            # x_5 = x2[5]

        y_0 = self.feature1(x_0)                    # head: (vis 32,the 32)
        y_1 = self.feature2(x_1)
        y_2 = self.feature3(x_2)
        y_3 = self.feature4(x_3)
        # y_4 = self.feature5(x_4)
        # y_5 = self.feature6(x_5)

        if modal == 0:
            y0 = self._chunk_fmout(y_0)
            y1 = self._chunk_fmout(y_1)
            y2 = self._chunk_fmout(y_2)
            y3 = self._chunk_fmout(y_3)
            # y4 = self._chunk_fmout(y_4)
            # y5 = self._chunk_fmout(y_5)

            # y_vis = torch.cat((y0[0], y1[0], y2[0], y3[0], y4[0], y5[0]), 1)
            # y_the = torch.cat((y0[1], y1[1], y2[1], y3[1], y4[1], y5[1]), 1)
            y_vis = torch.cat((y0[0], y1[0], y2[0], y3[0]), 1)
            y_the = torch.cat((y0[1], y1[1], y2[1], y3[1]), 1)
            
            y = torch.cat((y_vis,y_the), 0)                     # whole feat: (2 x batch_size, low_dim*npart)
        else:
            y = torch.cat((y_0,y_1,y_2,y_3), 1)
        
        out0 = self.classifier(y_0)
        out1 = self.classifier(y_1)
        out2 = self.classifier(y_2)
        out3 = self.classifier(y_3)

        if self.training:
            # return y, out, (self.l2norm(y_0), self.l2norm(y_1), self.l2norm(y_2), self.l2norm(y_3), self.l2norm(y_4), self.l2norm(y_5))
            return y, (out0,out1,out2,out3), (self.l2norm(y_0), self.l2norm(y_1), self.l2norm(y_2), self.l2norm(y_3))
        else:
            x_0 = self.l2norm(x_0)
            x_1 = self.l2norm(x_1)
            x_2 = self.l2norm(x_2)
            x_3 = self.l2norm(x_3)
            # x_4 = self.l2norm(x_4)
            # x_5 = self.l2norm(x_5)
            # x = torch.cat((x_0, x_1, x_2, x_3, x_4, x_5), 1)
            x = torch.cat((x_0, x_1, x_2, x_3), 1)

            y_0 = self.l2norm(y_0)
            y_1 = self.l2norm(y_1)
            y_2 = self.l2norm(y_2)
            y_3 = self.l2norm(y_3)
            # y_4 = self.l2norm(y_4)
            # y_5 = self.l2norm(y_5)

            # y = torch.cat((y_0, y_1, y_2, y_3, y_4, y_5), 1)            #(batch_size, low_dim*6)
            y = torch.cat((y_0,y_1,y_2,y_3), 1)

            return x, y, out0


class embed_net_mulcla(nn.Module):
    def __init__(self, low_dim, class_num, height, width, npart, share_net=1, branch_name='rgas'):
        super(embed_net_mulcla, self).__init__()
        
        self.npart = npart
        pool_dim = 2048
        self.thermal_module = thermal_module(share_net=share_net)
        self.visible_module = visible_module(share_net=share_net)
        self.base_resnet = base_resnet(share_net=share_net, height=height, width=width, scale=8, d_scale=8,branch_name=branch_name)

        self.feature1 = FeatureBlock(pool_dim, low_dim)
        self.feature2 = FeatureBlock(pool_dim, low_dim)
        self.feature3 = FeatureBlock(pool_dim, low_dim)
        self.feature4 = FeatureBlock(pool_dim, low_dim)
        
        self.classifier1 = nn.Linear(low_dim, class_num, bias=False)
        self.classifier2 = nn.Linear(low_dim, class_num, bias=False)
        self.classifier3 = nn.Linear(low_dim, class_num, bias=False)
        self.classifier4 = nn.Linear(low_dim, class_num, bias=False)
        
        for i in range(self.npart):
            exec('self.classifier{}.apply(weights_init_classifier)'.format(i+1))

        self.l2norm = Normalize(2)

    def _gen_feat_part(self, feat):
        '''
        gen partial feature maps
        ori : (b,c,h,w) -> p_fea: (b,c,npart)

        '''
        sx = feat.size(2) / self.npart
        sx = int(sx)
        kx = feat.size(2) - sx * (self.npart-1)
        kx = int(kx)
        
        x = F.avg_pool2d(feat, kernel_size=(kx, feat.size(3)), stride=(sx, feat.size(3)))
        x = x.view(x.size(0), x.size(1), x.size(2))  
        
        return x
    
    def _chunk_feat(self, feat):
        '''
        chunk feat (single modality) -> partial feat
        '''
        feat = feat.chunk(self.npart, 2)

        chunk_feat = []
        for i in range(self.npart):
            exec('chunk_feat.append(feat[{}].contiguous().view(feat[{}].size(0), -1))'.format(i,i))
        return chunk_feat

    def _chunk_fmout(self, fmout):
        '''
        chunk -> 
        |vis_fea|
        |the_fea|
        '''
        fmout = fmout.chunk(2,0)

        chunk_feat = []
        for i in range(2):
            exec('chunk_feat.append(fmout[{}].contiguous().view(fmout[{}].size(0), -1))'.format(i,i))
        return chunk_feat

    def _cam_att(self, pooled_feats, labels):
        '''
        : enhance pooled_feat with cam att
        feats: (2*b,c,h,w) ... Feature maps after backbone
        labels: (2*b,1)    ... Actual id labels
        '''
        # 1. Channel-wise self-attention
        bs, c = pooled_feats.shape[:2]
        feats = pooled_feats
        relu_feats = F.relu(feats)          # why relu?

        feats = feats.unsqueeze(1)
        relu_feats = relu_feats.unsqueeze(2)

        print("feats.shape, ",feats.shape)
        print("relu_feats.shape, ",relu_feats.shape)
        import pdb;pdb.set_trace()

        # bs x c x c x h x w
        att_feats = feats*relu_feats      # check if each group has the same feat maps?
        print("att_feats.shape, ",att_feats.shape)
        print(att_feats[0,0,1,:5,:5])
        print(att_feats[0,0,2,:5,:5])
        print(att_feats[0,0,3,:5,:5])
        import pdb;pdb.set_trace()

        # (bs x c x c) x h x w ???
        att_feats = att_feats.view(-1, *att_feats.shape[2:])            # check shape
        print("att_feats.shape, ",att_feats.shape)
        import pdb;pdb.set_trace()

        # 2. Get score of all the feature groups
        att_feats = self._gen_feat_part(att_feats)          # (bs x c x c) x npart
        att_feats = self._chunk_feat(att_feats)             # (bs x c x c) x npart

        _, out_0 = self.classifier(att_feats[0])
        _, out_1 = self.classifier(att_feats[1])
        _, out_2 = self.classifier(att_feats[2])
        _, out_3 = self.classifier(att_feats[3])
 
        out_0 = F.softmax(out_0,dim=1)
        out_1 = F.softmax(out_1,dim=1)
        out_2 = F.softmax(out_2,dim=1)
        out_3 = F.softmax(out_3,dim=1)

        probs = torch.cat((out_0, out_1, out_2, out_3), 0)
        probs = torch.mean(probs,0)
        
        probs = probs.view(bs, c, -1).permute(0, 2, 1)  # bs, num_cls, c
        
        print("check shape diff")
        w0 = probs[:, labels]
        wei = probs[torch.arange(0, labels.shape[0]), labels]
        print(w0.shape)
        print(wei.shape)
        import pdb;pdb.set_trace()

        weight = probs[torch.arange(0, labels.shape[0]), labels].unsqueeze(2).unsqueeze(2)      # check shape

        # 3. Weighted sum
        cam = (pooled_feats * weight).sum(dim=1)

        return cam

    def forward(self, x1, x2, modal=0, labels=None):
        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            x = torch.cat((x1, x2), 0)

        elif modal == 1:
            x = self.visible_module(x1)

        elif modal == 2:
            x = self.thermal_module(x2)
        
        x = self.base_resnet(x)

        if modal==0:
            [x1,x2] = x.chunk(2, 0)
            x1 = x1.contiguous()
            x2 = x2.contiguous()

            x1 = self._gen_feat_part(x1)
            x1 = self._chunk_feat(x1)

            x2 = self._gen_feat_part(x2)
            x2 = self._chunk_feat(x2)

            x_0 = torch.cat((x1[0], x2[0]), 0)
            x_1 = torch.cat((x1[1], x2[1]), 0)
            x_2 = torch.cat((x1[2], x2[2]), 0)
            x_3 = torch.cat((x1[3], x2[3]), 0)
            # x_4 = torch.cat((x1[4], x2[4]), 0)
            # x_5 = torch.cat((x1[5], x2[5]), 0)

        elif modal == 1:
            x1 = self._gen_feat_part(x)
            x1 = self._chunk_feat(x1)

            x_0 = x1[0]
            x_1 = x1[1]
            x_2 = x1[2]
            x_3 = x1[3]
            # x_4 = x1[4]
            # x_5 = x1[5]
        
        elif modal == 2:
            x2 = self._gen_feat_part(x)
            x2 = self._chunk_feat(x2)
            
            x_0 = x2[0]
            x_1 = x2[1]
            x_2 = x2[2]
            x_3 = x2[3]
            # x_4 = x2[4]
            # x_5 = x2[5]

        y_0 = self.feature1(x_0)                    # head: (vis 32, the 32)
        y_1 = self.feature2(x_1)
        y_2 = self.feature3(x_2)
        y_3 = self.feature4(x_3)
        # y_4 = self.feature5(x_4)
        # y_5 = self.feature6(x_5)

        if modal == 0:
            y0 = self._chunk_fmout(y_0)
            y1 = self._chunk_fmout(y_1)
            y2 = self._chunk_fmout(y_2)
            y3 = self._chunk_fmout(y_3)
            # y4 = self._chunk_fmout(y_4)
            # y5 = self._chunk_fmout(y_5)

            # y_vis = torch.cat((y0[0], y1[0], y2[0], y3[0], y4[0], y5[0]), 1)
            # y_the = torch.cat((y0[1], y1[1], y2[1], y3[1], y4[1], y5[1]), 1)
            y_vis = torch.cat((y0[0], y1[0], y2[0], y3[0]), 1)
            y_the = torch.cat((y0[1], y1[1], y2[1], y3[1]), 1)
            
            y = torch.cat((y_vis,y_the), 0)                     # whole feat: (2 x batch_size, low_dim*npart)
        else:
            y = torch.cat((y_0,y_1,y_2,y_3), 1)
        
        out0 = self.classifier1(y_0)
        out1 = self.classifier2(y_1)
        out2 = self.classifier3(y_2)
        out3 = self.classifier4(y_3)

        if self.training:
            # return y, out, (self.l2norm(y_0), self.l2norm(y_1), self.l2norm(y_2), self.l2norm(y_3), self.l2norm(y_4), self.l2norm(y_5))
            return y, (out0,out1,out2,out3), (self.l2norm(y_0), self.l2norm(y_1), self.l2norm(y_2), self.l2norm(y_3))
        else:
            x_0 = self.l2norm(x_0)
            x_1 = self.l2norm(x_1)
            x_2 = self.l2norm(x_2)
            x_3 = self.l2norm(x_3)
            # x_4 = self.l2norm(x_4)
            # x_5 = self.l2norm(x_5)
            # x = torch.cat((x_0, x_1, x_2, x_3, x_4, x_5), 1)
            x = torch.cat((x_0, x_1, x_2, x_3), 1)

            y_0 = self.l2norm(y_0)
            y_1 = self.l2norm(y_1)
            y_2 = self.l2norm(y_2)
            y_3 = self.l2norm(y_3)
            # y_4 = self.l2norm(y_4)
            # y_5 = self.l2norm(y_5)

            # y = torch.cat((y_0, y_1, y_2, y_3, y_4, y_5), 1)            #(batch_size, low_dim*npart)
            y = torch.cat((y_0,y_1,y_2,y_3), 1)

            return x, y