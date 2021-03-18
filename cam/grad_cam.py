
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

class _CAM(object):
    def __init__(self, model):
        super(_CAM, self).__init__()
        self.model = model
    
    def _register(self):
        self.fmaps = []
        self.grads = []

        def _forward_hook(module, input, output):
            self.fmaps.append(output.detach())

        def _backward_hook(module, grad_in, grad_out):
            self.grads.append(grad_out[0].detach())

        for name, module in self.model.named_modules():
            if name == 'base_resnet.base.layer4.2.conv3':
                self.han1 = module.register_forward_hook(_forward_hook)
                self.han2 = module.register_backward_hook(_backward_hook)

    def _remove(self):
        self.han1.remove()
        self.han2.remove()

    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.output).cuda()
        one_hot.scatter_(1, ids, 1.0)
        return one_hot

    def backward(self, output, ids):
        """
        Class-specific backpropagation
        """
        self.output = output
        one_hot = self._encode_one_hot(ids)
        self.model.zero_grad()
        self.output.backward(gradient=one_hot,retain_graph=True)
    
    def generate(self):
        fmap = self.fmaps[0]
        grad = torch.zeros_like(fmap)

        for g in self.grads:
            grad = grad + g
            print('--' *10)
            print(g[1,1,:5,:5])

            print("=> sum grad, ",torch.sum(g))

        import pdb;pdb.set_trace()

        weights = F.adaptive_avg_pool2d(grad, 1)           # (bs, c, 1, 1)

        s_wei = torch.sum(weights,dim=1).squeeze()
        print("weight sum {}".format(s_wei[:10]))
        import pdb;pdb.set_trace()
        
        # cam_fea = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        
        return weights




class _BaseWrapper(object):
    def __init__(self, model, forward_mode):
        super(_BaseWrapper, self).__init__()
        self.model = model
        self.forward_mode = forward_mode
        self.handlers = []  # a set of hook function handlers

    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.logits[0]).cuda()
        one_hot.scatter_(1, ids, 1.0)
        return one_hot

    def forward(self, image):
        self.image_shape = image.shape[2:]

        _,_,self.logits = self.model(image,image,self.forward_mode)         # [num_imgs, num_classes]

        sorted_logits = []
        for lo in self.logits:
            logits = F.softmax(lo, dim=1)
            tmp = logits.sort(dim=1, descending=True)
            sorted_logits.append(tmp[1])
        
        return sorted_logits              # ordered results

    def backward(self, ids, n_fea):
        """
        Class-specific backpropagation
        """
        one_hot = self._encode_one_hot(ids)
        self.model.zero_grad()
        self.logits[n_fea].backward(gradient=one_hot,retain_graph=True)

    def generate(self):
        raise NotImplementedError

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()


class BackPropagation(_BaseWrapper):
    def forward(self, image):
        self.image = image.requires_grad_()
        return super(BackPropagation, self).forward(self.image)

    def generate(self):
        gradient = self.image.grad.clone()
        self.image.grad.zero_()
        return gradient

class GradCAM(_BaseWrapper):
    def __init__(self, model, forward_mode, target_layer=None):
        super(GradCAM, self).__init__(model,forward_mode)
        self.target_layer = target_layer  # list
        
        self.fmaps = []
        self.grads = []
        
        def forward_hook(module, input, output):
            self.fmaps.append(output.detach())

        def backward_hook(module, grad_in, grad_out):
            self.grads.append(grad_out[0].detach())

        for name, module in self.model.named_modules():
            if self.target_layer is None or name in self.target_layer:
                self.handlers.append(module.register_forward_hook(forward_hook))
                self.handlers.append(module.register_backward_hook(backward_hook))
    

    def generate(self, n_fea):
        fmaps = self.fmaps[0]
        grads = self.grads[n_fea]                           # (bs, c, h, w)
        weights = F.adaptive_avg_pool2d(grads, 1)           # (bs, c, 1, 1)
        
        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)

        B, C, H, W = gcam.shape
        gcam = gcam.view(B, -1)
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= gcam.max(dim=1, keepdim=True)[0]
        gcam = gcam.view(B, C, H, W)

        return gcam


    def generate_add(self):
        fmaps = self.fmaps[0]
        grads = torch.zeros_like(fmaps)
        for g in self.grads:
            grads = grads + g
            # print('--' *10)
            # print(g[1,1,:8,:8])
        
        weights = F.adaptive_avg_pool2d(grads, 1)           # (bs, c, 1, 1)

        # tmp = weights.squeeze()
        # for i in range(10):
        #     ctmp = tmp[i,:]
            # print('--' *10)

            # print(i, ": max: ",torch.max(ctmp))
            # print(i, ": min: ",torch.min(ctmp))
            # print("weight sum {}".format(torch.sum(ctmp)))

        # import pdb;pdb.set_trace()

        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)

        B, C, H, W = gcam.shape
        gcam = gcam.view(B, -1)
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= gcam.max(dim=1, keepdim=True)[0]
        gcam = gcam.view(B, C, H, W)

        return gcam