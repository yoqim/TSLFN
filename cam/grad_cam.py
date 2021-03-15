
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

class _BaseWrapper(object):
    def __init__(self, model, forward_mode):
        super(_BaseWrapper, self).__init__()
        self.device = 'cuda'
        self.model = model
        self.forward_mode = forward_mode
        self.handlers = []  # a set of hook function handlers

    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.logits).to(self.device)
        one_hot.scatter_(1, ids, 1.0)
        return one_hot

    def forward(self, image):
        self.image_shape = image.shape[2:]

        _,_,self.logits = self.model(image,image,self.forward_mode)  # [1,num_classes]
        self.probs = F.softmax(self.logits, dim=1)
        return self.probs.sort(dim=1, descending=True)               # ordered results

    def backward(self, ids):
        """
        Class-specific backpropagation
        """
        one_hot = self._encode_one_hot(ids)
        self.model.zero_grad()
        self.logits.backward(gradient=one_hot, retain_graph=True)

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
    

    def generate(self, target_layer):
        fmaps = self.fmaps[0]
        grads = self.grads[0]
        weights = F.adaptive_avg_pool2d(grads, 1)

        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)

        B, C, H, W = gcam.shape
        gcam = gcam.view(B, -1)
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= gcam.max(dim=1, keepdim=True)[0]
        gcam = gcam.view(B, C, H, W)

        return gcam