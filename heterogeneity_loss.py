import numpy as np
from torch import nn, tensor
import torch


class hetero_loss(nn.Module):
    def __init__(self, margin=0.1, dist_type = 'l2'):
        super(hetero_loss, self).__init__()
        self.margin = margin
        self.dist_type = dist_type
        if dist_type == 'l2':
            self.dist = nn.MSELoss(reduction='sum')
        if dist_type == 'cos':
            self.dist = nn.CosineSimilarity(dim=0)
        if dist_type == 'l1':
            self.dist = nn.L1Loss()
    
    def forward(self, feat1, feat2, label1, label2):
        label_num = len(label1.unique())
        feat1 = feat1.chunk(label_num, 0)
        feat2 = feat2.chunk(label_num, 0)

        for i in range(label_num):
            center1 = torch.mean(feat1[i], dim=0)
            center2 = torch.mean(feat2[i], dim=0)

            if self.dist_type == 'l2' or self.dist_type == 'l1':
                if i == 0:
                    dist = max(0, self.dist(center1, center2) - self.margin)
                else:
                    dist += max(0, self.dist(center1, center2) - self.margin)
            elif self.dist_type == 'cos':
                if i == 0:
                    dist = max(0, 1-self.dist(center1, center2) - self.margin)
                else:
                    dist += max(0, 1-self.dist(center1, center2) - self.margin)
        
        return dist


class CenterTripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(CenterTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, feats, labels):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """

        label_uni = labels.unique()
        targets = torch.cat([label_uni,label_uni])
        label_num = len(label_uni)
        feat = feats.chunk(label_num*2, 0)
        center = []
        for i in range(label_num*2):
            center.append(torch.mean(feat[i], dim=0, keepdim=True))
        inputs = torch.cat(center)

        n = inputs.size(0)
        
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss,correct