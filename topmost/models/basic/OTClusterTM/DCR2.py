import torch
from torch import nn


class DCR2(nn.Module):
    def __init__(self, weight_loss_DCR, cluster_center):
        super().__init__()

        self.weight_loss_DCR = weight_loss_DCR
        self.cluster_center = cluster_center
        self.transp = None

    def KL(self, a, b):
        a = a.clamp(min=1e-6, max=1-1e-6)
        b = b.clamp(min=1e-6, max=1-1e-6)
        return (a * (a.log() - b.log() - 1) + b).sum()

    def forward(self, theta_prj, bert_emb):
        if self.weight_loss_DCR <= 1e-6:
            return 0.
        distance_theta = torch.cdist(theta_prj, self.cluster_center)
        prob_theta = distance_theta / distance_theta.sum(axis=1, keepdim=True)

        distance_bert = torch.cdist(bert_emb, self.cluster_center)
        prob_bert = distance_bert / distance_bert.sum(axis=1, keepdim=True)

        loss = (self.KL(prob_theta, prob_bert) + self.KL(prob_bert, prob_theta)) / 2.
        return loss * self.weight_loss_DCR