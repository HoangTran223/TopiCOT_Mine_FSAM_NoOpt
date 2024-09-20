import torch
from torch import nn


class DCR3(nn.Module):
    def __init__(self, weight_loss_DCR, sinkhorn_alpha, num_groups, num_data,
                 OT_max_iter=5000, stopThr=.5e-2):
        super().__init__()

        self.otlr = 1e-2
        self.num_groups = num_groups
        self.num_data = num_data
        self.sinkhorn_alpha = sinkhorn_alpha
        self.OT_max_iter = OT_max_iter
        self.weight_loss_DCR = weight_loss_DCR
        self.stopThr = stopThr
        self.epsilon = 1e-16
        self.transp = None
        self.cumgrad = torch.zeros((num_groups)).cuda()
        self.v = torch.zeros((num_groups)).cuda()
        self.nu = torch.ones((num_groups)).cuda() / num_groups

    def forward(self, M, group, batch_idx):
        if self.weight_loss_DCR <= 1e-6:
            return 0.
        else:
            # M: KxV cost matrix
            # sinkhorn alpha = 1/ entropic reg weight
            # a: Kx1 source distribution
            # b: Vx1 target distribution
            device = M.device
            # print('idx0: ', (idx == 0).any())
            if batch_idx != 0:
                self.cumgrad = self.cumgrad.detach()
                self.v = self.v.detach()
            else:
                self.cumgrad = torch.zeros((self.num_groups)).cuda()
                self.v = torch.zeros((self.num_groups)).cuda()
            group = group.to(device)
            # group /= group.sum()

            # SAG
            distmv = M - self.v.view(1, -1)
            exponent = -distmv * self.sinkhorn_alpha
            chi_unormalized = torch.exp(exponent) * self.nu
            chi = chi_unormalized / chi_unormalized.sum()

            grad = self.nu - chi
            self.cumgrad += grad.sum(axis=0) / self.num_data

            self.v -= self.otlr * self.cumgrad

            transpp = torch.exp((self.v-M)*self.sinkhorn_alpha)
            transp = transpp / transpp.sum()
            transp = transp.clamp(min=1e-6)

            self.transp = transp

            group = group.clamp(min=1e-6)
            transp = transp.clamp(min=1e-6)

            loss_DCR = (group * (group.log() - transp.log() - 1)
                        + transp).sum()
            loss_DCR *= self.weight_loss_DCR

            # print(loss_DCR)

            return loss_DCR
