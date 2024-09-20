import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .ECR import ECR
from .XGR import XGR


class L2gating(nn.Module):
    def __init__(self, vocab_size, num_groups):
        super().__init__()
        self.emb = torch.empty(num_groups, vocab_size)
        self.emb = nn.init.trunc_normal_(self.emb)
        self.emb = nn.Parameter(self.emb, requires_grad=True)

    def __call__(self, x):
        cdist = torch.cdist(x, self.emb)
        return -cdist


class SetToNegInf(nn.Module):
    def __init__(self, eps):
        super(SetToNegInf, self).__init__()
        self.eps = eps

    def forward(self, input_tensor):
        # Create a mask tensor where elements less than eps are marked as True
        mask = torch.logical_or((input_tensor < self.eps),
                                (input_tensor == torch.inf))

        # Use torch.where to conditionally set elements to -inf
        output_tensor = torch.where(mask, torch.tensor(float(
            '-inf'), dtype=input_tensor.dtype, device=input_tensor.device), input_tensor)

        # Save the mask for backward pass
        self.mask = mask

        return output_tensor

    def backward(self, grad_output):
        # Create a zero tensor with the same shape as the input tensor
        grad_input = torch.zeros_like(grad_output)

        # Set the gradient only for elements where mask is False (not set to -inf)
        grad_input[self.mask == False] = grad_output[self.mask == False]
        grad_input[torch.isnan(grad_input)] = 0

        print('set neg inf grad: ', grad_input)

        return grad_input


def masked_softmax(x, mask):
    x_mask = x * mask
    max_x_mask = torch.max(x_mask, dim=-1, keepdim=True).values
    x_mask = x_mask - max_x_mask
    exp_x_mask = torch.exp(x_mask) * mask
    sum_exp_x_mask = torch.sum(exp_x_mask, dim=-1, keepdim=True)
    return exp_x_mask / sum_exp_x_mask
    
    x_exp = torch.exp(x)
    x_exp_mask = x_exp * mask
    max_x_exp_mask = torch.max(x_exp_mask, dim=-1, keepdim=True)
    sum_masked_x_exp = torch.sum(x_exp*mask, dim=-1, keepdim=True)
    return (x_exp_mask / sum_masked_x_exp) * mask.float()


class XTMv2(nn.Module):
    def __init__(self, vocab_size, num_topics=50, num_groups=10, en_units=200,
                 dropout=0., pretrained_WE=None, embed_size=200, beta_temp=0.2,
                 weight_loss_XGR=250.0, weight_loss_ECR=250.0,
                 alpha_ECR=20.0, alpha_XGR=4.0, sinkhorn_max_iter=1000):
        super().__init__()

        self.num_topics = num_topics
        self.num_groups = num_groups
        self.beta_temp = beta_temp

        self.a = 1 * np.ones((1, num_topics)).astype(np.float32)
        self.mu2 = nn.Parameter(torch.as_tensor(
            (np.log(self.a).T - np.mean(np.log(self.a), 1)).T))
        self.var2 = nn.Parameter(torch.as_tensor(
            (((1.0 / self.a) * (1 - (2.0 / num_topics))).T + (1.0 / (num_topics * num_topics)) * np.sum(1.0 / self.a, 1)).T))

        self.mu2.requires_grad = False
        self.var2.requires_grad = False

        # self.gating_alpha = nn.Sequential(L2gating(vocab_size, num_groups),
        #                                   nn.Softmax(dim=-1))

        self.gating_alpha = nn.Sequential(nn.Linear(vocab_size, num_groups, bias=True),
                                          nn.Softmax(dim=-1))
        # self.set_neg_inf = SetToNegInf(1e-8)

        self.fc11 = nn.Linear(vocab_size, en_units)
        self.fc12 = nn.Linear(en_units, en_units)
        self.fc21 = nn.Linear(en_units, num_topics)
        self.fc22 = nn.Linear(en_units, num_topics)
        self.fc1_dropout = nn.Dropout(dropout)
        self.theta_dropout = nn.Dropout(dropout)

        self.mean_bn = nn.BatchNorm1d(num_topics)
        self.mean_bn.weight.requires_grad = False
        self.logvar_bn = nn.BatchNorm1d(num_topics)
        self.logvar_bn.weight.requires_grad = False
        self.decoder_bn = nn.BatchNorm1d(vocab_size, affine=True)
        self.decoder_bn.weight.requires_grad = False

        if pretrained_WE is not None:
            self.word_embeddings = torch.from_numpy(pretrained_WE).float()
        else:
            self.word_embeddings = nn.init.trunc_normal_(
                torch.empty(vocab_size, embed_size))
        self.word_embeddings = nn.Parameter(F.normalize(self.word_embeddings))

        self.topic_embeddings = torch.empty(
            (num_topics, self.word_embeddings.shape[1]))
        nn.init.trunc_normal_(self.topic_embeddings, std=0.1)
        self.topic_embeddings = nn.Parameter(
            F.normalize(self.topic_embeddings))

        assert (num_topics % num_groups == 0,
                'num_topics should be divisible by num_groups')
        self.num_topics_per_group = num_topics // num_groups

        self.ECR = ECR(weight_loss_ECR, alpha_ECR, sinkhorn_max_iter)
        self.XGR = XGR(weight_loss_XGR, alpha_XGR, sinkhorn_max_iter)
        self.group_connection_regularizer = torch.ones(
            (self.num_topics_per_group, self.num_topics_per_group))
        for _ in range(num_groups-1):
            self.group_connection_regularizer = torch.block_diag(self.group_connection_regularizer, torch.ones(
                (self.num_topics_per_group, self.num_topics_per_group)))
        self.group_connection_regularizer /= self.group_connection_regularizer.sum()

    def get_beta(self):
        dist = self.pairwise_euclidean_distance(
            self.topic_embeddings, self.word_embeddings)
        beta = F.softmax(-dist / self.beta_temp, dim=0)
        return beta

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + (eps * std)
        else:
            return mu

    def set_to_neg_inf(self, input_tensor, eps):
        # Create a mask tensor where elements less than eps are marked as True
        mask = input_tensor < eps

        # Use torch.where to conditionally set elements to -inf
        output_tensor = torch.where(mask, torch.tensor(float(
            '-inf'), dtype=input_tensor.dtype, device=input_tensor.device), input_tensor)

        return output_tensor

    def global_expert_neg_entropy_loss(self, p):
        sum_p = torch.mean(p, dim=0)
        return torch.mean(sum_p * torch.log(sum_p + 1e-8)) * 200.
    
    def local_expert_entropy_loss(self, p):
        return -torch.sum(p * torch.log(p + 1e-8), axis=1).mean() * 5.


    def encode(self, input, eps=1e-8):
        e1 = F.softplus(self.fc11(input))
        e1 = F.softplus(self.fc12(e1))
        e1 = self.fc1_dropout(e1)
        mu = self.mean_bn(self.fc21(e1))
        logvar = self.logvar_bn(self.fc22(e1))
        z = self.reparameterize(mu, logvar)

        p = self.gating_alpha(input)
        p = p.clamp(eps, 1-eps)
        global_loss = self.global_expert_neg_entropy_loss(p)
        local_loss = self.local_expert_entropy_loss(p)
        alpha = F.gumbel_softmax(torch.log(p), tau=1e-2)
        # alpha = self.set_neg_inf(alpha)
        alpha_all_topics = alpha.unsqueeze(dim=-1) \
                                .repeat(*[[1]*alpha.ndim + [self.num_topics_per_group]]) \
                                .flatten(start_dim=-2)

        theta = masked_softmax(alpha_all_topics * z, alpha_all_topics >= eps)

        loss_KL_gauss = self.compute_loss_KL_gauss(
            mu, logvar)
        loss_KL_ber = self.compute_loss_KL_ber(p)

        return theta, loss_KL_gauss, loss_KL_ber, global_loss, local_loss

    def get_theta(self, input):
        theta, loss_KL_gauss, loss_KL_ber, global_loss, local_loss = self.encode(input)
        if self.training:
            return theta, loss_KL_gauss, loss_KL_ber, global_loss, local_loss
        else:
            return theta

    def compute_loss_KL_gauss(self, mu, logvar):
        var = logvar.exp()
        var_division = var / self.var2
        diff = mu - self.mu2
        diff_term = diff * diff / self.var2
        logvar_division = self.var2.log() - logvar
        # KLD: N*K
        KLD = 0.5 * ((var_division + diff_term +
                     logvar_division).sum(axis=1) - self.num_topics)
        KLD = KLD.mean()
        return KLD

    def compute_loss_KL_ber(self, p, eps=1e-4):
        q = (torch.ones_like(p) / p.shape[-1]).to(p.device)
        p1 = p.clamp(eps, 1.0 - eps)
        KLD = p * (p1.log() - q.log()) + (1 - p) * \
            ((1 - p1).log() - (1 - q).log())
        KLD = KLD.mean()
        return KLD * 10.

    def get_loss_ECR(self):
        cost = self.pairwise_euclidean_distance(
            self.topic_embeddings, self.word_embeddings)
        loss_ECR = self.ECR(cost)
        return loss_ECR

    def get_loss_XGR(self):
        cost = self.pairwise_euclidean_distance(
            self.topic_embeddings, self.topic_embeddings)
        loss_XGR = self.XGR(cost, self.group_connection_regularizer)
        return loss_XGR

    def pairwise_euclidean_distance(self, x, y):
        cost = torch.sum(x ** 2, axis=1, keepdim=True) + \
            torch.sum(y ** 2, dim=1) - 2 * torch.matmul(x, y.t())
        return cost

    def forward(self, input):
        input = input['data']
        theta, loss_KL_gauss, loss_KL_ber, global_loss, local_loss = self.encode(input)
        loss_KL_ber = 0.
        beta = self.get_beta()

        recon = F.softmax(self.decoder_bn(torch.matmul(theta, beta)), dim=-1)
        recon_loss = -(input * recon.log()).sum(axis=1).mean()

        loss_TM = recon_loss + loss_KL_gauss + loss_KL_ber + global_loss + local_loss

        loss_ECR = self.get_loss_ECR()
        loss_XGR = self.get_loss_XGR()
        loss = loss_TM + loss_ECR + loss_XGR

        rst_dict = {
            'loss': loss,
            'loss_TM': loss_TM,
            'loss_KL_gauss': loss_KL_gauss,
            'loss_KL_ber': loss_KL_ber,
            'loss_ECR': loss_ECR,
            'loss_XGR': loss_XGR,
            'global_loss': global_loss,
            'local_loss': local_loss,
        }

        return rst_dict
