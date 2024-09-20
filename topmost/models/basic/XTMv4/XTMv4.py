import numpy as np
import torch
import copy
from torch import nn
import torch.nn.functional as F
import logging
from .ECR import ECR
from .XGR import XGR

cnt = 0


class Distgating(nn.Module):
    def __init__(self, vocab_size, num_groups, p):
        super().__init__()
        self.p = p
        self.emb = torch.empty(num_groups, vocab_size)
        self.emb = nn.init.trunc_normal_(self.emb)
        self.emb = nn.Parameter(self.emb, requires_grad=True)

    def __call__(self, x):
        cdist = torch.cdist(x, self.emb, p=self.p)
        return -cdist


def masked_softmax(vector: torch.Tensor,
                   mask: torch.Tensor,
                   dim: int = -1,
                   memory_efficient: bool = False,
                   mask_fill_value: float = -1e32) -> torch.Tensor:
    """
    ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular softmax.

    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.

    If ``memory_efficient`` is set to true, we will simply use a very large negative number for those
    masked positions so that the probabilities of those positions would be approximately 0.
    This is not accurate in math, but works for most cases and consumes less memory.

    In the case that the input vector is completely masked and ``memory_efficient`` is false, this function
    returns an array of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of
    a model that uses categorical cross-entropy loss. Instead, if ``memory_efficient`` is true, this function
    will treat every element as equal, and do softmax over equal numbers.
    """
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill(
                (1 - mask).byte(), mask_fill_value)
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result


class XTMv4(nn.Module):
    def __init__(self, vocab_size, num_topics=50, num_groups=10, en_units=200,
                 dropout=0., pretrained_WE=None, embed_size=200, beta_temp=0.2,
                 weight_loss_XGR=250.0, weight_loss_ECR=250.0,
                 alpha_ECR=20.0, alpha_XGR=4.0, sinkhorn_max_iter=1000,
                 gating_func='dot', weight_global_expert=250., weight_local_expert=250.,
                 k=1):
        super().__init__()

        self.num_topics = num_topics
        self.num_groups = num_groups
        self.beta_temp = beta_temp
        assert (num_topics % num_groups == 0,
                'num_topics should be divisible by num_groups')
        self.num_topics_per_group = num_topics // num_groups
        self.n_global_group = 0

        self.a = 1 * np.ones((1, num_topics)).astype(np.float32)
        self.mu2 = nn.Parameter(torch.as_tensor(
            (np.log(self.a).T - np.mean(np.log(self.a), 1)).T))
        self.var2 = nn.Parameter(torch.as_tensor(
            (((1.0 / self.a) * (1 - (2.0 / num_topics))).T + (1.0 / (num_topics * num_topics)) * np.sum(1.0 / self.a, 1)).T))

        self.mu2.requires_grad = False
        self.var2.requires_grad = False

        # self.gating_alpha = nn.Sequential(L2gating(vocab_size, num_groups),
        #                                   nn.Softmax(dim=-1))

        if gating_func == 'dot':
            self.gating_alpha = nn.Sequential(
                # nn.BatchNorm1d(vocab_size),
                nn.Linear(vocab_size, self.num_groups - self.n_global_group, bias=False))
        elif gating_func == 'dot_bias':
            self.gating_alpha = nn.Sequential(
                # nn.BatchNorm1d(vocab_size),
                nn.Linear(vocab_size, self.num_groups - self.n_global_group, bias=True))
        elif gating_func == 'L1':
            print('dont use this')
            self.gating_alpha = nn.Sequential(
                Distgating(vocab_size, self.num_groups - self.n_global_group, 1))
        elif gating_func == 'L2':
            self.gating_alpha = nn.Sequential(
                Distgating(vocab_size, self.num_groups - self.n_global_group, 2))
        else:
            raise NotImplementedError

        self.weight_global_expert = weight_global_expert
        self.weight_local_expert = weight_local_expert

        self.ffn_enc_list = nn.ModuleList()
        for _ in range(self.num_groups - self.n_global_group):
            self.ffn_enc_list.append(nn.Sequential(
                nn.Linear(vocab_size, en_units),
                nn.Softplus(),
                nn.Linear(en_units, en_units),
                nn.Softplus(),
                nn.Dropout(dropout)
            ))
        self.ffn_mu_list = nn.ModuleList()
        self.ffn_logvar_list = nn.ModuleList()
        for _ in range(self.num_groups - self.n_global_group):
            self.ffn_mu_list.append(nn.Sequential(
                nn.Linear(en_units, self.num_topics_per_group),
                nn.BatchNorm1d(self.num_topics_per_group)
            ))
            self.ffn_logvar_list.append(nn.Sequential(
                nn.Linear(en_units, self.num_topics_per_group),
                nn.BatchNorm1d(self.num_topics_per_group)
            ))
            
        # self.fc11 = nn.Linear(vocab_size, en_units)
        # self.fc12 = nn.Linear(en_units, en_units)
        # self.fc21 = nn.Linear(en_units, num_topics)
        # self.fc22 = nn.Linear(en_units, num_topics)
        # self.fc1_dropout = nn.Dropout(dropout)
        self.theta_dropout = nn.Dropout(dropout)

        self.mean_bn = nn.BatchNorm1d(num_topics)
        self.mean_bn.weight.requires_grad = False
        self.logvar_bn = nn.BatchNorm1d(num_topics)
        self.logvar_bn.weight.requires_grad = False
        self.decoder_bn = nn.BatchNorm1d(vocab_size, affine=True)
        self.decoder_bn.weight.requires_grad = False

        self.k = k

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

        self.group_embeddings = torch.empty(
            (num_groups, self.word_embeddings.shape[1]))
        nn.init.trunc_normal_(self.group_embeddings, std=0.1)
        self.group_embeddings = nn.Parameter(
            F.normalize(self.group_embeddings))

        self.ECR = ECR(weight_loss_ECR, alpha_ECR, sinkhorn_max_iter)
        self.XGR = ECR(weight_loss_XGR, alpha_XGR, sinkhorn_max_iter)
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
        return torch.mean(sum_p * torch.log(sum_p + 1e-6)) * self.weight_global_expert

    def local_expert_entropy_loss(self, p):
        return -torch.sum(p * torch.log(p + 1e-6), axis=1).mean() * self.weight_local_expert

    def encode(self, input, eps=1e-8):
        list_z = []
        list_mu = []
        list_logvar = []
        p = F.softmax(self.gating_alpha(input), dim=1)
        for i in range(self.num_groups - self.n_global_group):
            e1 = self.ffn_enc_list[i](input)
            mu = self.ffn_mu_list[i](e1)
            logvar = self.ffn_logvar_list[i](e1)
            list_mu.append(mu)
            list_logvar.append(logvar)
            z = self.reparameterize(mu, logvar)
            list_z.append(z*p[:, i].unsqueeze(1))
        mu = torch.cat(list_mu, dim=1)
        logvar = torch.cat(list_logvar, dim=1)
        z = torch.cat(list_z, dim=1)
        
        print(mu.shape, logvar.shape, z.shape)
    

        z_softmax = F.softmax(z, dim=1)

        theta = z_softmax
        theta /= theta.sum(dim=1, keepdim=True)
        global_loss = 0.
        local_loss = 0.

        print(f'global_loss: {global_loss}')
        print(f'local_loss: {local_loss}')
        print('**************')


        loss_KL_gauss = self.compute_loss_KL_gauss(mu, logvar)
        loss_KL_ber = 0.

        return theta, loss_KL_gauss, loss_KL_ber, global_loss, local_loss

    def get_theta(self, input):
        theta, loss_KL_gauss, loss_KL_ber, global_loss, local_loss = self.encode(
            input)
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
            self.group_embeddings, self.topic_embeddings)
        logger = logging.getLogger('main')
        # logger.info(f'group-topic shape: {cost.shape}')
        # logger.info(f'group-topic: {cost[:3, :7]}')
        loss_XGR = self.XGR(cost)
        return loss_XGR

    def get_OT_plan_group_topic(self):
        return self.XGR.transp

    def pairwise_euclidean_distance(self, x, y):
        cost = torch.sum(x ** 2, axis=1, keepdim=True) + \
            torch.sum(y ** 2, dim=1) - 2 * torch.matmul(x, y.t())
        return cost

    def forward(self, input):
        input = input['data']
        theta, loss_KL_gauss, loss_KL_ber, global_loss, local_loss = self.encode(
            input)
        beta = self.get_beta()

        recon = F.softmax(self.decoder_bn(torch.matmul(theta, beta)), dim=-1)
        recon_loss = -(input * recon.log()).sum(axis=1).mean()

        loss_TM = recon_loss + loss_KL_gauss + loss_KL_ber

        loss_ECR = self.get_loss_ECR()
        loss_XGR = self.get_loss_XGR()
        loss = loss_TM + loss_ECR + loss_XGR + global_loss + local_loss

        rst_dict = {
            'loss': loss,
            'loss_TM': loss_TM,
            'loss_KL_gauss': loss_KL_gauss,
            'loss_KL_ber': loss_KL_ber,
            'loss_ECR': loss_ECR,
            'loss_XGR': loss_XGR,
            'global_loss': global_loss,
            'local_loss': local_loss
        }

        return rst_dict
