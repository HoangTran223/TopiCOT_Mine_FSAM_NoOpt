import torch
import torch.nn as nn
import numpy as np
import cvxpy as cp

class NashMTL:
    def __init__(self, num_tasks, device='cpu', update_weights_every=1, optim_niter=20, max_norm=1.0):
        self.num_tasks = num_tasks
        self.device = device
        self.update_weights_every = update_weights_every
        self.optim_niter = optim_niter
        self.max_norm = max_norm
        self.step = 0
        self.prvs_alpha = np.ones(self.num_tasks, dtype=np.float32)
        self.normalization_factor = np.ones((1,), dtype=np.float32)
        self.init_gtg = np.eye(self.num_tasks, dtype=np.float32)
        self._init_optim_problem()
    
    def _init_optim_problem(self):
        self.cp = cp
        self.alpha_param = cp.Variable(shape=(self.num_tasks,), nonneg=True)
        self.prvs_alpha_param = cp.Parameter(shape=(self.num_tasks,), value=self.prvs_alpha)
        self.G_param = cp.Parameter(shape=(self.num_tasks, self.num_tasks), value=self.init_gtg)
        self.normalization_factor_param = cp.Parameter(shape=(1,), value=self.normalization_factor)
        self.phi_alpha = self._calc_phi_alpha_linearization()
        G_alpha = self.G_param @ self.alpha_param
        constraints = []
        for i in range(self.num_tasks):
            constraints.append(
                -cp.log(self.alpha_param[i] * self.normalization_factor_param)
                - cp.log(G_alpha[i])
                <= 0
            )
        obj = cp.Minimize(
            cp.sum(G_alpha) + self.phi_alpha / self.normalization_factor_param
        )
        self.prob = cp.Problem(obj, constraints)
    
    def _calc_phi_alpha_linearization(self):
        G_prvs_alpha = self.G_param @ self.prvs_alpha_param
        prvs_phi_tag = 1 / self.prvs_alpha_param + (1 / G_prvs_alpha) @ self.G_param
        phi_alpha = prvs_phi_tag @ (self.alpha_param - self.prvs_alpha_param)
        return phi_alpha
    
    def _stop_criteria(self, gtg, alpha_t):
        return (
            (self.alpha_param.value is None)
            or (np.linalg.norm(gtg @ alpha_t - 1 / (alpha_t + 1e-10)) < 1e-3)
            or (
                np.linalg.norm(self.alpha_param.value - self.prvs_alpha_param.value)
                < 1e-6
            )
        )
    
    def solve_optimization(self, gtg):
        self.G_param.value = gtg
        self.normalization_factor_param.value = self.normalization_factor

        alpha_t = self.prvs_alpha
        for _ in range(self.optim_niter):
            self.alpha_param.value = alpha_t
            self.prvs_alpha_param.value = alpha_t

            try:
                self.prob.solve(solver=self.cp.ECOS, warm_start=True, max_iters=100)
            except:
                self.alpha_param.value = self.prvs_alpha_param.value

            if self._stop_criteria(gtg, alpha_t):
                break

            alpha_t = self.alpha_param.value

        if alpha_t is not None:
            self.prvs_alpha = alpha_t

        return self.prvs_alpha
    
    def compute_weights(self, grads):
        if self.step % self.update_weights_every == 0:
            grads = [g.view(-1) for g in grads]
            grads = torch.stack(grads)  
            grads = grads.to(self.device)
            GTG = torch.mm(grads, grads.t())
            self.normalization_factor = torch.norm(GTG).detach().cpu().numpy().reshape((1,))
            GTG_normalized = GTG / self.normalization_factor.item()
            gtg_np = GTG_normalized.cpu().detach().numpy()
            alpha = self.solve_optimization(gtg_np)
        else:
            alpha = self.prvs_alpha

        self.step += 1
        weights = torch.from_numpy(alpha).to(torch.float32).to(self.device)
        weights = weights / weights.sum()
        weights = weights.detach().cpu().numpy()
        return weights

