import torch
import torch.nn as nn
import numpy as np
import cvxpy as cp

class MGDA:
    def __init__(self):
        pass

    def compute_weights(self, grads):
        num_tasks = len(grads)
        #grads = [g.view(-1).cpu().numpy() for g in grads]
        grads = [g.view(-1).numpy() for g in grads]
        grads = np.stack(grads) 

        GG = grads @ grads.T 

        alphas = cp.Variable(num_tasks)

        objective = cp.Minimize(0.5 * cp.quad_form(alphas, GG))

        constraints = [cp.sum(alphas) == 1, alphas >= 0]

        problem = cp.Problem(objective, constraints)
        problem.solve()

        weights = alphas.value
        return weights

