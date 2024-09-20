import torch
import random
import numpy as np

class PCGrad:
    def __init__(self):
        pass

    def compute_weights(self, grads):
        num_tasks = len(grads)
        grads = [g.clone() for g in grads] 

        pc_grads = [g.clone() for g in grads]
        for i in range(num_tasks):
            task_indices = list(range(num_tasks))
            random.shuffle(task_indices)
            for j in task_indices:
                if i != j:
                    g_i = pc_grads[i]
                    g_j = grads[j]
                    g_i_dot_g_j = torch.dot(g_i, g_j)
                    if g_i_dot_g_j < 0:
                        pc_grads[i] = g_i - (g_i_dot_g_j / (g_j.norm() ** 2 + 1e-8)) * g_j

        grads_norm = [g.norm().item() for g in pc_grads]
        total_norm = sum(grads_norm)
        weights = [norm / total_norm for norm in grads_norm]
        weights = np.array(weights)

        return weights, pc_grads
