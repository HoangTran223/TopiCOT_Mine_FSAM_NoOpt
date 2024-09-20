import torch
import torch.nn as nn
import numpy as np

class IMTL:
    def __init__(self):
        pass

    '''def compute_weights(self, grads):
        num_tasks = len(grads)
        grads = [g.view(-1) for g in grads]
        grads = torch.stack(grads)  
        
        grads_norm = torch.norm(grads, p=2, dim=-1, keepdim=True)
        grads_unit = grads / (grads_norm + 1e-8)
        
        D = grads[0:1].repeat(num_tasks - 1, 1) - grads[1:]
        U = grads_unit[0:1].repeat(num_tasks - 1, 1) - grads_unit[1:]
        
        A = torch.matmul(D, U.t())  
        b = torch.matmul(grads[0], U.t())  
        
        A += 1e-8 * torch.eye(A.size(0)).to(A.device)
        
        A_inv = torch.inverse(A)
        
        alpha = torch.matmul(b, A_inv)  
        
        alpha = torch.cat((1 - alpha.sum().unsqueeze(0), alpha), dim=0)  
        
        weights = alpha / alpha.sum()
        weights = weights.detach().cpu().numpy()
        
        return weights'''
    
    def compute_weights(self, grads):
        num_tasks = len(grads)
        grads = torch.stack([g.view(-1) for g in grads])
        #grads = torch.stack(grads)  

        #grads_norm = torch.norm(grads, p=2, dim=-1, keepdim=True)
        #grads_unit = grads / (torch.norm(grads, p=2, dim=-1, keepdim=True) + 1e-8)

        D = grads[0:1].repeat(num_tasks - 1, 1) - grads[1:]
        #U = grads_unit[0:1].repeat(num_tasks - 1, 1) - grads_unit[1:]
        U = (grads / (torch.norm(grads, p=2, dim=-1, keepdim=True) + 1e-8))[0:1].repeat(num_tasks - 1, 1) - (grads / (torch.norm(grads, p=2, dim=-1, keepdim=True) + 1e-8))[1:]
        A = torch.matmul(D, U.t())  
        b = torch.matmul(grads[0], U.t())  

        A += 1e-8 * torch.eye(A.size(0)).to(A.device)

        A_inv = torch.inverse(A)

        alpha = torch.matmul(b, A_inv)  

        alpha = torch.cat((1 - alpha.sum().unsqueeze(0), alpha), dim=0)  

        weights = alpha / alpha.sum()
        weights = weights.detach().cpu().numpy()
        return weights
