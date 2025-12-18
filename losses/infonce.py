'''
Implementation of the InfoNCE loss, aka NT-Xent
'''
import torch

def infonce(features, temperature):
    bsz = features.size(0)
    idxs = torch.arange(0, bsz)
    sim = features @ features.T / temperature
    simexp = torch.exp(sim)
    loss = (simexp.sum(dim=1) - simexp.diag()).log() - sim[idxs, idxs - bsz // 2]
    
    return loss.mean()

def pair_cosine_similarity(x, eps=1):
    # n = x.norm(p=2, dim=1, keepdim=True)
    # return (x @ x.t()) / (n * n.t()).clamp(min=eps)
    return x @ x.t()

def nt_xent(x, t=0.5):
    # print("device of x is {}".format(x.device))
    x = pair_cosine_similarity(x)
    x = torch.exp(x / t)
    idx = torch.arange(x.size()[0])
    # Put positive pairs on the diagonal
    idx[::2] += 1
    idx[1::2] -= 1
    x = x[idx]
    # subtract the similarity of 1 from the numerator
    x = x.diag() / (x.sum(0) - torch.exp(torch.tensor(1 / t)))
    return -torch.log(x).mean()