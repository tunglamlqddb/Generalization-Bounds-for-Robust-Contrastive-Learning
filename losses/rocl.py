'''
Implementation of the first RoCL loss, with 3 positive samples.
'''
import torch

def rocl_loss(features, temperature=0.5):

    sim = features @ features.T / temperature
    simexp = torch.exp(sim)

    # remove diagonals
    sim -= sim.diag()
    sim = torch.stack(torch.tensor_split(sim, 3, dim=0), dim=0)
    sim = torch.concat(torch.tensor_split(sim, 3, dim=2), dim=0)
    
    # sanity check
    assert sim.size(0) == 9
    loss = 2 * (simexp.sum(dim=1) - simexp.diag()).log().sum() - sim.diag_embed().sum()
    
    return loss / features.size(0)

def my_rocl_loss(features, temperature=0.5):
    N2 = features.size(0)
    N = N2 // 3

    sim = features @ features.T / temperature
    simexp = torch.exp(sim)
    # remove diagonals
    simexp = simexp * (1 - torch.eye(N2,N2).cuda()) 
    NT_xent_loss = -torch.log(simexp/(torch.sum(simexp,dim=1).view(N2,1) + 1e-8) + 1e-8)
        
    NT_xent_loss_total = (1./float(N2)) * torch.sum(
                                  torch.diag(NT_xent_loss[0:N,N:2*N])   + torch.diag(NT_xent_loss[N:2*N,0:N]) 
                                + torch.diag(NT_xent_loss[0:N,2*N:])    + torch.diag(NT_xent_loss[2*N:,0:N])
                                + torch.diag(NT_xent_loss[N:2*N,2*N:])  + torch.diag(NT_xent_loss[2*N:,N:2*N])
                                                        )
    return NT_xent_loss_total
