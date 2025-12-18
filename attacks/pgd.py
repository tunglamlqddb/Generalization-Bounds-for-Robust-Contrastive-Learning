import torch

@torch.no_grad()
def pgd_attack(x, loss_fn, epsilon=8/255, alpha=1/255, nb_iter=20, min=0, max=1):
    '''
    Untargeted PGD-Linf attack.
    Note that loss_fn can be partial to handle only x!
    '''

    # random initialization
    x_adv = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)
    
    for _ in range(nb_iter):
        x_adv.requires_grad_()
        with torch.enable_grad():
            grad = torch.autograd.grad(loss_fn(x_adv), x_adv)[0]
        x_adv = x + torch.clamp(
            x_adv + torch.sign(grad) * alpha - x, min=-epsilon, max=epsilon)

        # if we pass through a normalizer there's no valid range    
        if min is not None and max is not None:
            x_adv = torch.clamp(x_adv, min=min, max=max)
            
        x_adv = x_adv.detach()
    
    return x_adv
