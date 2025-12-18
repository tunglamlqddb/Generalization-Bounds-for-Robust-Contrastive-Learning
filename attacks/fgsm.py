import torch

@torch.no_grad()
def fgsm_attack(x, loss_fn, epsilon=8/255, min=0, max=1):
    '''
    Untargeted FGSM-Linf attack.
    Note that loss_fn can be partial to handle only x!
    '''

    with torch.enable_grad():
        x.requires_grad_()
        grad = torch.autograd.grad(loss_fn(x), x)

    x_adv = x + torch.sign(grad) * epsilon

    # if we pass through a normalizer there's no valid range    
    if min is not None and max is not None:
        x_adv = torch.clamp(x_adv, min=min, max=max)
        
    return x_adv.detach()
