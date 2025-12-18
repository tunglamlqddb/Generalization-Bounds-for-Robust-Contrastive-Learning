'''
Just loss creator functions to be used with other attacks.
'''
import torch


def normal_loss(model, criterion):
    model.set_mode('ssl')
    model.eval()
    return lambda x_adv: criterion(model(x_adv, adv=True))



def hard_pos(x, model):
    '''
    Focus on only generating hard positives.
    '''
    model.set_mode('ssl')
    model.eval()

    return lambda x_adv: -(model(x, adv=False) * model(x_adv, adv=True)).sum(dim=1).mean() 


def full_loss(x, model, criterion):
    '''
    Take the full loss loss into consideration.
    '''
    model.set_mode('ssl')
    model.eval()
    return lambda x_adv: criterion(torch.cat([model(x, adv=False), model(x_adv, adv=True)], dim=0))


def discr_loss(model, discriminator, label, reduction='mean', mode='eval', adv=False, detach=False):
    '''
    BCE loss on sample 
    label = True or Fake
    '''
    if mode=='eval':
        model.set_mode('ssl')
        model.eval()
        discriminator.eval()
    elif mode=='train':
        model.set_mode('ssl')
        model.train()
        discriminator.train()
    if detach:
        return lambda feat: torch.nn.BCELoss(reduction=reduction)(discriminator(feat.detach()), label)
    else:
        return lambda feat: torch.nn.BCELoss(reduction=reduction)(discriminator(feat), label)
    
