'''
I just want to put everything in a folder so that it looks cleaner.
'''
from .losses import hard_pos, full_loss, discr_loss, normal_loss
from attacks import pgd_attack
import torch
from losses import my_rocl_loss as rocl_loss 
from utils.fr_util import generate_high

d_criterion = torch.nn.BCELoss(reduction='mean')

def standard(images, model, criterion, adv=False):
    features = model(images, adv=adv)
    loss = criterion(features)
    return loss


def rosa(images, model, criterion, atk_obj='hard', **kwargs):
    '''
    @input atk_obj  : whether attack aims to create 'hard' positives, or
        targets the 'full' loss
    '''
    bsz = images.size(0) // 2
    p, q = torch.split(images, [bsz, bsz], dim=0)
    
    if atk_obj == 'hard':
        atk_obj = hard_pos(q, model)
    elif atk_obj == 'full':
        atk_obj = full_loss(q, model, criterion)
    else:
        raise 'Invalid attack objective!'
    
    q_ = pgd_attack(q, atk_obj, **kwargs)
    
    # loss
    ...

def gen_adv(images, model, criterion, args, discriminator=None, d_w=1.0, atk_obj='full', method='rocl'):
    '''
    images = [p,q]
    Note discriminator to use
    return adv samples
        - only q_ with ae4cl, rocl
        - p_, q_ with aclds
    '''
    model.eval()
    if discriminator is not None:
        discriminator.eval()

    bsz = images.size(0) // 2
    if method != 'aclds':
        p, q = torch.split(images, [bsz, bsz], dim=0)
    if method=='ae4cl':   # NCE(q_,q)
        tar = q
    elif method=='rocl':  # NCE(p,q_)
        tar = p
    elif method=='aclds': # NCE(p_,q_)
        bsz = images.size(0)    

    # NCE loss
    if atk_obj == 'hard':
        atk_obj = hard_pos(q, model)    # could this be tar?
    elif atk_obj == 'full':
        if method == 'aclds':
            atk_obj = normal_loss(model, criterion)
        else:
            atk_obj = full_loss(tar, model, criterion)   # NOTE: loss(p, q_) or loss(q)
    # D_loss
    if discriminator is not None:
        fake = torch.zeros((bsz, 1), requires_grad=False).cuda()
        discr_obj = discr_loss(model, discriminator, fake, reduction=args.d_reduction, mode='eval', adv=True)
        
        atk_obj_d = lambda x_adv: atk_obj(x_adv) - d_w * discr_obj(model(x_adv, adv=True)) 
        if method != 'aclds':
            q_ = pgd_attack(q, atk_obj_d, epsilon=args.epsilon, alpha=args.alpha, nb_iter=args.nb_iter) 
        else:
            q_ = pgd_attack(images, atk_obj_d, epsilon=args.epsilon, alpha=args.alpha, nb_iter=args.nb_iter) 
    else:
        if method != 'aclds':
            q_ = pgd_attack(q, atk_obj, epsilon=args.epsilon, alpha=args.alpha, nb_iter=args.nb_iter) 
        else:
            q_ = pgd_attack(images, atk_obj, epsilon=args.epsilon, alpha=args.alpha, nb_iter=args.nb_iter) 
    
    model.train()
    if discriminator is not None:
        discriminator.train()

    return q_


def rosa_loss(imgs, adv_img, model, criterion, discriminator=None, d_reduction='mean', method='ae4cl'):
    '''
    return a tuple of losses: 
        NCE on benign (p,q) 
        NCE on adv 
            rocl:       (p,q,q_)
            rocl_new:   (p,q_)
            ae4cl:      (q,q_)

        d_loss: optional
    '''
    assert(model.training)
    if discriminator is not None: assert(discriminator.training)
    
    bsz = imgs.size(0) // 2
    # p, q = torch.split(imgs, [bsz, bsz], dim=0)
    q_ = adv_img
    
    # get all feats here to ensure one forward for bn
    feat_r = model(imgs, adv=False)
    feat_r_p, feat_r_q = torch.split(feat_r, [bsz, bsz], dim=0)
    feat_f = model(q_, adv=True)
    
    # NCE loss on benign
    benign_loss = criterion(feat_r)
    # NCE loss on adv
    if method=='ae4cl':
        adv_loss = criterion(torch.cat([feat_r_q, feat_f], dim=0))
    elif method=='rocl_new':
        adv_loss = criterion(torch.cat([feat_r_p, feat_f], dim=0))
    elif method=='rocl':
        adv_loss = rocl_loss(torch.cat([feat_r_p, feat_r_q, feat_f], dim=0))
        reg_loss = criterion(torch.cat([feat_r_p, feat_f], dim=0))
        adv_loss += 1/256 * reg_loss
    elif method=='aclds':
        adv_loss = criterion(feat_f)
    else:
        raise 'Not implemented this LOSS yet!!'
    # D loss
    if discriminator is not None:
        if method != 'aclds':
            d_loss = get_d_loss(feat_r_q, feat_f, model, discriminator, d_reduction, bsz)
        else:
            d_loss = get_d_loss(feat_r, feat_f, model, discriminator, d_reduction, bsz)
    
    if discriminator is not None:
        return benign_loss, adv_loss, d_loss  
    else: return benign_loss, adv_loss


def get_d_loss(feat_r, feat_f, model, discriminator, reduction, bsz, detach=False):
    valid = torch.ones((bsz, 1), requires_grad=False).cuda()
    fake = torch.zeros((bsz, 1), requires_grad=False).cuda()
    d_loss_real = discr_loss(model, discriminator, valid, reduction, mode='train', adv=False, detach=detach)(feat_r)
    d_loss_fake = discr_loss(model, discriminator, fake, reduction, mode='train', adv=True, detach=detach)(feat_f)
    d_loss = 1/2 * (d_loss_real + d_loss_fake)
    return d_loss


def gen_adv_cl(images1, images2, images_org, model, criterion, args, discriminator=None, d_w=1.0):   # for the baseline AdvCL
    model.eval()
    if discriminator is not None:
        discriminator.eval()

    bsz = images1.size(0)
    x1 = images1.clone().detach()
    x2 = images2.clone().detach()
    x_cl = images_org.clone().detach()
    images_org_high = generate_high(x_cl.clone(), r=8.)
    x_HFC = images_org_high.clone().detach()
    
    x_cl = x_cl + torch.zeros_like(x1).uniform_(-8/255, 8/255)

    for i in range(args.nb_iter):
        x_cl.requires_grad_()
        with torch.enable_grad():
            f_proj = model(x_cl, adv=True)
            f1_proj = model(x1, adv=False)
            f2_proj = model(x2, adv=False)
            f_high_proj = model(x_HFC, adv=False)
            features = torch.cat([f_proj.unsqueeze(1), f1_proj.unsqueeze(1), f2_proj.unsqueeze(1), f_high_proj.unsqueeze(1)], dim=1)
            loss_contrast = criterion(features)
            if discriminator is not None:
                fake = torch.zeros((bsz, 1), requires_grad=False).cuda()
                d_loss = d_criterion(discriminator(f_proj), fake)
                loss = loss_contrast - d_w*d_loss
            else:
                loss = loss_contrast

        grad_x_cl = torch.autograd.grad(loss, x_cl)[0]
        x_cl = x_cl.detach() + args.alpha * torch.sign(grad_x_cl.detach())
        x_cl = torch.min(torch.max(x_cl, images_org - args.epsilon), images_org + args.epsilon)
        x_cl = torch.clamp(x_cl, 0, 1)

    model.train()
    if discriminator is not None:
        discriminator.train()
        
    return x1, x2, x_cl, x_HFC

    
