import torch
import sys
from utils.utils import (
    enable_running_stats, disable_running_stats, AverageMeter, save_model,
    warmup_lr, cosine_lr, get_grad
)
from forwards import gen_adv, rosa_loss, get_d_loss, gen_adv_cl



def update_adv_no_d(images, labels, model, criterion, optimizer, use_sam, args, epoch, minibatch_idx, len_loader):
    '''
    Adv training with SAM on benign samples
    No Discriminator involved
    return benign_loss and adv_loss before SAM
    '''
    
    images = torch.cat([images[0], images[1]], dim=0)
    images = images.to(args.device, non_blocking=True)
    labels = labels.to(args.device, non_blocking=True)
    
    enable_running_stats(model)  # put it here?
    # get adv with no D 
    optimizer.zero_grad()
    adv_images = gen_adv(images, model, criterion, args, atk_obj='full', method=args.method_gen)
    # actual loss 
    benign_loss, adv_loss = rosa_loss(images, adv_images, model, criterion, method=args.method_loss, d_reduction=args.d_reduction)
    benign_loss *= args.benign_w 
    benign_loss_return, adv_loss_return = benign_loss.item(), adv_loss.item()
    # SAM on benign
    if not use_sam:
        optimizer.zero_grad()
        (benign_loss + adv_loss).backward()
        optimizer.step()
    else:
        if args.benign:    # only SAM on benign
            optimizer.zero_grad()
            adv_loss.backward(retain_graph=True)
            adv_grads = get_grad(optimizer.param_groups)
            optimizer.zero_grad()
            benign_loss.backward()
            optimizer.first_step(zero_grad=True)
            disable_running_stats(model)
            benign_loss = args.benign_w * criterion(model(images, adv=False))   # get benign_loss again
            benign_loss.backward()
            optimizer.second_step(zero_grad=True, g_other=adv_grads)
        else:              # SAM on all loss
            optimizer.zero_grad()
            (benign_loss + adv_loss).backward()
            optimizer.first_step(zero_grad=True)
            disable_running_stats(model)
            benign_loss, adv_loss = rosa_loss(images, adv_images, model, criterion, method=args.method_loss, d_reduction=args.d_reduction)
            benign_loss *= args.benign_w 
            (benign_loss + adv_loss).backward()
            optimizer.second_step(zero_grad=True)
    
    # warm-up learning rate
    if epoch < args.warm_epochs:
        for param_group in optimizer.param_groups:
            param_group['lr'] = warmup_lr(
                args.learning_rate, epoch, args.warm_epochs, args.epochs,
                minibatch_idx, len_loader, args.eta_min
            )

    return benign_loss_return, adv_loss_return
    
    
    
def update_adv_d(images, labels, model, discriminator, criterion, optimizer, d_optimizer, d_w, d_step, args, epoch, minibatch_idx, len_loader):
    '''
    d_w: weight for d_loss: can be warmed up
    int d_step=3 or 4: follow 3-step or 4-step

    Adv training with SAM on benign samples
    and with Discriminator involved
    return benign_loss and adv_loss before SAM
    '''
    
    images = torch.cat([images[0], images[1]], dim=0)
    images = images.to(args.device, non_blocking=True)
    labels = labels.to(args.device, non_blocking=True)
    bsz = labels.shape[0] # will this work with STL-10?

    enable_running_stats(model)  # put it here?
    # get adv with D or without D depend on args.use_d_gen
    optimizer.zero_grad()
    d_optimizer.zero_grad()
    if d_step == 3 and args.use_d_gen:
        adv_images = gen_adv(images, model, criterion, args, discriminator, d_w=d_w, atk_obj='full', method=args.method_gen)
    else:   # d_step==4 or use_d_gen==False
        adv_images = gen_adv(images, model, criterion, args, discriminator=None, atk_obj='full', method=args.method_gen)
    
    # train D
    optimizer.zero_grad()
    d_optimizer.zero_grad()
    model.eval()           # set eval to model when training D so that model(x) does not affect bn
    feat_r = model(images[bsz:], adv=False)     
    feat_f = model(adv_images, adv=True)
    d_loss = get_d_loss(feat_r, feat_f, model, discriminator, args.d_reduction, bsz, detach=True)    # NOTE need detach from z?
    d_loss.backward()
    d_optimizer.step()
    model.train()           # set train back to model
    
    if d_step == 4:
        adv_images = gen_adv(images, model, criterion, args, discriminator, d_w=d_w, atk_obj='full', method=args.method_gen)
    
    # actual loss 
    optimizer.zero_grad()
    d_optimizer.zero_grad()
    # model.train()
    benign_loss, adv_loss, d_loss = rosa_loss(images, adv_images, model, criterion, discriminator, d_reduction=args.d_reduction, method=args.method_loss)
    benign_loss *= args.benign_w 
    d_loss *= d_w
    benign_loss_return, adv_loss_return, d_loss_return = benign_loss.item(), adv_loss.item(), d_loss.item()
    # SAM on benign
    if not args.sam:
        optimizer.zero_grad()
        d_optimizer.zero_grad()
        (benign_loss + adv_loss - d_loss).backward()
        optimizer.step()
    else:
        if args.benign:    # only SAM on benign
            optimizer.zero_grad()
            d_optimizer.zero_grad()
            (adv_loss - d_loss).backward(retain_graph=True)
            adv_grads = get_grad(optimizer.param_groups)      # also get from D_optimizer? Nah
            optimizer.zero_grad()
            benign_loss.backward()
            optimizer.first_step(zero_grad=True)
            disable_running_stats(model)
            benign_loss = args.benign_w * criterion(model(images, adv=False))   # get benign_loss again
            benign_loss.backward()
            optimizer.second_step(zero_grad=True, g_other=adv_grads)
        else:              # SAM on all loss
            optimizer.zero_grad()
            (benign_loss + adv_loss - d_loss).backward()
            optimizer.first_step(zero_grad=True)
            d_optimizer.zero_grad()
            disable_running_stats(model)
            benign_loss, adv_loss, d_loss = rosa_loss(images, adv_images, model, criterion, discriminator, d_reduction=args.d_reduction, method=args.method_loss)
            benign_loss *= args.benign_w 
            d_loss *= d_w
            (benign_loss + adv_loss - d_loss).backward()
            optimizer.second_step(zero_grad=True)
    
    # warm-up learning rate
    if epoch < args.warm_epochs:
        for param_group in optimizer.param_groups:
            param_group['lr'] = warmup_lr(
                args.learning_rate, epoch, args.warm_epochs, args.epochs,
                minibatch_idx, len_loader, args.eta_min
            )
        for param_group in d_optimizer.param_groups:
            param_group['lr'] = warmup_lr(
                args.learning_rate_d, epoch-args.start_from, args.warm_epochs, args.epochs-args.start_from,
                minibatch_idx, len_loader, args.eta_min
            )

    return benign_loss_return, adv_loss_return, d_loss_return
    
    
    
def update_adv_cl(images, labels, model, criterion, optimizer, args, epoch, minibatch_idx, len_loader):   # for baseline AdvCL
    
    images_org = images[2].to(args.device, non_blocking=True)
    images1 = images[0].to(args.device, non_blocking=True)
    images2 = images[1].to(args.device, non_blocking=True)
    labels = labels.to(args.device, non_blocking=True)
    
    optimizer.zero_grad()
    
    x1, x2, x_cl, x_HFC = gen_adv_cl(images1, images2, images_org, model, criterion, args)
    f_proj = model(x_cl, adv=True)
    f1_proj = model(x1, adv=False)
    f2_proj = model(x2, adv=False)
    f_high_proj = model(x_HFC, adv=False)
    features = torch.cat([f_proj.unsqueeze(1), f1_proj.unsqueeze(1), f2_proj.unsqueeze(1), f_high_proj.unsqueeze(1)], dim=1)
    loss = criterion(features)
     
    loss.backward()
    optimizer.step()
    
    # warm-up learning rate
    if epoch < args.warm_epochs:
        for param_group in optimizer.param_groups:
            param_group['lr'] = warmup_lr(
                args.learning_rate, epoch, args.warm_epochs, args.epochs,
                minibatch_idx, len_loader, args.eta_min
            )

    return loss.item()


def update_adv_cl_full(images, labels, model, discriminator, criterion, optimizer, d_optimizer, d_w, args, epoch, minibatch_idx, len_loader):   # for baseline AdvCL
    
    images_org = images[2].to(args.device, non_blocking=True)
    bsz = images_org.shape[0]
    images1 = images[0].to(args.device, non_blocking=True)
    images2 = images[1].to(args.device, non_blocking=True)
    labels = labels.to(args.device, non_blocking=True)
    
    # enable_running_stats(model)
    optimizer.zero_grad()
    d_optimizer.zero_grad()
    # 3steps & use_d_gen=True
    x1, x2, x_cl, x_HFC = gen_adv_cl(images1, images2, images_org, model, criterion, args, discriminator, d_w=d_w)
    
    #train D
    optimizer.zero_grad()
    d_optimizer.zero_grad()
    model.eval()
    feat_r = model(images_org, adv=False)     
    feat_f = model(x_cl, adv=True)
    d_loss = get_d_loss(feat_r, feat_f, model, discriminator, args.d_reduction, bsz, detach=True)
    d_loss.backward()
    d_optimizer.step()
    model.train()           
    
    # actual loss
    optimizer.zero_grad()
    d_optimizer.zero_grad()
    
    f_org_proj = model(images_org, adv=False)
    f_proj = model(x_cl, adv=True)
    f1_proj = model(x1, adv=False)
    f2_proj = model(x2, adv=False)
    f_high_proj = model(x_HFC, adv=False)
    
    if args.benign_type == 'simclr':
        features_benign = torch.cat([f1_proj.unsqueeze(1), f2_proj.unsqueeze(1)], dim=1)
        benign_loss = criterion(features_benign) * args.benign_w
    elif args.benign_type == 'advcl':
        features = torch.cat([f_org_proj.unsqueeze(1), f1_proj.unsqueeze(1), f2_proj.unsqueeze(1), f_high_proj.unsqueeze(1)], dim=1)
        benign_loss = criterion(features) * args.benign_w
        
    d_loss = get_d_loss(f_org_proj, f_proj, model, discriminator, args.d_reduction, bsz) * d_w
    
    features = torch.cat([f_proj.unsqueeze(1), f1_proj.unsqueeze(1), f2_proj.unsqueeze(1), f_high_proj.unsqueeze(1)], dim=1)

    adv_loss = criterion(features)

    # SAM on benign
    optimizer.zero_grad()
    d_optimizer.zero_grad()
    (adv_loss - d_loss).backward(retain_graph=True)
    adv_grads = get_grad(optimizer.param_groups)      
    optimizer.zero_grad()
    benign_loss.backward()
    optimizer.first_step(zero_grad=True)
    # disable_running_stats(model)
    if args.benign_type == 'simclr':
        f1_proj = model(x1, adv=False)
        f2_proj = model(x2, adv=False)
        features_benign = torch.cat([f1_proj.unsqueeze(1), f2_proj.unsqueeze(1)], dim=1)
        benign_loss = args.benign_w * criterion(features_benign)   
    elif args.benign_type == 'advcl':
        f_org_proj = model(images_org, adv=False)
        f1_proj = model(x1, adv=False)
        f2_proj = model(x2, adv=False)
        f_high_proj = model(x_HFC, adv=False)
        features = torch.cat([f_org_proj.unsqueeze(1), f1_proj.unsqueeze(1), f2_proj.unsqueeze(1), f_high_proj.unsqueeze(1)], dim=1)
        benign_loss = args.benign_w * criterion(features)
        
    benign_loss.backward()
    optimizer.second_step(zero_grad=True, g_other=adv_grads)
    
    # warm-up learning rate
    if epoch < args.warm_epochs:
        for param_group in optimizer.param_groups:
            param_group['lr'] = warmup_lr(
                args.learning_rate, epoch, args.warm_epochs, args.epochs,
                minibatch_idx, len_loader, args.eta_min
            )
        for param_group in d_optimizer.param_groups:
            param_group['lr'] = warmup_lr(
                args.learning_rate_d, epoch-args.start_from, args.warm_epochs, args.epochs-args.start_from,
                minibatch_idx, len_loader, args.eta_min
            )

    return adv_loss.item()