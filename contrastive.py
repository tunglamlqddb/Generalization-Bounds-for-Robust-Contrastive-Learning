from utils.utils import reproducibility
# reproducibility()

from models import ResNetSSL, discriminator
from models.discriminator import Discriminator
from utils.sam import SAM
from utils.utils import (
    enable_running_stats, disable_running_stats, AverageMeter, save_model,
    warmup_lr, cosine_lr
)
from args.contrastive import get_args, get_save_folder
from datasets import get_dataset
from losses import infonce, SupConLoss, nt_xent
from forwards import standard as simclr
from functools import partial
import os, sys, time, torch
from forwards.minibatch_func import update_adv_no_d, update_adv_d, update_adv_cl, update_adv_cl_full
from datetime import datetime
import logging


def train(trainloader, model, criterion, forward_fn, optimizer, epoch):
    '''
    @input forward_fn   : returns the loss. could do all sorts of funny
        augmentation inside.
    Vanilla SSL with SAM
    '''
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(trainloader):
        data_time.update(time.time() - end)

        # warm-up learning rate
        if epoch < args.warm_epochs:
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr(
                    args.learning_rate, epoch, args.warm_epochs, args.epochs,
                    idx, len(trainloader), args.eta_min
                )

        images = torch.cat([images[0], images[1]], dim=0)
        images = images.to(args.device, non_blocking=True)
        labels = labels.to(args.device, non_blocking=True)
        bsz = labels.shape[0] # will this work with STL-10?

        # compute loss
        if args.sam: enable_running_stats(model) # the if not really needed
        loss = forward_fn(images, model, criterion)

        # update metric
        losses.update(loss.item(), bsz)

        if args.sam:
            # first forward-backward pass 
            loss.backward()
            optimizer.first_step(zero_grad=True)

            # second forward-backward pass
            disable_running_stats(model)
            forward_fn(images, model, criterion).backward()
            optimizer.second_step(zero_grad=True)
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % args.print_freq == 0:
            logger.debug('Train: [{0}][{1}/{2}]\t'
                'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                epoch, idx + 1, len(trainloader), batch_time=batch_time,
                data_time=data_time, loss=losses))
            sys.stdout.flush()


def train_adv_no_d(trainloader, model, criterion, optimizer, epoch):
    '''
    Adv training with SAM on benign samples
    '''
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    benign_losses = AverageMeter()
    adv_losses = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(trainloader):
        data_time.update(time.time() - end)

        benign_loss, adv_loss = update_adv_no_d(images, labels, model, criterion, optimizer, args.sam, args, epoch, idx, len(trainloader))
        losses.update(benign_loss + adv_loss, labels.shape[0])
        benign_losses.update(benign_loss, labels.shape[0])
        adv_losses.update(adv_loss, labels.shape[0])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % args.print_freq == 0:
            logger.debug('Train: [{0}][{1}/{2}]\t'
                'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                'benign {b_loss.val:.3f} ({b_loss.avg:.3f})\t'
                'adv {a_loss.val:.3f} ({a_loss.avg:.3f})\t'.format(
                epoch, idx + 1, len(trainloader), batch_time=batch_time,
                data_time=data_time, loss=losses, b_loss=benign_losses, a_loss=adv_losses))
            sys.stdout.flush()


def train_adv_d(trainloader, model, discriminator, criterion, optimizer, d_optimizer, d_w, d_step, epoch):

    model.train()
    discriminator.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    benign_losses = AverageMeter()
    adv_losses = AverageMeter()
    d_losses = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(trainloader):
        data_time.update(time.time() - end)

        benign_loss, adv_loss, d_loss =  \
            update_adv_d(images, labels, model, discriminator, criterion, optimizer, d_optimizer, d_w, d_step, args, epoch, idx, len(trainloader))
        
        losses.update(benign_loss + adv_loss, labels.shape[0])
        benign_losses.update(benign_loss, labels.shape[0])
        adv_losses.update(adv_loss, labels.shape[0])
        d_losses.update(d_loss, labels.shape[0])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % args.print_freq == 0:
            logger.debug('Train: [{0}][{1}/{2}]\t'
                'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                'benign {b_loss.val:.3f} ({b_loss.avg:.3f})\t'
                'adv {a_loss.val:.3f} ({a_loss.avg:.3f})\t'
                'd {d_loss.val:.3f} ({d_loss.avg:.3f})\t'.format(
                epoch, idx + 1, len(trainloader), batch_time=batch_time,
                data_time=data_time, loss=losses, b_loss=benign_losses, a_loss=adv_losses, d_loss=d_losses))
            sys.stdout.flush()


def train_adv_cl(trainloader, model, criterion, optimizer, epoch, discriminator=None, d_optimizer=None, d_w=1.0):   # for the baseline AdvCL
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    end = time.time()
    for idx, (images, labels) in enumerate(trainloader):
        data_time.update(time.time() - end)
        if discriminator is None:
            loss = update_adv_cl(images, labels, model, criterion, optimizer, args, epoch, idx, len(trainloader))
        else:
            loss = update_adv_cl_full(images, labels, model, discriminator, criterion, optimizer, d_optimizer, d_w, args, epoch, idx, len(trainloader))
        losses.update(loss, labels.shape[0])
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % args.print_freq == 0:
            logger.debug('Train: [{0}][{1}/{2}]\t'
                'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'loss {loss.val:.3f} ({loss.avg:.3f})\t'.format(
                epoch, idx + 1, len(trainloader), batch_time=batch_time,
                data_time=data_time, loss=losses))
            sys.stdout.flush()


if __name__ == '__main__':
    global args
    args = get_args()
    reproducibility(args.seed)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    stdout_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(stdout_handler)

    # if loads checkpoint, we override all settings
    if args.ckpt != '' and not args.discr:
        ckpt = torch.load(args.ckpt)
        ckpt['args'].ckpt = args.ckpt
        ckpt['args'].epochs = args.epochs
        ckpt['args'].exp_id = args.exp_id
        ckpt['args'].run_eval_bn = args.run_eval_bn
        args = ckpt['args']
        get_save_folder(args)

    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d_%H:%M:%S")
    output_file_handler = logging.FileHandler(args.save_folder+'/log_'+dt_string+".log")
    logger.addHandler(output_file_handler)
    logger.debug(args.save_folder)
    logger.debug(args)

    num_classes, size, trainloader = get_dataset(
        dataset=args.dataset, bsz=args.batch_size, root=args.data_folder,
        num_workers=args.num_workers, cj_str=args.cj_str, original_at=args.original_at
    )

    # build model
    model = ResNetSSL(name=args.model_arch, num_classes=num_classes, full=size>32).to(args.device)
    model.set_mode('ssl')
    if args.discr:
        discriminator = Discriminator(feature_dim=128, num_layer=args.d_num_layer).to(args.device)
    else:
        discriminator = None

    if args.device != 'cpu':
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True

    # build optimizer
    if args.sam:
        if args.optim == 'SGD':
            optimizer = SAM(model.parameters(), torch.optim.SGD, lr=args.learning_rate,
                momentum=args.momentum, weight_decay=args.weight_decay, rho=args.rho, adaptive=args.adaptive)
        elif args.optim == 'Adam':
            optimizer = SAM(model.parameters(), torch.optim.Adam, lr=args.learning_rate,
                weight_decay=args.weight_decay, rho=args.rho, adaptive=args.adaptive)
    else:
        if args.optim == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.optim == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                         weight_decay=args.weight_decay)

    if args.discr:
        if args.d_optim == 'SGD':
            d_optimizer = torch.optim.SGD(discriminator.parameters(), lr=args.learning_rate_d,
                momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.d_optim == 'Adam':
            d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate_d)
    else:
        d_optimizer = None
        
    # load checkpoints
    if args.ckpt != '':
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if args.discr:
            discriminator.load_state_dict(ckpt['discriminator'])
            d_optimizer.load_state_dict(ckpt['d_optimizer'])
        starting_epoch = ckpt['epoch'] + 1
    else:
        starting_epoch = 1

    if args.strategy == 'advcl_baseline':
        criterion = SupConLoss(temperature=args.temp)
    else:
        if args.criterion_type == 'infonce':
            criterion = partial(infonce, temperature=args.temp)
        elif args.criterion_type == 'nt_xent':
            criterion = partial(nt_xent, t=args.temp)

    if args.scheduler == 'torch':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_annealing(step,
                                                    args.epochs,
                                                    1,  # since lr_lambda computes multiplicative factor
                                                    1e-6 / args.lr,
                                                    warmup_steps=10)
        )

    # training routine
    for epoch in range(starting_epoch, args.epochs + 1):
        
        # train for one epoch
        time1 = time.time()
        if args.strategy=='base':
            train(trainloader, model, criterion, simclr, optimizer, epoch)
        elif args.strategy=='adv':
            if not args.discr:
                train_adv_no_d(trainloader, model, criterion, optimizer, epoch)
            else:
                d_w = args.d_w
                train_adv_d(trainloader, model, discriminator, criterion, optimizer, d_optimizer, d_w, args.d_step, epoch)
        elif args.strategy == 'advcl_baseline':
            if not args.discr:
                train_adv_cl(trainloader, model, criterion, optimizer, epoch)
            else:
                train_adv_cl(trainloader, model, criterion, optimizer, epoch, discriminator, d_optimizer, args.d_w)
                
        time2 = time.time()
        logger.debug('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        if epoch % args.save_freq == 0:
            save_file = os.path.join(args.save_folder, f'epoch_{epoch}.pth')
            save_model(model, discriminator, optimizer, d_optimizer, args, epoch, save_file)
        
        # schedule: move to the end of epoch
        for param_group in optimizer.param_groups:
            param_group['lr'] = cosine_lr(args.learning_rate, epoch, args.epochs, args.eta_min)
        if args.discr:
            for param_group in d_optimizer.param_groups:
                param_group['lr'] = cosine_lr(args.learning_rate_d, epoch, args.epochs, args.eta_min)

    # save the last model
    save_file = os.path.join(args.save_folder, 'last.pth')
    save_model(model, discriminator, optimizer, d_optimizer, args, args.epochs, save_file)


    # linear
    # if args.run_eval:
    #     logger.debug("START LINEAR!!")
    #     cmd = "python linear.py --device cuda --batch_size 512 --learning_rate 0.1 --model_arch resnet18\
    #     --ckpt " + save_file
    #     os.system(cmd)
        
    if args.run_eval_bn:
        logger.debug("START LINEAR CORRECT BN!! NO AT")
        cmd = "python linear_correct_bn.py --device cuda --batch_size 512 --learning_rate " + str(args.learning_rate_linear) + " --model_arch " + str(args.model_arch) + " --dataset " \
        +str(args.dataset) + " --ckpt " + save_file + " --seed " + str(args.seed) + " --run_unseen_attacks " + str(args.run_unseen_attacks)
        os.system(cmd)
        
        logger.debug("START LINEAR CORRECT BN!! AT")
        cmd = "python linear_correct_bn.py --device cuda --batch_size 512 --learning_rate " + str(args.learning_rate_linear) + " --model_arch " + str(args.model_arch) + " --dataset " \
            + str(args.dataset)+ " --at --ckpt " + save_file + " --seed " + str(args.seed) + " --run_unseen_attacks " + str(args.run_unseen_attacks)
        os.system(cmd)

        logger.debug("START LINEAR CORRECT BN STl10!! NO AT")
        cmd = "python linear_correct_bn.py --device cuda --batch_size 512 --learning_rate " + str(args.learning_rate_linear) + " --model_arch " + str(args.model_arch) + " --dataset stl10" \
        + " --ckpt " + save_file + " --seed " + str(args.seed) + " --run_unseen_attacks " + str(args.run_unseen_attacks)
        os.system(cmd)

        logger.debug("START LINEAR CORRECT BN STL10!! AT")
        cmd = "python linear_correct_bn.py --device cuda --batch_size 512 --learning_rate " + str(args.learning_rate_linear) + " --model_arch " + str(args.model_arch) + " --dataset stl10" \
        + " --at --ckpt " + save_file + " --seed " + str(args.seed) + " --run_unseen_attacks " + str(args.run_unseen_attacks)
        os.system(cmd)
