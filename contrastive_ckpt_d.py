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
from losses import infonce
from forwards import standard as simclr
from functools import partial
import os, sys, time, torch
from forwards.minibatch_func import update_adv_no_d, update_adv_d
from datetime import datetime
import logging, copy


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


if __name__ == '__main__':
    global args
    args = get_args()
    reproducibility(args.seed)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    stdout_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(stdout_handler)

    # load ckpt and add values of D to old args
    # logger.debug("Args before merged:", args)
    if args.ckpt != '':
        ckpt = torch.load(args.ckpt)
        ckpt['args'].ckpt = args.ckpt
        args_old = ckpt['args']
        
        args_old.exp_id = args.exp_id
        args_old.epochs = args.epochs        
        args_old.run_eval = args.run_eval    
        args_old.run_eval_bn = args.run_eval_bn
        args_old.discr = args.discr
        args_old.d_num_layer = args.d_num_layer
        args_old.learning_rate_d = args.learning_rate_d
        args_old.d_w = args.d_w
        args_old.d_step = args.d_step
        args_old.use_d_gen = args.use_d_gen
        args_old.start_from = args.start_from
        args_old.d_reduction = args.d_reduction

        # NOTE: add this one to take rho of new args
        args_old.rho = args.rho


        # new_args_bk = copy.deepcopy(args)
        args = args_old
        get_save_folder(args)
    # logger.debug("Args after merged:", args)
    
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
        num_workers=args.num_workers, cj_str=args.cj_str
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
        optimizer = SAM(model.parameters(), torch.optim.SGD, lr=args.learning_rate,
            momentum=args.momentum, weight_decay=args.weight_decay, rho=args.rho, adaptive=args.adaptive)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
            momentum=args.momentum, weight_decay=args.weight_decay)
    if args.discr:
        d_optimizer = torch.optim.SGD(discriminator.parameters(), lr=args.learning_rate_d,
            momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        d_optimizer = None
        
    # load checkpoints
    if args.ckpt != '':
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        starting_epoch = ckpt['epoch'] + 1
    else:
        starting_epoch = 1

    criterion = partial(infonce, temperature=args.temp)

    # training routine
    for epoch in range(starting_epoch, args.epochs + 1):
        for param_group in optimizer.param_groups:
            param_group['lr'] = cosine_lr(args.learning_rate, epoch, args.epochs, args.eta_min)
        if args.discr:
            for param_group in d_optimizer.param_groups:
                param_group['lr'] = cosine_lr(args.learning_rate_d, epoch-args.start_from, args.epochs-args.start_from, args.eta_min)
        
        # train for one epoch
        time1 = time.time()
        
        d_w = args.d_w
        train_adv_d(trainloader, model, discriminator, criterion, optimizer, d_optimizer, d_w, args.d_step, epoch)
                
        time2 = time.time()
        logger.debug('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        if epoch % args.save_freq == 0:
            save_file = os.path.join(args.save_folder, f'epoch_{epoch}.pth')
            save_model(model, discriminator, optimizer, d_optimizer, args, epoch, save_file)

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
        +str(args.dataset) + " --ckpt " + save_file + " --seed " + str(args.seed)
        os.system(cmd)
        
        logger.debug("START LINEAR CORRECT BN!! AT")
        cmd = "python linear_correct_bn.py --device cuda --batch_size 512 --learning_rate " + str(args.learning_rate_linear) + " --model_arch " + str(args.model_arch) + " --dataset " \
            + str(args.dataset)+ " --at --ckpt " + save_file + " --seed " + str(args.seed)
        os.system(cmd)