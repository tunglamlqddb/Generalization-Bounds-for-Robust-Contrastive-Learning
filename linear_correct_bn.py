from utils.utils import reproducibility
# reproducibility()

from models import ResNetSSL
from utils.sam import SAM
from utils.utils import (
    enable_running_stats, disable_running_stats, AverageMeter, save_model,
    cosine_lr, accuracy
)
from args.linear import get_args
from datasets import get_dataset
import sys, time, torch, logging
from attacks import pgd_attack
import torchattacks


def train(trainloader, model, criterion, optimizer, epoch, at=False, adv=False):   # train us nat_bn
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(trainloader):
        data_time.update(time.time() - end)

        images = images.to(args.device, non_blocking=True)
        labels = labels.to(args.device, non_blocking=True)
        bsz = labels.shape[0]

        if at:
            model.eval()
            attk_func = lambda x_adv: criterion(model(x_adv, adv=True), labels)
            x_adv = pgd_attack(images, attk_func, epsilon=0.0314, alpha=0.007, nb_iter=10)
            images = x_adv   #.to(args.device, non_blocking=True)
            model.train()
        
        # compute loss
        if args.sam: enable_running_stats(model) # the if not really needed
        output = model(images, disable_feature_grad=True, adv=at)
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, = accuracy(output, labels, topk=(1,))
        top1.update(acc1, bsz)

        if args.sam:
            loss.backward()
            optimizer.first_step(zero_grad=True)

            disable_running_stats(model)
            criterion(model(images, disable_feature_grad=True, adv=at), labels).backward()
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
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(trainloader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(testloader, model, criterion, adv=False):
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top1_adv_bn = AverageMeter()
    top5 = AverageMeter()
    robust_nat_bn = AverageMeter()
    robust_adv_bn = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(testloader):
            images = images.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)
            bsz = labels.shape[0]

            # forward
            output = model(images, adv=False)

            # update metric
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1, bsz)      #top1_clean_bn
            top5.update(acc5, bsz)

            acc1_adv_bn, _ = accuracy(model(images, adv=True), labels, topk=(1, 5))
            top1_adv_bn.update(acc1_adv_bn, bsz)

            attk_func = lambda x_adv: criterion(model(x_adv, adv=adv), labels)
            x_adv = pgd_attack(images, attk_func, epsilon=0.0314, alpha=0.00314, nb_iter=20)
            acc1r_nat_bn, = accuracy(model(x_adv, adv=False), labels)
            acc1r_adv_bn, = accuracy(model(x_adv, adv=True), labels)   
            robust_nat_bn.update(acc1r_nat_bn, bsz)
            robust_adv_bn.update(acc1r_adv_bn, bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % args.print_freq == 0:
                logger.debug('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Acc_nat@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc_adv@1 {top1_adv.val:.3f} ({top1_adv.avg:.3f})\t'
                      'AccR_nat@1 {top1r_nat.val:.3f} ({top1r_nat.avg:.3f})\t'
                      'AccR_adv@1 {top1r_adv.val:.3f} ({top1r_adv.avg:.3f})'.format(
                       idx, len(testloader), batch_time=batch_time, top1=top1, top1_adv=top1_adv_bn, top1r_nat=robust_nat_bn, top1r_adv=robust_adv_bn))

    logger.debug(' * Acc@1 {top1.avg:.3f}, Acc_adv@1 {top1_adv.avg:.3f}, AccR_nat@1 {top1r_nat.avg:.3f}, AccR_adv@1 {top1r_adv.avg:.3f}'.format(top1=top1, top1_adv=top1_adv_bn, top1r_nat=robust_nat_bn, top1r_adv=robust_adv_bn))
    return losses.avg, top1.avg, top1_adv_bn.avg, robust_nat_bn.avg, robust_adv_bn.avg


def validate_unseen(testloader, model, criterion, adv=False, name='ours', args=None):
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top1_adv_bn = AverageMeter()
    top5 = AverageMeter()
    robust_nat_bn = AverageMeter()
    robust_adv_bn = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(testloader):
        images = images.to(args.device, non_blocking=True)
        labels = labels.to(args.device, non_blocking=True)
        bsz = labels.shape[0]

        # forward
        output = model(images, adv=False)

        # update metric
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1, bsz)      #top1_clean_bn
        top5.update(acc5, bsz)

        acc1_adv_bn, _ = accuracy(model(images, adv=True), labels, topk=(1, 5))
        top1_adv_bn.update(acc1_adv_bn, bsz)

        x_adv = create_adv_for_test(model, images, labels, name=name, epsilon=0.0314, alpha=0.00314, nb_iter=20, args=args)
        acc1r_nat_bn, = accuracy(model(x_adv, adv=False), labels)
        acc1r_adv_bn, = accuracy(model(x_adv, adv=True), labels)   
        robust_nat_bn.update(acc1r_nat_bn, bsz)
        robust_adv_bn.update(acc1r_adv_bn, bsz)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % args.print_freq == 0:
            logger.debug('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Acc_nat@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Acc_adv@1 {top1_adv.val:.3f} ({top1_adv.avg:.3f})\t'
                    'AccR_nat@1 {top1r_nat.val:.3f} ({top1r_nat.avg:.3f})\t'
                    'AccR_adv@1 {top1r_adv.val:.3f} ({top1r_adv.avg:.3f})'.format(
                    idx, len(testloader), batch_time=batch_time, top1=top1, top1_adv=top1_adv_bn, top1r_nat=robust_nat_bn, top1r_adv=robust_adv_bn))

    logger.debug(' * Acc@1 {top1.avg:.3f}, Acc_adv@1 {top1_adv.avg:.3f}, AccR_nat@1 {top1r_nat.avg:.3f}, AccR_adv@1 {top1r_adv.avg:.3f}'.format(top1=top1, top1_adv=top1_adv_bn, top1r_nat=robust_nat_bn, top1r_adv=robust_adv_bn))
    return losses.avg, top1.avg, top1_adv_bn.avg, robust_nat_bn.avg, robust_adv_bn.avg



def create_adv_for_test(model, images, labels, name='ours', epsilon=0.0314, alpha=0.00314, nb_iter=20, args=None):
    if name == 'ours':
        attk_func = lambda x_adv: criterion(model(x_adv, adv=True), labels)
        return pgd_attack(images, attk_func, epsilon=0.0314, alpha=0.00314, nb_iter=20)
    elif name == 'l2':
        attack = torchattacks.PGDL2(model, eps=0.5, alpha=0.05, steps=20, random_start=True)
        return attack(images, labels)
    elif name == 'l1':
        pass
    elif name == 'autoattack':
        if args.dataset == 'cifar10': n_classes = 10
        elif args.dataset == 'cifar100': n_classes = 100
        elif args.dataset == 'tinyimagenet': n_classes = 200
        elif args.dataset == 'stl10': n_classes = 10
        attack = torchattacks.AutoAttack(model, norm='Linf', eps=8/255, version='standard', n_classes=n_classes, seed=None, verbose=False)
        return attack(images, labels)


if __name__ == '__main__':
    global args
    args = get_args()
    reproducibility(args.seed)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    stdout_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(stdout_handler)
    save_folder = args.ckpt.rsplit('/', 1)[0]
    output_file_handler = logging.FileHandler(save_folder+'/linear_correct_bn_at'+str(args.at)+'lr'+str(args.learning_rate)+'_'+str(args.dataset)+'.log')
    logger.addHandler(output_file_handler)
    logger.debug('-------------------')
    logger.debug(save_folder)
    logger.debug(args)

    if "stra_adv" in args.ckpt or "advcl_baseline" in args.ckpt: adv=True
    elif "stra_base" in args.ckpt: adv=False

    num_classes, size, trainloader = get_dataset(
        dataset=args.dataset, bsz=args.batch_size, root=args.data_folder,
        num_workers=args.num_workers, mode='linear', train=True
    )
    _, _, testloader = get_dataset(
        dataset=args.dataset, bsz=args.batch_size, root=args.data_folder,
        num_workers=args.num_workers, mode='linear', train=False
    )

    # build model and load weights
    model = ResNetSSL(name=args.model_arch, num_classes=num_classes, full=size>32).to(args.device)
    
    ckpt = torch.load(args.ckpt, map_location='cpu')
    if args.dataset not in args.ckpt:
        print(ckpt['model']['fc.weight'].shape)
        tmp_fc = torch.nn.Linear(ckpt['model']['fc.weight'].shape[1], num_classes)
        ckpt['model']['fc.weight'] = tmp_fc.weight
        ckpt['model']['fc.bias'] = tmp_fc.bias
        
    model.load_state_dict(ckpt['model'])
    model.set_mode('linear')

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

    criterion = torch.nn.CrossEntropyLoss()

    best_acc = (0, 0, 0, 0)
    ckpt_root = args.ckpt.rsplit('.', 1)
    best_ckpt_name = ckpt_root[0] + '_correct_bn_at' + str(args.at)+'lr'+str(args.learning_rate)+'_'+str(args.dataset) + '_best.' + ckpt_root[1]
    last_ckpt_name = ckpt_root[0] + '_correct_bn_at' + str(args.at)+'lr'+str(args.learning_rate)+'_'+str(args.dataset) + '_last.' + ckpt_root[1]
    

    # training routine
    for epoch in range(1, args.epochs + 1):
        for param_group in optimizer.param_groups:
            param_group['lr'] = cosine_lr(args.learning_rate, epoch, args.epochs, args.eta_min)

        # train for one epoch
        time1 = time.time()
        loss, acc = train(trainloader, model, criterion, optimizer, epoch, args.at, adv)
        time2 = time.time()
        logger.debug('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
            epoch, time2 - time1, acc))

        # eval for one epoch
        _, acc1, acc1_adv, acc_adv_nat_bn, acc_adv_adv_bn = validate(testloader, model, criterion, adv)
        if acc_adv_adv_bn > best_acc[3]:
            best_acc = (acc1, acc1_adv, acc_adv_nat_bn, acc_adv_adv_bn)
            save_model(model, None, optimizer, None, args, epoch, best_ckpt_name)

    logger.debug('[+] Best accuracy:')
    logger.debug(f'Top 1 nat: {best_acc[0]:.2f}')
    logger.debug(f'Top 1 adv: {best_acc[1]:.2f}')
    logger.debug(f'Robust_nat: {best_acc[2]:.2f}')
    logger.debug(f'Robust_adv: {best_acc[3]:.2f}')
    save_model(model, None, optimizer, None, args, epoch, last_ckpt_name) # epoch shouldn't have gone out of scope yet

    if args.run_unseen_attacks:
        # change default_bn to True as torchattacks lib cannot pass adv=True
        model.set_default_bn(True)

        model.load_state_dict(torch.load(last_ckpt_name)['model'])
        model = model.cuda()
        model.set_mode('linear')
        _, acc1, acc1_adv, acc_adv_nat_bn, acc_adv_adv_bn = validate_unseen(testloader, model, criterion, adv, name='l2', args=args)
        logger.debug('[+] Last accuracy:')
        logger.debug(f'L2 Top 1 nat: {acc1:.2f}')
        logger.debug(f'L2 Top 1 adv: {acc1_adv:.2f}')
        logger.debug(f'L2 Robust_nat: {acc_adv_nat_bn:.2f}')
        logger.debug(f'L2 Robust_adv: {acc_adv_adv_bn:.2f}')
        
        _, acc1, acc1_adv, acc_adv_nat_bn, acc_adv_adv_bn = validate_unseen(testloader, model, criterion, adv, name='autoattack', args=args)
        logger.debug('[+] Last accuracy:')
        logger.debug(f'Auto Top 1 nat: {acc1:.2f}')
        logger.debug(f'Auto Top 1 adv: {acc1_adv:.2f}')
        logger.debug(f'Auto Robust_nat: {acc_adv_nat_bn:.2f}')
        logger.debug(f'Auto Robust_adv: {acc_adv_adv_bn:.2f}')
    
        # return default_bn to False 
        model.set_default_bn(False)
