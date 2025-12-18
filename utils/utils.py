import math
import torch
from torch.nn.modules.batchnorm import _BatchNorm
import random, copy


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


# these are needed for SAM
def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)


def get_grad(param_groups):    
    grads = copy.deepcopy(param_groups)
    for g_idx,group in enumerate(param_groups):
        for p_idx,p in enumerate(group["params"]):
            if p.grad is None: continue
            grads[g_idx][p_idx] = p.grad.data.clone()
    return grads

# fix all randomizers
def reproducibility(seed=0x6e676f63746e71):
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_model(model, discriminator, optimizer, d_optimizer, args, epoch, save_file):
    torch.save({
        'args': args,
        'epoch': epoch,
        'model': model.state_dict(),
        'discriminator': discriminator.state_dict() if discriminator is not None else None,
        'optimizer': optimizer.state_dict(),
        'd_optimizer':d_optimizer.state_dict() if d_optimizer is not None else None,
    }, save_file)


# Linear warmup + Cosine LR scheduler
def cosine_lr(base_lr, curr_epoch, total_epoch, eta_min=1e-4):
    return eta_min + (base_lr - eta_min) * (1 + math.cos(math.pi * curr_epoch / total_epoch)) / 2


def warmup_lr(base_lr, curr_epoch, warm_epoch, total_epoch, batch_id, total_batches, eta_min=1e-4):
    if curr_epoch <= warm_epoch:
        end_lr = cosine_lr(base_lr, warm_epoch, total_epoch, eta_min)
        p = (batch_id + (curr_epoch - 1) * total_batches) / (warm_epoch * total_batches)
        return eta_min + p * (end_lr - eta_min)


