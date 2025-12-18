from .base import get_phase1_transform, get_phase2_transform, get_eval_transform, get_dataloader
from torchvision.datasets import CIFAR10

mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)
size = 32
num_classes = 10

def load(mode='ssl', train=True, bsz=512, root='./data', num_workers=8, cj_str=0.5, original_at='default'):
    '''
    @input mode         : either 'ssl' or 'linear'
    @input train        : whether we are loading training or testing data
    @input bsz          : batch size
    @input root         : root folder of data
    @input num_workers  : number of dataloader workers
    @input cj_str       : color jitter strength (for the SSL phase)
    '''
    if mode == 'ssl':
        dataset = CIFAR10(root=root, download=True, train=True,
            transform=get_phase1_transform(size, mean, std, cj_str, original_at=original_at)
    )
    elif train:
        dataset = CIFAR10(root=root, download=True, train=True,
            transform=get_phase2_transform(size, mean, std)
    )
    else:
        dataset = CIFAR10(root=root, download=True, train=False,
            transform=get_eval_transform(size, mean, std))

    dataloader = get_dataloader(dataset, bsz, shuffle=train, num_workers=num_workers)
    return dataloader