from .base import get_phase1_transform, get_phase2_transform, get_eval_transform, get_dataloader
from torchvision import datasets


mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
size = 64
num_classes = 200

def load(mode='ssl', train=True, bsz=512, root='./data', num_workers=8, cj_str=0.5, original_at='default'):
    print("Train on TinyImageNet!!")
    '''
    @input mode         : either 'ssl' or 'linear'
    @input train        : whether we are loading training or testing data
    @input bsz          : batch size
    @input root         : root folder of data
    @input num_workers  : number of dataloader workers
    @input cj_str       : color jitter strength (for the SSL phase)
    '''
    if mode == 'ssl':   
        dataset = datasets.ImageFolder(root=root + '/tiny-imagenet-200/train/', transform=get_phase1_transform(size, mean, std, cj_str, blur_size=5, original_at=original_at))
    elif train:
        dataset = datasets.ImageFolder(root=root + '/tiny-imagenet-200/train/', transform=get_phase2_transform(size, mean, std))
    else:
        dataset = datasets.ImageFolder(root=root + '/tiny-imagenet-200/val/', transform=get_eval_transform(size, mean, std))

    dataloader = get_dataloader(dataset, bsz, shuffle=train, num_workers=num_workers)
    return dataloader


