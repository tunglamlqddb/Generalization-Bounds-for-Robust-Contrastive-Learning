'''
IMPORTANT: Input normalization disabled for ease of attack.
'''

import torch
from torchvision import transforms


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform, phase2_transform, original_at='default'):
        self.transform = transform
        self.phase2_transform = phase2_transform
        self.original_at = original_at

    def __call__(self, x):
        if self.original_at == 'default':
            return [self.transform(x), self.transform(x), self.phase2_transform(x)]    # return original x here for AdvCL
        elif self.original_at == 'first':
            return [self.phase2_transform(x), self.transform(x), self.phase2_transform(x)]
        elif self.original_at == 'second':
            return [self.transform(x), self.phase2_transform(x), self.phase2_transform(x)]


def get_phase1_transform(size, mean, std, s, blur_size=3, original_at='default'):

    base_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=int(blur_size))
        ], p=0.5),
        transforms.ToTensor(),
        # transforms.Normalize(mean=mean, std=std),
    ])
    
    phase2_transform = get_phase2_transform(size, mean, std)

    return TwoCropTransform(base_transform, phase2_transform, original_at)


def get_phase2_transform(size, mean, std):
    return transforms.Compose([
        # transforms.RandomResizedCrop(size=size, scale=(0.2, 1.)),
        transforms.Resize(size=size),
        transforms.RandomCrop(size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=mean, std=std),
    ])


def get_eval_transform(size, mean, std):
    return transforms.Compose([
        transforms.Resize(size),
        # transforms.CenterCrop(size),
        transforms.ToTensor(),
        # transforms.Normalize(mean=mean, std=std),
    ])


def get_dataloader(dataset, bsz, shuffle=False, num_workers=8):
    return torch.utils.data.DataLoader(
        dataset, batch_size=bsz, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True)
