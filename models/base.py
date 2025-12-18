import torch
from torch.nn.functional import normalize


class GeneralSSL(torch.nn.Module):
    def __init__(self):
        super(GeneralSSL, self).__init__()
        self.mode = 'ssl' # or 'linear'
        self.default_bn = False

    def set_mode(self, mode):
        assert mode in ['ssl', 'linear'], "Invalid mode!"
        self.mode = mode
        self.train(self.training)

    def set_default_bn(self, mode):
        assert mode in [False, True], "Invalid mode!"
        self.default_bn = mode
    

    def train(self, mode=True):
        self.training = mode

        if mode:
            if self.mode == 'ssl':
                self.encoder.train()
                self.head.train()
                self.fc.eval()
            elif self.mode == 'linear':
                self.encoder.eval()
                self.head.eval()
                self.fc.train()
        else:
            self.encoder.eval()
            self.head.eval()
            self.fc.eval()


    def forward(self, x, return_both=False, disable_feature_grad=False, adv=None):    # NOTE to change this to default
        '''
        @input return_both:             returns both the intermediate features and the
            projected embedding/classification
        @input disable_feature_grad:    run torch.no_grad() on encoder part
            (feature extraction) to speed up computation
        @input adv:                     whether to use the adversarial BN branch
        '''

        if adv is None:
            adv = self.default_bn
            
        with (torch.no_grad if disable_feature_grad else torch.enable_grad)():
            # convenient hack with the normalizing, but should improve training
            feat_enc = normalize(self.encoder(x, adv=adv), dim=1)

        if self.mode == 'ssl':
            feat = normalize(self.head(feat_enc), dim=1)
        else:
            feat = self.fc(feat_enc)

        if return_both:
            return feat_enc, feat
        return feat


class ParamSequential(torch.nn.Sequential):
    def forward(self, *inputs, **kwargs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs, **kwargs)
            else:
                inputs = module(inputs, **kwargs)
        return inputs


def normalize_fn(tensor, mean, std):
    """Differentiable version of torchvision.functional.normalize"""
    # here we assume the color channel is in at dim=1
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)
    

class NormalizeByChannelMeanStd(torch.nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)