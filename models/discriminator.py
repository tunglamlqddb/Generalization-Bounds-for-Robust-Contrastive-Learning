import torch

class Discriminator(torch.nn.Module):
    '''
    Just a simple Linear + Sigmoid.
    Returns probability of the data being from an adversarial distribution.
    '''
    def __init__(self, feature_dim=128, num_layer=0) -> None:
        super().__init__()
        if num_layer==0:
            self.d = torch.nn.Sequential(
                torch.nn.Linear(feature_dim, 1),
                torch.nn.Sigmoid()
            )
        elif num_layer==1:
            self.d = torch.nn.Sequential(
                torch.nn.Linear(feature_dim, feature_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(feature_dim, 1),
                torch.nn.Sigmoid()
            )

    def forward(self, x):
        return self.d(x)
