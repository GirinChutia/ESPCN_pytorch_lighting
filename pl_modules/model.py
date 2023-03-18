import torch
import torch.nn as nn

def init_weights(m,name='orthogonal'):
    if type(m) == nn.Conv2d:
        if name == 'xavier':
            torch.nn.init.xavier_uniform_(m.weight,
                                         gain=nn.init.calculate_gain('tanh'))
        if name == 'kaiming':
            torch.nn.init.kaiming_uniform_(m.weight)
        if name == 'orthogonal': 
            torch.nn.init.orthogonal_(m.weight,
                                      gain=nn.init.calculate_gain('tanh'))
        m.bias.data.fill_(0.01)
             
class ESPCN_model(nn.Module):
    def __init__(self, scale : int, channels : int=3) -> None:
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels=channels, out_channels=64, 
                                kernel_size=5, padding=2)
        self.conv_2 = nn.Conv2d(in_channels=64, out_channels=32,
                                kernel_size=3, padding=1)
        self.conv_3 = nn.Conv2d(in_channels=32, 
                                out_channels=(channels * scale * scale), kernel_size=3, padding=1)
        self.tanh = nn.Tanh()
        self.pixel_shuffle = nn.PixelShuffle(scale)

    def forward(self, X_in):
        X = self.tanh(self.conv_1(X_in))
        X = self.tanh(self.conv_2(X))
        X = self.conv_3(X)
        X = self.pixel_shuffle(X)
        return X