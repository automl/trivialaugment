import torch
from torch import nn

class SeqConvNet(nn.Module):
    def __init__(self,D_out,fixed_dropout=None,in_channels=3,channels=(64,64),h_dims=(200,100),adaptive_dropout_creator=None,batch_norm=False):
        super().__init__()
        print("Using SeqConvNet")
        assert len(channels) == 2 == len(h_dims)
        pool = lambda: nn.MaxPool2d(2,2)
        dropout = lambda: torch.nn.Dropout(p=fixed_dropout)
        dropout_li = lambda: ([] if fixed_dropout is None else [dropout()])
        relu = lambda: torch.nn.ReLU(inplace=False)
        flatten = lambda l: [item for sublist in l for item in sublist]
        convs = [nn.Conv2d(in_channels, channels[0], 5),nn.Conv2d(channels[0], channels[1], 5)]
        fcs = [nn.Linear(channels[1] * 5 * 5, h_dims[0]),nn.Linear(h_dims[0], h_dims[1])]
        self.final_fc = nn.Linear(h_dims[1], D_out)
        self.conv_blocks = nn.Sequential(*flatten([[conv,relu(),pool()] + dropout_li() for conv in convs]))
        self.bn = nn.BatchNorm1d(h_dims[1], momentum=.9) if batch_norm else nn.Identity()
        self.fc_blocks = nn.Sequential(*flatten([[fc,relu()] + dropout_li() for fc in fcs]))
        self.adaptive_dropouters = [adaptive_dropout_creator(h_dims[1])] if adaptive_dropout_creator is not None else []

    def forward(self, x):
        x = self.conv_blocks(x)
        x = torch.nn.Flatten()(x)
        x = self.fc_blocks(x)
        if self.adaptive_dropouters:
            x = self.adaptive_dropouters[0](x)
        x = self.bn(x)
        x = self.final_fc(x)
        return x

