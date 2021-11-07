import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, in_feat, hid_feat=None, out_feat=None,
                 dropout=0.5):
        super().__init__()

        if not hid_feat:
            hid_feat = in_feat
        if not out_feat:
            out_feat = in_feat
        self.fc1 = nn.Linear(in_feat, hid_feat)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hid_feat, out_feat)
        self.droprateout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x).to('cuda')
        x = self.act(x).to('cuda')
        x = self.fc2(x).to('cuda')
        return self.droprateout(x).to('cuda')


class Discriminator(nn.Module):
    def __init__(self, in_feat, hid_feat=None, out_feat=None, num_classes=1, dropout=0.):
        super().__init__()

        if not hid_feat:
            hid_feat = in_feat
        if not out_feat:
            out_feat = in_feat
        self.fc1 = nn.Linear(in_feat, hid_feat)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hid_feat, out_feat)
        self.droprateout = nn.Dropout(dropout)
        self.regressor = nn.Linear(out_feat, num_classes)

    def forward(self, x):
        x = self.fc1(x).to('cuda')
        x = self.act(x).to('cuda')
        x = self.fc2(x).to('cuda')
        x = self.droprateout(x).to('cuda')
        return self.regressor(x).to('cuda')
