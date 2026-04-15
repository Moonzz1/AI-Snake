import torch, torch.nn as nn

class QNet(nn.Module):
    def __init__(self, input_size=11, hidden=256, output_size=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_size)
        )

    def forward(self, x):
        return self.net(x)