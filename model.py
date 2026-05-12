# model.py

import torch, torch.nn as nn

class DuelingQNet(nn.Module):
    """
    Dueling DQN avec Batch Normalization et Dropout pour une meilleure généralisation.
    input_size doit correspondre au vecteur d'état dans snake_env._get_state().
    """

    def __init__(self, input_size=20, hidden=256, output_size=3):
        super().__init__()
        # Tronc commun — BN après activation (pattern post-BN)
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(p=0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
        )

        # Branche valeur V(s)
        self.value = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1)
        )

        # Branche avantage A(s, a)
        self.advantage = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, output_size)
        )

    def forward(self, x):
        shared = self.shared(x)
        v = self.value(shared)
        a = self.advantage(shared)
        # Combinaison Wang et al. 2016 : Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
        return v + (a - a.mean(dim=1, keepdim=True))