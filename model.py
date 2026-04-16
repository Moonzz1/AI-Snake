import torch, torch.nn as nn

class DuelingQNet(nn.Module):
    def __init__(self, input_size=14, hidden=256, output_size=3):
        super().__init__()
        # Tronc commun
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU()
        )
        # Branche valeur : "cet état est-il bon ?"
        self.value = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1)       # → 1 scalaire V(s)
        )
        # Branche avantage : "cette action est-elle meilleure que les autres ?"
        self.advantage = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, output_size)  # → 3 valeurs A(s,a)
        )

    def forward(self, x):
        shared = self.shared(x)
        v = self.value(shared)
        a = self.advantage(shared)
        # Formule de combinaison (Wang et al. 2016)
        return v + (a - a.mean(dim=1, keepdim=True))