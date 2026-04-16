from per_memory import PrioritizedMemory
import torch, torch.nn as nn, numpy as np, random
from model import DuelingQNet

class DQNAgent:
    def __init__(self):
        self.memory  = PrioritizedMemory(capacity=100_000, alpha=0.6)
        self.epsilon = 1.0
        self.gamma   = 0.9
        self.model   = DuelingQNet()
        self.target  = DuelingQNet()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 2)
        state_t = torch.FloatTensor(state).unsqueeze(0)
        return self.model(state_t).argmax().item()

    def remember(self, state, action, reward, next_state, done):
        # ✅ Même interface qu'avant
        self.memory.remember(state, action, reward, next_state, done)

    def train(self, batch_size=64):
        if len(self.memory) < batch_size:
            return

        # ✅ sample() retourne maintenant aussi les indices et les poids
        batch, indices, weights = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.FloatTensor(np.array(states))
        next_states = torch.FloatTensor(np.array(next_states))
        rewards     = torch.FloatTensor(rewards)
        dones       = torch.BoolTensor(dones)
        weights     = torch.FloatTensor(weights)

        # Double DQN
        best_actions = self.model(next_states).argmax(1, keepdim=True)
        max_next_q   = self.target(next_states).gather(1, best_actions).squeeze(1)
        target_q     = rewards + self.gamma * max_next_q * (~dones)

        current_q = self.model(states).gather(1, torch.LongTensor(actions).unsqueeze(1)).squeeze(1)

        # ✅ TD errors pour mise à jour des priorités
        td_errors = (target_q - current_q).detach().cpu().numpy()

        # ✅ Loss pondérée par importance sampling
        loss = (weights * nn.MSELoss(reduction='none')(current_q, target_q.detach())).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ✅ Met à jour les priorités dans le SumTree
        self.memory.update_priorities(indices, td_errors)

        self.epsilon = max(0.005, self.epsilon * 0.9995)