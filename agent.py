from collections import deque
import random, torch, numpy as np
import torch.nn as nn
from model import DuelingQNet

class DQNAgent:
    def __init__(self):
        self.memory = deque(maxlen=100_000)
        self.epsilon = 1.0       # exploration initiale
        self.gamma   = 0.9       # discount factor
        self.lr      = 0.001
        self.model   = DuelingQNet()
        self.target  = DuelingQNet()    # réseau cible stable
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 2)  # exploration
        state_t = torch.FloatTensor(state).unsqueeze(0)
        return self.model(state_t).argmax().item()  # exploitation

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch_size=64):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.FloatTensor(np.array(states))
        next_states = torch.FloatTensor(np.array(next_states))
        rewards     = torch.FloatTensor(rewards)
        dones       = torch.BoolTensor(dones)

        current_q  = self.model(states).gather(1, torch.LongTensor(actions).unsqueeze(1))
        with torch.no_grad():
            # 1. Le modèle principal choisit la meilleure action
            best_actions = self.model(next_states).argmax(1, keepdim=True)
            # 2. Le réseau cible évalue cette action (plus stable)
            max_next_q   = self.target(next_states).gather(1, best_actions).squeeze(1)

            target_q = rewards + self.gamma * max_next_q * (~dones)

        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Décroissance de l'exploration
        self.epsilon = max(0.01, self.epsilon * 0.9995)