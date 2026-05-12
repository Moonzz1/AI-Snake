# agent.py

import torch
import torch.nn as nn
import numpy as np
import random
from per_memory import PrioritizedMemory
from model import DuelingQNet

STATE_SIZE = 20  # correspond au vecteur dans snake_env._get_state()

class DQNAgent:
    def __init__(self):
        self.memory    = PrioritizedMemory(capacity=100_000, alpha=0.6)
        self.epsilon   = 1.0
        self.eps_min   = 0.01
        self.eps_decay = 0.9997     # décroissance plus lente → exploration plus longue
        self.gamma     = 0.95       # horizon temporel plus grand (était 0.9)

        self.model  = DuelingQNet(input_size=STATE_SIZE)
        self.target = DuelingQNet(input_size=STATE_SIZE)
        self.target.load_state_dict(self.model.state_dict())
        self.target.eval()

        # Adam + scheduler cosinus pour stabiliser en fin d'entraînement
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=1e-5
        )

        # Huber loss (moins sensible aux grands TD-errors que MSE)
        self.loss_fn = nn.SmoothL1Loss(reduction='none')

        self._steps = 0     # compteur global de steps pour le target update soft

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 2)
        self.model.eval()
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            action  = self.model(state_t).argmax().item()
        self.model.train()
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.remember(state, action, reward, next_state, done)

    def train(self, batch_size=128):   # batch plus grand = gradients plus stables
        if len(self.memory) < batch_size:
            return

        batch, indices, weights = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.FloatTensor(np.array(states))
        next_states = torch.FloatTensor(np.array(next_states))
        rewards     = torch.FloatTensor(rewards)
        dones       = torch.BoolTensor(dones)
        weights     = torch.FloatTensor(weights)
        actions_t   = torch.LongTensor(actions).unsqueeze(1)

        # Double DQN : l'online network choisit l'action, le target l'évalue
        self.model.eval()
        with torch.no_grad():
            best_actions = self.model(next_states).argmax(1, keepdim=True)
        self.model.train()

        with torch.no_grad():
            max_next_q = self.target(next_states).gather(1, best_actions).squeeze(1)

        target_q  = rewards + self.gamma * max_next_q * (~dones)
        current_q = self.model(states).gather(1, actions_t).squeeze(1)

        # TD-errors → mise à jour des priorités PER
        td_errors = (target_q - current_q).detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)

        # Huber loss pondérée (importance sampling PER)
        loss = (weights * self.loss_fn(current_q, target_q.detach())).mean()

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping — évite les explosions de gradient
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Décroissance epsilon
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

        # Soft update du target network (τ=0.005) — plus stable que le hard update
        self._soft_update(tau=0.005)
        self._steps += 1

    def _soft_update(self, tau=0.005):
        """Mélange progressif : θ_target ← τ·θ_online + (1-τ)·θ_target"""
        for tp, op in zip(self.target.parameters(), self.model.parameters()):
            tp.data.copy_(tau * op.data + (1 - tau) * tp.data)

    def step_scheduler(self):
        """Appeler à la fin de chaque épisode."""
        self.scheduler.step()

    def save(self, path="best_model.pth"):
        torch.save({
            'model': self.model.state_dict(),
            'target': self.target.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self._steps,
        }, path)

    def load(self, path="best_model.pth"):
        ckpt = torch.load(path, map_location='cpu')
        self.model.load_state_dict(ckpt['model'])
        self.target.load_state_dict(ckpt['target'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.epsilon = ckpt.get('epsilon', self.eps_min)
        self._steps  = ckpt.get('steps', 0)