# train.py

import pygame
import torch
from collections import deque
import numpy as np
from snake_env import SnakeEnv
from agent import DQNAgent

# ────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────
EPISODES     = 2000
HEADLESS     = False  # True = pas de fenêtre, entraînement plus rapide
RENDER_EVERY = 50     # Affiche 1 épisode sur RENDER_EVERY en mode HEADLESS

env   = SnakeEnv(headless=HEADLESS)
agent = DQNAgent()
record = 0

# Fenêtre glissante sur 100 épisodes pour des statistiques stables
score_window = deque(maxlen=100)

for episode in range(1, EPISODES + 1):
    state = env.reset()
    done  = False
    total_reward = 0

    while not done:
        # Vider la file d'événements Pygame (ne pas bloquer)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                agent.save("best_model.pth")
                pygame.quit()
                quit()

        action                    = agent.choose_action(state)
        next_state, reward, done  = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        agent.train()
        state        = next_state
        total_reward += reward

        # Rendu : toujours si non-headless, sinon 1 fois sur RENDER_EVERY
        if not HEADLESS or episode % RENDER_EVERY == 0:
            env.render()

    # Fin d'épisode
    agent.step_scheduler()
    score_window.append(env.score)
    avg_score = np.mean(score_window)

    if env.score > record:
        record = env.score
        agent.save("best_model.pth")

    # Log compact
    print(
        f"Ep {episode:5d} | "
        f"Score: {env.score:3d} | "
        f"Record: {record:3d} | "
        f"Avg100: {avg_score:5.1f} | "
        f"ε: {agent.epsilon:.4f} | "
        f"Reward: {total_reward:7.2f}"
    )

print("Entraînement terminé.")