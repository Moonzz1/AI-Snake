# train.py

import pygame
import torch
import numpy as np
from collections import deque
from snake_env import SnakeEnv
from agent import DQNAgent
from nn_visualizer import NNVisualizer

# ────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────
EPISODES     = 2000
HEADLESS     = False   # False = fenêtre + visualiseur activé
VIS_WIDTH    = 600     # largeur du panneau neurones (0 = désactivé)
RENDER_EVERY = 1       # 1 = toujours afficher; augmenter pour accélérer

env   = SnakeEnv(headless=HEADLESS, vis_width=VIS_WIDTH)
agent = DQNAgent()

# Attache le visualiseur si le rendu est actif
if not HEADLESS and VIS_WIDTH > 0:
    vis = NNVisualizer(env.screen, x_offset=env.w, width=VIS_WIDTH)
    env.set_visualizer(vis)

record       = 0
score_window = deque(maxlen=100)

for episode in range(1, EPISODES + 1):
    state = env.reset()
    done  = False
    total_reward = 0

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                agent.save("best_model.pth")
                pygame.quit()
                quit()
            # Touche V : bascule le panneau neurones
            if event.type == pygame.KEYDOWN and event.key == pygame.K_v:
                if env.visualizer:
                    env.visualizer = None
                elif VIS_WIDTH > 0:
                    env.visualizer = NNVisualizer(env.screen, x_offset=env.w, width=VIS_WIDTH)

        # Action + Q-values pour le visualiseur
        state_t  = torch.FloatTensor(state).unsqueeze(0)
        agent.model.eval()
        with torch.no_grad():
            q_vals = agent.model(state_t).squeeze(0).numpy()
        agent.model.train()

        action = agent.choose_action(state)

        next_state, reward, done = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        agent.train()

        if not HEADLESS and episode % RENDER_EVERY == 0:
            env.render(
                agent=agent,
                state=state,
                action=action,
                q_values=q_vals,
                episode=episode,
            )

        state        = next_state
        total_reward += reward

    # Fin d'épisode
    agent.step_scheduler()
    score_window.append(env.score)
    avg_score = np.mean(score_window)

    if env.score > record:
        record = env.score
        agent.save("best_model.pth")

    print(
        f"Ep {episode:5d} | "
        f"Score: {env.score:3d} | "
        f"Record: {record:3d} | "
        f"Avg100: {avg_score:5.1f} | "
        f"ε: {agent.epsilon:.4f} | "
        f"Reward: {total_reward:7.2f}"
    )

print("Entraînement terminé.")