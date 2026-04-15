# train.py
import pygame
from snake_env import SnakeEnv
from agent import DQNAgent

env   = SnakeEnv()
agent = DQNAgent()
TARGET_UPDATE = 10
record = 0

for episode in range(1000):
    state = env.reset()
    done  = False

    while not done:

        # ⚠️ INDISPENSABLE : vider la file d'événements Pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        action                   = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        agent.train()
        state = next_state
        env.render()

    if env.score > record:
        record = env.score
        # Sauvegarder le meilleur modèle
        import torch
        torch.save(agent.model.state_dict(), "best_model.pth")

    if episode % TARGET_UPDATE == 0:
        agent.target.load_state_dict(agent.model.state_dict())

    print(f"Épisode {episode:4d} | Score: {env.score} | Record: {record} | ε: {agent.epsilon:.3f}")