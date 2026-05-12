# snake_env.py

import pygame
import numpy as np
import random
from collections import deque

BLOCK = 20

class SnakeEnv:
    def __init__(self, w=400, h=400, headless=False):
        self.w, self.h = w, h
        self.headless = headless
        pygame.init()
        if not headless:
            self.screen = pygame.display.set_mode((w, h))
            self.clock  = pygame.time.Clock()
        else:
            # Mode sans fenêtre pour entraînements rapides
            import os; os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            pygame.display.init()
            self.screen = pygame.display.set_mode((w, h))
        self.reset()

    def reset(self):
        cx, cy = self.w // 2, self.h // 2
        self.snake      = [(cx, cy), (cx - BLOCK, cy), (cx - 2*BLOCK, cy)]
        self.direction  = (BLOCK, 0)
        self.score      = 0
        self.food       = self._place_food()
        self.frame_iter = 0
        self.prev_dist  = self._manhattan_to_food()
        return self._get_state()

    def step(self, action):
        old_head = self.snake[0]
        self._change_direction(action)
        self.snake.insert(0, self._next_head())
        self.frame_iter += 1

        reward, done = 0, False

        # Mort (collision ou timeout)
        if self._collision() or self.frame_iter > 150 * len(self.snake):
            # Pénalité graduée selon l'espace restant
            free = self._flood_fill(self.snake[0]) / ((self.w // BLOCK) * (self.h // BLOCK))
            return self._get_state(), -10 - 5 * (1 - free), True

        # Mange la pomme
        if self.snake[0] == self.food:
            self.score     += 1
            reward          = 10 + 0.5 * self.score  # récompense croissante
            self.food       = self._place_food()
            self.prev_dist  = self._manhattan_to_food()
        else:
            self.snake.pop()
            # Récompense de progression basée sur la distance
            curr_dist  = self._manhattan_to_food()
            reward     = 0.1 if curr_dist < self.prev_dist else -0.15
            self.prev_dist = curr_dist

            # Pénalité légère si l'espace libre est faible (encourage la prudence)
            head_space = self._flood_fill(self.snake[0]) / ((self.w // BLOCK) * (self.h // BLOCK))
            if head_space < 0.2:
                reward -= 0.05

        return self._get_state(), reward, done

    # ──────────────────────────────────────────
    # État : 14 features + 4 features de danger à 2 pas
    # ──────────────────────────────────────────

    def _get_state(self):
        head   = self.snake[0]
        dx, dy = self.direction
        total  = (self.w // BLOCK) * (self.h // BLOCK)

        # Positions immédiates (1 case)
        next_s = (head[0]+dx,        head[1]+dy)
        next_r = (head[0]-dy,        head[1]+dx)
        next_l = (head[0]+dy,        head[1]-dx)

        # Positions à 2 cases (anticipation)
        next_s2 = (head[0]+2*dx,     head[1]+2*dy)
        next_r2 = (head[0]-2*dy,     head[1]+2*dx)
        next_l2 = (head[0]+2*dy,     head[1]-2*dx)

        # Dangers immédiats
        danger_s  = float(self._collision_at(next_s))
        danger_r  = float(self._collision_at(next_r))
        danger_l  = float(self._collision_at(next_l))

        # Dangers à 2 cases
        danger_s2 = float(self._collision_at(next_s2))
        danger_r2 = float(self._collision_at(next_r2))
        danger_l2 = float(self._collision_at(next_l2))

        # Espace libre normalisé
        space_s = self._flood_fill(next_s) / total if not self._collision_at(next_s) else 0.0
        space_r = self._flood_fill(next_r) / total if not self._collision_at(next_r) else 0.0
        space_l = self._flood_fill(next_l) / total if not self._collision_at(next_l) else 0.0

        # Peut-on atteindre la queue depuis la tête ?
        tail = self.snake[-1]
        can_reach_tail = float(self._can_reach(head, tail))

        # Direction actuelle (one-hot)
        dir_r = float(dx > 0); dir_l = float(dx < 0)
        dir_u = float(dy < 0); dir_d = float(dy > 0)

        # Position relative de la nourriture
        food_l = float(self.food[0] < head[0]); food_r = float(self.food[0] > head[0])
        food_u = float(self.food[1] < head[1]); food_d = float(self.food[1] > head[1])

        # Distance normalisée à la nourriture
        food_dist = (abs(self.food[0]-head[0]) + abs(self.food[1]-head[1])) / (self.w + self.h)

        # Longueur normalisée du serpent
        snake_len = len(self.snake) / total

        return np.array([
            danger_s,  danger_r,  danger_l,          # 3 — danger immédiat
            danger_s2, danger_r2, danger_l2,          # 3 — danger à 2 pas (NOUVEAU)
            space_s,   space_r,   space_l,            # 3 — espace libre
            can_reach_tail,                           # 1 — peut rejoindre sa queue (NOUVEAU)
            dir_r, dir_l, dir_u, dir_d,               # 4 — direction
            food_l, food_r, food_u, food_d,           # 4 — direction nourriture
            food_dist,                                # 1 — distance normalisée (NOUVEAU)
            snake_len,                                # 1 — taille normalisée (NOUVEAU)
        ], dtype=np.float32)                          # 20 valeurs au total

    # ──────────────────────────────────────────
    # Méthodes internes
    # ──────────────────────────────────────────

    def _collision(self):
        head = self.snake[0]
        return (
            head in self.snake[1:]
            or head[0] < 0 or head[0] >= self.w
            or head[1] < 0 or head[1] >= self.h
        )

    def _collision_at(self, pos):
        return (
            pos in self.snake
            or pos[0] < 0 or pos[0] >= self.w
            or pos[1] < 0 or pos[1] >= self.h
        )

    def _flood_fill(self, start):
        """BFS — compte les cases accessibles depuis start."""
        if self._collision_at(start):
            return 0
        body    = set(self.snake)
        visited = set()
        queue   = deque([start])
        while queue:
            pos = queue.popleft()
            if pos in visited:
                continue
            visited.add(pos)
            x, y = pos
            for n in [(x+BLOCK,y),(x-BLOCK,y),(x,y+BLOCK),(x,y-BLOCK)]:
                nx, ny = n
                if n not in visited and 0 <= nx < self.w and 0 <= ny < self.h and n not in body:
                    queue.append(n)
        return len(visited)

    def _can_reach(self, start, target):
        """Vérifie si 'target' est accessible depuis 'start' par BFS."""
        if self._collision_at(start):
            return False
        body    = set(self.snake[1:])   # permet de passer là où la tête est déjà
        visited = set()
        queue   = deque([start])
        while queue:
            pos = queue.popleft()
            if pos == target:
                return True
            if pos in visited:
                continue
            visited.add(pos)
            x, y = pos
            for n in [(x+BLOCK,y),(x-BLOCK,y),(x,y+BLOCK),(x,y-BLOCK)]:
                nx, ny = n
                if n not in visited and 0 <= nx < self.w and 0 <= ny < self.h and n not in body:
                    queue.append(n)
        return False

    def _manhattan_to_food(self):
        head = self.snake[0]
        return abs(head[0]-self.food[0]) + abs(head[1]-self.food[1])

    def _next_head(self):
        hx, hy = self.snake[0]
        dx, dy = self.direction
        return (hx + dx, hy + dy)

    def _change_direction(self, action):
        dx, dy = self.direction
        if action == 1:
            self.direction = (-dy, dx)
        elif action == 2:
            self.direction = (dy, -dx)

    def _place_food(self):
        while True:
            pos = (
                random.randrange(0, self.w, BLOCK),
                random.randrange(0, self.h, BLOCK)
            )
            if pos not in self.snake:
                return pos

    def render(self):
        if self.headless:
            return
        self.screen.fill((0, 0, 0))
        # Dégradé de vert pour le corps (tête plus claire)
        for i, seg in enumerate(self.snake):
            green = max(80, 200 - i * 3)
            pygame.draw.rect(self.screen, (0, green, 0), (*seg, BLOCK, BLOCK))
        pygame.draw.rect(self.screen, (220, 50, 50), (*self.food, BLOCK, BLOCK))
        pygame.display.flip()
        self.clock.tick(60)