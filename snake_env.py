# snake_env.py

import pygame
import numpy as np
import random

BLOCK = 20

class SnakeEnv:
    def __init__(self, w=400, h=400):
        self.w, self.h = w, h
        pygame.init()
        self.screen = pygame.display.set_mode((w, h))
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        cx, cy = self.w // 2, self.h // 2
        self.snake = [(cx, cy), (cx - BLOCK, cy), (cx - 2*BLOCK, cy)]
        self.direction = (BLOCK, 0)
        self.score = 0
        self.food = self._place_food()
        self.frame_iter = 0
        return self._get_state()         # ← appelé ici

    def step(self, action):
        self._change_direction(action)
        self.snake.insert(0, self._next_head())
        self.frame_iter += 1

        reward, done = 0, False
        if self._collision() or self.frame_iter > 100 * len(self.snake):
            return self._get_state(), -10, True

        if self.snake[0] == self.food:
            self.score += 1
            reward = 10
            self.food = self._place_food()
        else:
            self.snake.pop()

        return self._get_state(), reward, done  # ← appelé ici aussi

    # ──────────────────────────────────────────
    # Méthodes internes (préfixe _ par convention)
    # ──────────────────────────────────────────

    def _get_state(self):
        head      = self.snake[0]
        dx, dy    = self.direction
        total     = (self.w // BLOCK) * (self.h // BLOCK)  # nb total de cases

        # Cases candidates dans chaque direction
        next_straight = (head[0]+dx,  head[1]+dy)
        next_right    = (head[0]-dy,  head[1]+dx)
        next_left     = (head[0]+dy,  head[1]-dx)

        # Dangers immédiats (booléens)
        danger_s = self._collision_at(next_straight)
        danger_r = self._collision_at(next_right)
        danger_l = self._collision_at(next_left)

        # Espace libre normalisé (0.0 → piège total, 1.0 → tout ouvert)
        space_s = self._flood_fill(next_straight) / total
        space_r = self._flood_fill(next_right)    / total
        space_l = self._flood_fill(next_left)     / total

        # Direction actuelle (one-hot)
        dir_r = dx > 0;  dir_l = dx < 0
        dir_u = dy < 0;  dir_d = dy > 0

        # Position relative de la nourriture
        food_l = self.food[0] < head[0];  food_r = self.food[0] > head[0]
        food_u = self.food[1] < head[1];  food_d = self.food[1] > head[1]

        return np.array([
            danger_s, danger_r, danger_l,         # 3 booléens
            space_s,  space_r,  space_l,          # 3 flottants ← NOUVEAU
            dir_r, dir_l, dir_u, dir_d,           # 4 booléens
            food_l, food_r, food_u, food_d        # 4 booléens
        ], dtype=float)                            # 14 valeurs au total

    def _collision(self):
        head = self.snake[0]
        return (
            head in self.snake[1:]                      # mord son corps
            or head[0] < 0 or head[0] >= self.w        # mur gauche/droit
            or head[1] < 0 or head[1] >= self.h        # mur haut/bas
        )

    def _collision_at(self, pos):
        return (
            pos in self.snake
            or pos[0] < 0 or pos[0] >= self.w
            or pos[1] < 0 or pos[1] >= self.h
        )
    
    def _flood_fill(self, start):
        """Compte les cases accessibles en BFS depuis 'start'."""
        # Si le point de départ est déjà un mur, retourne 0
        if self._collision_at(start):
            return 0

        body    = set(self.snake)       # tout le corps est un obstacle
        visited = set()
        queue   = [start]

        while queue:
            pos = queue.pop(0)
            if pos in visited:
                continue
            x, y = pos
            # Vérifie les 4 cases voisines
            for voisin in [(x+BLOCK, y), (x-BLOCK, y), (x, y+BLOCK), (x, y-BLOCK)]:
                vx, vy = voisin
                if (voisin not in visited
                    and 0 <= vx < self.w
                    and 0 <= vy < self.h
                    and voisin not in body):
                    queue.append(voisin)
            visited.add(pos)

        return len(visited)

    def _next_head(self):
        hx, hy = self.snake[0]
        dx, dy = self.direction
        return (hx + dx, hy + dy)

    def _change_direction(self, action):
        # 0 = tout droit, 1 = tourner droite, 2 = tourner gauche
        dx, dy = self.direction
        if action == 1:
            self.direction = (-dy, dx)   # rotation +90°
        elif action == 2:
            self.direction = (dy, -dx)   # rotation -90°

    def _place_food(self):
        while True:
            pos = (
                random.randrange(0, self.w, BLOCK),
                random.randrange(0, self.h, BLOCK)
            )
            if pos not in self.snake:
                return pos

    def render(self):
        self.screen.fill((0, 0, 0))
        for seg in self.snake:
            pygame.draw.rect(self.screen, (0, 200, 0), (*seg, BLOCK, BLOCK))
        pygame.draw.rect(self.screen, (200, 0, 0), (*self.food, BLOCK, BLOCK))
        pygame.display.flip()
        self.clock.tick(60)