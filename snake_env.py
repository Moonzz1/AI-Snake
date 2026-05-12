# snake_env.py

import pygame
import numpy as np
import random
from collections import deque

BLOCK = 20
GAME_W = 400
GAME_H = 400

class SnakeEnv:
    def __init__(self, w=GAME_W, h=GAME_H, headless=False, vis_width=600):
        """
        w, h        : taille du plateau de jeu
        headless    : True = pas de rendu (entraînement rapide)
        vis_width   : largeur du panneau visualiseur (0 = désactivé)
        """
        self.w, self.h       = w, h
        self.headless        = headless
        self.vis_width       = vis_width if not headless else 0
        self.total_width     = w + self.vis_width
        self.visualizer      = None

        pygame.init()
        if not headless:
            self.screen = pygame.display.set_mode((self.total_width, h))
            pygame.display.set_caption("AI Snake — Visualisation du réseau")
            self.clock  = pygame.time.Clock()
        else:
            import os; os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            pygame.display.init()
            self.screen = pygame.display.set_mode((w, h))

        # Sous-surface dédiée au jeu
        self.game_surf = pygame.Surface((w, h))

        self.reset()

    def set_visualizer(self, vis):
        """Attache le visualiseur de neurones (NNVisualizer)."""
        self.visualizer = vis

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

        if self._collision() or self.frame_iter > 150 * len(self.snake):
            free = self._flood_fill(self.snake[0]) / ((self.w // BLOCK) * (self.h // BLOCK))
            return self._get_state(), -10 - 5 * (1 - free), True

        if self.snake[0] == self.food:
            self.score    += 1
            reward         = 10 + 0.5 * self.score
            self.food      = self._place_food()
            self.prev_dist = self._manhattan_to_food()
        else:
            self.snake.pop()
            curr_dist  = self._manhattan_to_food()
            reward     = 0.1 if curr_dist < self.prev_dist else -0.15
            self.prev_dist = curr_dist
            head_space = self._flood_fill(self.snake[0]) / ((self.w // BLOCK) * (self.h // BLOCK))
            if head_space < 0.2:
                reward -= 0.05

        return self._get_state(), reward, done

    # ──────────────────────────────────────────

    def _get_state(self):
        head   = self.snake[0]
        dx, dy = self.direction
        total  = (self.w // BLOCK) * (self.h // BLOCK)

        next_s  = (head[0]+dx,     head[1]+dy)
        next_r  = (head[0]-dy,     head[1]+dx)
        next_l  = (head[0]+dy,     head[1]-dx)
        next_s2 = (head[0]+2*dx,   head[1]+2*dy)
        next_r2 = (head[0]-2*dy,   head[1]+2*dx)
        next_l2 = (head[0]+2*dy,   head[1]-2*dx)

        danger_s  = float(self._collision_at(next_s))
        danger_r  = float(self._collision_at(next_r))
        danger_l  = float(self._collision_at(next_l))
        danger_s2 = float(self._collision_at(next_s2))
        danger_r2 = float(self._collision_at(next_r2))
        danger_l2 = float(self._collision_at(next_l2))

        space_s = self._flood_fill(next_s) / total if not self._collision_at(next_s) else 0.0
        space_r = self._flood_fill(next_r) / total if not self._collision_at(next_r) else 0.0
        space_l = self._flood_fill(next_l) / total if not self._collision_at(next_l) else 0.0

        tail = self.snake[-1]
        can_reach_tail = float(self._can_reach(head, tail))

        dir_r = float(dx > 0); dir_l = float(dx < 0)
        dir_u = float(dy < 0); dir_d = float(dy > 0)

        food_l = float(self.food[0] < head[0]); food_r = float(self.food[0] > head[0])
        food_u = float(self.food[1] < head[1]); food_d = float(self.food[1] > head[1])

        food_dist = (abs(self.food[0]-head[0]) + abs(self.food[1]-head[1])) / (self.w + self.h)
        snake_len = len(self.snake) / total

        return np.array([
            danger_s, danger_r, danger_l,
            danger_s2, danger_r2, danger_l2,
            space_s, space_r, space_l,
            can_reach_tail,
            dir_r, dir_l, dir_u, dir_d,
            food_l, food_r, food_u, food_d,
            food_dist, snake_len,
        ], dtype=np.float32)

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
        if self._collision_at(start):
            return False
        body    = set(self.snake[1:])
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

    def render(self, agent=None, state=None, action=None, q_values=None, episode=0):
        """
        Dessine le jeu + éventuellement le panneau neurones.
        Paramètres optionnels pour la visualisation du réseau.
        """
        if self.headless:
            return

        # ── Jeu ──
        self.game_surf.fill((10, 10, 15))
        # Grille légère
        for gx in range(0, self.w, BLOCK):
            pygame.draw.line(self.game_surf, (20, 20, 28), (gx, 0), (gx, self.h))
        for gy in range(0, self.h, BLOCK):
            pygame.draw.line(self.game_surf, (20, 20, 28), (0, gy), (self.w, gy))

        # Serpent avec dégradé
        for i, seg in enumerate(self.snake):
            green = max(80, 210 - i * 4)
            pygame.draw.rect(self.game_surf, (30, green, 60), (*seg, BLOCK-1, BLOCK-1))

        # Pomme avec halo
        fx, fy = self.food
        pygame.draw.circle(self.game_surf, (180, 30, 30),
                           (fx + BLOCK//2, fy + BLOCK//2), BLOCK//2 + 3)
        pygame.draw.circle(self.game_surf, (240, 70, 70),
                           (fx + BLOCK//2, fy + BLOCK//2), BLOCK//2)

        # Score
        font = pygame.font.SysFont("monospace", 16, bold=True)
        sc = font.render(f"Score: {self.score}", True, (200, 200, 200))
        self.game_surf.blit(sc, (8, 8))

        self.screen.blit(self.game_surf, (0, 0))

        # ── Visualiseur neurones ──
        if self.visualizer and agent is not None and state is not None:
            self.visualizer.update(state, agent.model, action or 0, q_values if q_values is not None else np.zeros(3))
            self.visualizer.draw()
            self.visualizer.update_stats(agent.epsilon, self.score, episode)

        pygame.display.flip()
        self.clock.tick(60)