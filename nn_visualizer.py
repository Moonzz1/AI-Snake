# nn_visualizer.py
# Affichage temps réel du réseau de neurones Dueling DQN dans une fenêtre Pygame

import pygame
import numpy as np
import torch

# ────────────────────────────────────────────────────────
# Palette de couleurs
# ────────────────────────────────────────────────────────
C_BG          = (18, 18, 24)
C_PANEL_BG    = (26, 26, 36)
C_TITLE       = (200, 200, 220)
C_LABEL       = (140, 140, 160)
C_NEURON_OFF  = (45, 45, 60)
C_POS         = (80, 200, 120)   # activation forte positive → vert
C_NEG         = (220, 80, 80)    # activation forte négative → rouge
C_LINE_POS    = (80, 180, 110, 60)   # RGBA connexion positive (alpha)
C_LINE_NEG    = (200, 70, 70, 60)    # RGBA connexion négative
C_CHOSEN      = (255, 200, 60)   # neurone de sortie sélectionné
C_ACTION      = [(80, 180, 255), (80, 255, 160), (255, 140, 80)]
C_VALUE_COL   = (160, 100, 240)  # branche Valeur
C_ADV_COL     = (240, 160, 60)   # branche Avantage

ACTION_NAMES  = ["Tout droit", "Droite", "Gauche"]
FEATURE_NAMES = [
    "Danger↑", "Danger→", "Danger←",
    "Danger↑×2","Danger→×2","Danger←×2",
    "Espace↑", "Espace→", "Espace←",
    "Atteint queue",
    "Dir →", "Dir ←", "Dir ↑", "Dir ↓",
    "Pomme ←", "Pomme →", "Pomme ↑", "Pomme ↓",
    "Dist pomme", "Taille"
]


def lerp_color(c1, c2, t):
    """Interpolation linéaire entre deux couleurs RGB."""
    t = max(0.0, min(1.0, t))
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def activation_color(val, alpha=220):
    """Retourne une couleur RGB proportionnelle à l'activation."""
    if val >= 0:
        t = min(val, 1.0)
        r, g, b = lerp_color(C_NEURON_OFF, C_POS, t)
    else:
        t = min(-val, 1.0)
        r, g, b = lerp_color(C_NEURON_OFF, C_NEG, t)
    return (r, g, b)


class NNVisualizer:
    """
    Panneau de visualisation du réseau de neurones Dueling DQN.
    Doit être dessiné sur une surface pygame distincte ou dans le même écran.
    
    Usage :
        vis = NNVisualizer(surface, x_offset=400)
        vis.update(state_vector, model, chosen_action, q_values)
        vis.draw()
    """

    def __init__(self, surface: pygame.Surface, x_offset: int = 400, width: int = 600):
        self.surf    = surface
        self.x0      = x_offset
        self.w       = width
        self.h       = surface.get_height()
        self.font_sm = pygame.font.SysFont("monospace", 11)
        self.font_md = pygame.font.SysFont("monospace", 13, bold=True)
        self.font_lg = pygame.font.SysFont("monospace", 15, bold=True)

        # Données mises à jour chaque step
        self.state        = np.zeros(20)
        self.activations  = {}   # dict couche → vecteur numpy
        self.chosen_action = 0
        self.q_values      = np.zeros(3)
        self.v_value       = 0.0
        self.advantages    = np.zeros(3)

        # Layout horizontal des colonnes (fractions de self.w)
        self.col_input_x   = self.x0 + int(self.w * 0.07)
        self.col_shared_x  = self.x0 + int(self.w * 0.28)
        self.col_val_x     = self.x0 + int(self.w * 0.52)
        self.col_adv_x     = self.x0 + int(self.w * 0.72)
        self.col_out_x     = self.x0 + int(self.w * 0.93)

    # ─────────────────────────────────────────────────────
    # API publique
    # ─────────────────────────────────────────────────────

    def update(self, state: np.ndarray, model, chosen_action: int, q_values: np.ndarray):
        """Capture les activations internes du modèle pour la frame courante."""
        self.state         = state
        self.chosen_action = chosen_action
        self.q_values      = q_values

        model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(state).unsqueeze(0)

            # Activations du tronc commun, couche par couche
            h = x
            shared_acts = []
            for layer in model.shared:
                h = layer(h)
                if isinstance(layer, torch.nn.ReLU):
                    shared_acts.append(h.squeeze(0).numpy())

            self.activations["shared_1"] = shared_acts[0] if len(shared_acts) > 0 else np.zeros(256)
            self.activations["shared_2"] = shared_acts[1] if len(shared_acts) > 1 else np.zeros(256)

            # Branche valeur
            hv = h
            for layer in model.value:
                hv = layer(hv)
            self.v_value = hv.item()

            # Branche avantage
            ha = h
            for layer in model.advantage:
                ha = layer(ha)
            self.advantages = ha.squeeze(0).numpy()

        model.train()

    def draw(self):
        """Dessine le panneau entier sur self.surf."""
        panel = pygame.Rect(self.x0, 0, self.w, self.h)
        pygame.draw.rect(self.surf, C_PANEL_BG, panel)

        # Titre
        self._text("Réseau de Neurones", self.x0 + self.w // 2, 14,
                   self.font_lg, C_TITLE, center=True)

        y_top    = 36
        y_bottom = self.h - 10

        self._draw_input_column(y_top, y_bottom)
        self._draw_shared_columns(y_top, y_bottom)
        self._draw_value_column(y_top, y_bottom)
        self._draw_adv_column(y_top, y_bottom)
        self._draw_output_column(y_top, y_bottom)
        self._draw_connections(y_top, y_bottom)
        self._draw_stats_bar()

    # ─────────────────────────────────────────────────────
    # Colonnes individuelles
    # ─────────────────────────────────────────────────────

    def _draw_input_column(self, y_top, y_bottom):
        n      = 20
        radius = 8
        cx     = self.col_input_x
        self._text("Entrée", cx, y_top - 6, self.font_md, C_LABEL, center=True)
        self._text("(20)", cx, y_top + 8, self.font_sm, C_LABEL, center=True)

        positions = self._neuron_positions(n, cx, y_top + 22, y_bottom - 80, radius)
        self._neuron_positions_cache = {"input": positions}

        for i, (px, py) in enumerate(positions):
            val  = float(self.state[i])
            col  = activation_color(val)
            pygame.draw.circle(self.surf, col, (px, py), radius)
            pygame.draw.circle(self.surf, C_LABEL, (px, py), radius, 1)

            # Étiquette à droite du neurone
            label = FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else str(i)
            label_surf = self.font_sm.render(label, True, C_LABEL)
            self.surf.blit(label_surf, (px + radius + 3, py - 6))

    def _draw_shared_columns(self, y_top, y_bottom):
        """Affiche les 2 couches du tronc commun sous forme de 32 neurones (sous-échantillon)."""
        vis_n  = 32   # on ne montre que 32/256 neurones
        radius = 6
        cols   = [
            ("Couche 1", "shared_1", self.col_shared_x - 22),
            ("Couche 2", "shared_2", self.col_shared_x + 22),
        ]
        self._text("Tronc commun (×256)", self.col_shared_x, y_top - 6,
                   self.font_md, C_LABEL, center=True)

        cached = {}
        for label, key, cx in cols:
            acts = self.activations.get(key, np.zeros(256))
            # Sous-échantillon régulier : 1 neurone sur 8
            step     = max(1, len(acts) // vis_n)
            sub_acts = acts[::step][:vis_n]

            positions = self._neuron_positions(vis_n, cx, y_top + 22, y_bottom - 80, radius)
            cached[key] = positions

            for i, (px, py) in enumerate(positions):
                val = float(sub_acts[i]) if i < len(sub_acts) else 0.0
                # Normalise entre -1 et 1 pour la couleur
                val_norm = np.tanh(val / (np.std(sub_acts) + 1e-8))
                col = activation_color(val_norm)
                pygame.draw.circle(self.surf, col, (px, py), radius)
                pygame.draw.circle(self.surf, C_LABEL, (px, py), radius, 1)

        self._neuron_positions_cache["shared_1"] = cached.get("shared_1", [])
        self._neuron_positions_cache["shared_2"] = cached.get("shared_2", [])

    def _draw_value_column(self, y_top, y_bottom):
        cx     = self.col_val_x
        radius = 9
        self._text("Valeur V(s)", cx, y_top - 6, self.font_md, C_VALUE_COL, center=True)

        mid_y = (y_top + y_bottom) // 2
        val_norm = float(np.tanh(self.v_value / 10.0))
        col = activation_color(val_norm)
        pygame.draw.circle(self.surf, col, (cx, mid_y), radius)
        pygame.draw.circle(self.surf, C_VALUE_COL, (cx, mid_y), radius + 2, 2)

        label = f"V={self.v_value:+.2f}"
        self._text(label, cx, mid_y + radius + 12, self.font_sm, C_VALUE_COL, center=True)
        self._neuron_positions_cache["value"] = [(cx, mid_y)]

    def _draw_adv_column(self, y_top, y_bottom):
        cx     = self.col_adv_x
        radius = 9
        self._text("Avantage A(s,a)", cx, y_top - 6, self.font_md, C_ADV_COL, center=True)

        n = 3
        positions = self._neuron_positions(n, cx, y_top + 80, y_bottom - 80, radius)
        self._neuron_positions_cache["advantage"] = positions

        for i, (px, py) in enumerate(positions):
            val = float(self.advantages[i]) if i < len(self.advantages) else 0.0
            val_norm = float(np.tanh(val / (np.std(self.advantages) + 1e-8)))
            col = activation_color(val_norm)
            pygame.draw.circle(self.surf, col, (px, py), radius)
            pygame.draw.circle(self.surf, C_ADV_COL, (px, py), radius + 2, 2)
            self._text(f"A={val:+.2f}", px, py + radius + 10,
                       self.font_sm, C_ADV_COL, center=True)

    def _draw_output_column(self, y_top, y_bottom):
        cx     = self.col_out_x
        radius = 12
        self._text("Sortie Q(s,a)", cx, y_top - 6, self.font_md, C_TITLE, center=True)

        n = 3
        positions = self._neuron_positions(n, cx, y_top + 70, y_bottom - 90, radius)
        self._neuron_positions_cache["output"] = positions

        # Softmax pour la "confiance" visuelle
        q = self.q_values
        q_shifted = q - q.max()
        exp_q  = np.exp(np.clip(q_shifted, -10, 0))
        probs  = exp_q / (exp_q.sum() + 1e-8)

        for i, (px, py) in enumerate(positions):
            chosen = (i == self.chosen_action)
            col = C_CHOSEN if chosen else C_ACTION[i]

            # Cercle principal
            pygame.draw.circle(self.surf, col, (px, py), radius + (4 if chosen else 0))
            if chosen:
                pygame.draw.circle(self.surf, (255, 255, 255), (px, py), radius + 6, 2)

            # Barre de confiance
            bar_w = int(probs[i] * 60)
            bar_x = px - 30
            bar_y = py + radius + 8
            pygame.draw.rect(self.surf, C_NEURON_OFF, (bar_x, bar_y, 60, 7))
            pygame.draw.rect(self.surf, col, (bar_x, bar_y, bar_w, 7))

            # Labels
            self._text(ACTION_NAMES[i], px, bar_y + 12,
                       self.font_sm, col, center=True)
            self._text(f"Q={q[i]:+.2f}", px, bar_y + 24,
                       self.font_sm, col, center=True)

    def _draw_connections(self, y_top, y_bottom):
        """Dessine quelques connexions représentatives entre les colonnes."""
        cache = self._neuron_positions_cache

        # input → shared_1 (1 trait sur 5)
        if "input" in cache and "shared_1" in cache:
            for i, p1 in enumerate(cache["input"]):
                if i % 4 == 0:
                    for j, p2 in enumerate(cache["shared_1"]):
                        if j % 5 == 0:
                            val = float(self.state[i])
                            col = C_LINE_POS if val > 0 else C_LINE_NEG
                            self._draw_line_alpha(p1, p2, col)

        # shared_2 → value
        if "shared_2" in cache and "value" in cache:
            for p1 in cache["shared_2"][::3]:
                for p2 in cache["value"]:
                    self._draw_line_alpha(p1, p2, C_LINE_POS)

        # shared_2 → advantage
        if "shared_2" in cache and "advantage" in cache:
            for p1 in cache["shared_2"][::3]:
                for p2 in cache["advantage"]:
                    self._draw_line_alpha(p1, p2, C_LINE_POS)

        # value + advantage → output
        if "value" in cache and "output" in cache:
            for p1 in cache["value"]:
                for p2 in cache["output"]:
                    self._draw_line_alpha(p1, p2, (*C_VALUE_COL, 80))

        if "advantage" in cache and "output" in cache:
            for p1 in cache["advantage"]:
                for p2 in cache["output"]:
                    self._draw_line_alpha(p1, p2, (*C_ADV_COL, 80))

    def _draw_stats_bar(self):
        """Barre du bas : ε et score."""
        y = self.h - 22
        pygame.draw.line(self.surf, C_LABEL,
                         (self.x0+ 10, y - 4), (self.x0 + self.w - 10, y - 4), 1)
        self._text("ε = —   Score = —   Ep = —",
                   self.x0 + self.w // 2, y + 4,
                   self.font_sm, C_LABEL, center=True)

    def update_stats(self, epsilon: float, score: int, episode: int):
        """Mise à jour rapide de la ligne de stats (à appeler après draw())."""
        y = self.h - 18
        rect = pygame.Rect(self.x0 + 10, y - 2, self.w - 20, 18)
        pygame.draw.rect(self.surf, C_PANEL_BG, rect)
        label = f"ε = {epsilon:.3f}   Score = {score}   Épisode = {episode}"
        self._text(label, self.x0 + self.w // 2, y + 2,
                   self.font_sm, C_TITLE, center=True)

    # ─────────────────────────────────────────────────────
    # Utilitaires
    # ─────────────────────────────────────────────────────

    def _neuron_positions(self, n, cx, y_top, y_bottom, radius):
        if n == 1:
            return [(cx, (y_top + y_bottom) // 2)]
        step = (y_bottom - y_top) / (n - 1) if n > 1 else 0
        return [(cx, int(y_top + i * step)) for i in range(n)]

    def _text(self, text, x, y, font, color, center=False):
        surf = font.render(text, True, color)
        if center:
            rect = surf.get_rect(center=(x, y))
        else:
            rect = surf.get_rect(topleft=(x, y))
        self.surf.blit(surf, rect)

    def _draw_line_alpha(self, p1, p2, color_rgba):
        """Trace une ligne semi-transparente sur un surface temporaire."""
        r, g, b = color_rgba[:3]
        a = color_rgba[3] if len(color_rgba) == 4 else 60
        tmp = pygame.Surface(self.surf.get_size(), pygame.SRCALPHA)
        pygame.draw.line(tmp, (r, g, b, a), p1, p2, 1)
        self.surf.blit(tmp, (0, 0))