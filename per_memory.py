import numpy as np

class SumTree:
    """Arbre binaire pour échantillonnage proportionnel en O(log n)."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree     = np.zeros(2 * capacity - 1)   # nœuds internes + feuilles
        self.data     = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.write     = 0  # pointeur circulaire

    def _propagate(self, idx, change):
        """Remonte la modification de priorité jusqu'à la racine."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """Descend dans l'arbre pour trouver la feuille correspondant à s."""
        left  = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    @property
    def total(self):
        return self.tree[0]  # racine = somme totale des priorités

    def add(self, priority, data):
        idx = self.write + self.capacity - 1  # index dans l'arbre
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        """Retourne (index_arbre, priorité, donnée) pour un tirage s."""
        idx  = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedMemory:
    """Buffer de replay prioritaire basé sur SumTree."""

    def __init__(self, capacity=100_000, alpha=0.6, beta_start=0.4, beta_frames=50_000):
        self.tree        = SumTree(capacity)
        self.capacity    = capacity
        self.alpha       = alpha        # degré de priorité (0=uniforme, 1=full)
        self.beta        = beta_start   # correction importance sampling
        self.beta_increment = (1.0 - beta_start) / beta_frames  # monte vers 1.0
        self.epsilon     = 1e-6         # évite priorité = 0
        self.max_priority = 1.0         # priorité max vue jusqu'ici

    def remember(self, state, action, reward, next_state, done):
        """Ajoute avec la priorité maximale connue (optimiste)."""
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, (state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch, indices, weights = [], [], []
        segment = self.tree.total / batch_size

        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(batch_size):
            # Tire un nombre dans chaque segment égal de la distribution
            s   = np.random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self.tree.get(s)

            # Importance sampling weight (corrige le biais d'échantillonnage)
            prob   = priority / self.tree.total
            weight = (self.tree.n_entries * prob) ** (-self.beta)

            indices.append(idx)
            weights.append(weight)
            batch.append(data)

        # Normalise les poids
        weights = np.array(weights, dtype=np.float32)
        weights /= weights.max()

        return batch, indices, weights

    def update_priorities(self, indices, td_errors):
        """Met à jour les priorités après apprentissage."""
        for idx, error in zip(indices, td_errors):
            priority = (abs(error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return self.tree.n_entries