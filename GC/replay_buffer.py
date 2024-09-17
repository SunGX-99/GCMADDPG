import numpy as np
import random

class ReplayBuffer(object):
    def __init__(self, size):
        """Create a Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []  # Initialize storage
        self._maxsize = int(size)
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def clear(self):
        """Clear the buffer."""
        self._storage = []
        self._next_idx = 0

    def add(self, obs_t, action, reward, obs_tp1, done):
        """Add a transition to the buffer."""
        data = (obs_t, action, reward, obs_tp1, done)

        if len(self._storage) < self._maxsize:
            if self._next_idx >= len(self._storage):
                self._storage.append(data)
            else:
                self._storage[self._next_idx] = data
        else:
            self._storage[self._next_idx] = data

        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        """Convert sampled indices to numpy arrays."""
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def make_index(self, batch_size):
        """Generate random indices for sampling."""
        if len(self._storage) == 0:
            raise ValueError("Buffer is empty.")
        return [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]

    def make_latest_index(self, batch_size):
        """Generate indices from the latest entries."""
        if len(self._storage) == 0:
            raise ValueError("Buffer is empty.")
        idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes):
        """Sample data based on indices."""
        return self._encode_sample(idxes)

    def sample(self, batch_size):
        """Sample a batch of experiences."""
        if batch_size <= 0:
            return self._encode_sample(range(len(self._storage)))
        if len(self._storage) == 0:
            raise ValueError("Buffer is empty.")
        return self._encode_sample(self.make_index(batch_size))

    def collect(self):
        """Collect all data from the buffer."""
        return self.sample(-1)
