"""
Abstract reward shaping interface and utilities for SymbolicGym.
"""
from abc import ABC, abstractmethod


class RewardShaping(ABC):
    """Abstract reward shaping interface."""

    @abstractmethod
    def compute(self, state, action, next_state, info):
        pass

    @abstractmethod
    def shaping_weights(self):
        pass

    @abstractmethod
    def anneal(self, step):
        pass


class LinearAnnealingReward(RewardShaping):
    """
    Anneals reward from dense to sparse over time.
    """

    def __init__(self, dense_weight=1.0, sparse_weight=0.0, anneal_steps=10000):
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.anneal_steps = anneal_steps
        self.current_step = 0

    def compute(self, state, action, next_state, info):
        # Example: weighted sum of dense and sparse rewards
        dense = info.get("dense_reward", 0.0)
        sparse = info.get("sparse_reward", 0.0)
        w = self.dense_weight * max(0, 1 - self.current_step / self.anneal_steps)
        w_sparse = self.sparse_weight + (1 - w)
        return w * dense + w_sparse * sparse

    def shaping_weights(self):
        return {
            "dense": self.dense_weight,
            "sparse": self.sparse_weight,
            "anneal_steps": self.anneal_steps,
        }

    def anneal(self, step=1):
        self.current_step += step


class LinearAnnealingRewardShaping(RewardShaping):
    """Reward shaping with linear annealing between dense and sparse rewards."""

    def __init__(self, dense_weight=1.0, sparse_weight=1.0, anneal_steps=10000):
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.anneal_steps = anneal_steps
        self.current_step = 0

    def compute(self, state, action, next_state, info):
        # Accept both 'dense'/'sparse' and 'dense_reward'/'sparse_reward' keys for compatibility
        dense = info.get("dense_reward", info.get("dense", 0.0))
        sparse = info.get("sparse_reward", info.get("sparse", 0.0))
        w = self.dense_weight * max(0, 1 - self.current_step / self.anneal_steps)
        w_sparse = self.sparse_weight + (1 - w)
        return w * dense + w_sparse * sparse

    def get_shaping_weights(self):
        # For compatibility with test, use 'dense_weight' and 'sparse_weight' keys
        return {
            "dense_weight": self.dense_weight,
            "sparse_weight": self.sparse_weight,
            "anneal_steps": self.anneal_steps,
        }

    def shaping_weights(self):
        # For compatibility with abstract base class and legacy code
        return self.get_shaping_weights()

    def compute_reward(self, state, action, next_state, info):
        # Alias for compute, for compatibility with test usage
        return self.compute(state, action, next_state, info)

    def anneal(self, step=1):
        self.current_step += step
